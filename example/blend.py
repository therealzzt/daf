# =============================================================================
# CacheBlend 最小示例：演示"复用缓存 KV" vs "全量重算"的速度与质量对比
#
# 整体流程（对 inputs/ 下 10 个样本逐个执行）：
#   阶段一：把文档逐块喂给模型，从每层网络里"偷"出 KV 缓存拼起来（collect 模式）
#   阶段二：打开缓存融合开关跑推理，只重算少量偏差大的 token（check 模式）
#   阶段三：关掉开关正常跑一遍完整 prefill，作为对照组
# =============================================================================

# vLLM 的推理引擎类 和 采样参数类（控制生成方式）
from vllm import LLM, SamplingParams
# PyTorch：本脚本用它的 clone()/cat() 操作 KV 张量
import torch
# 读取 inputs/ 目录下的 JSON 测试样本
import json
# HuggingFace 的分词器：负责"文字 <-> token 编号"的相互转换
from transformers import AutoTokenizer

# --- 加载推理引擎（把模型权重装进显卡）---
llm = LLM(model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",  # 模型名：Mistral-7B 指令版的 AWQ 4bit 量化版（原版 bf16 要 ~15GB 显存，8GB 显卡装不下，量化后只要 ~4GB）
          quantization="awq",                            # 量化算法：AWQ（Activation-aware Weight Quantization，按激活分布感知地压缩权重到 4bit）
          gpu_memory_utilization=0.9,                    # 显存占用上限比例：vLLM 最多用显卡总显存的 90%（8GB × 0.9 ≈ 7.2GB，含权重+KV 缓存）
          max_model_len=4096,                            # 最大上下文长度（token 数）：限制单次输入最长 4096，越小 KV 缓存越省显存
          enforce_eager=True,                            # 关闭 CUDA Graph 优化：CUDA Graph 会把计算图录制下来加速，但要多占几百 MB 显存；显存紧张所以关掉
          )
# 单独加载分词器（与模型同名仓库）
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
# 强制让引擎使用我们手里这个分词器实例，保证脚本里 encode 的结果和引擎内部完全一致
llm.set_tokenizer(tokenizer)

#TODO (Jiayi): fix last len   # 原作者(Jiayi)留下的待办：last_len 的计算有 bug（已在第 52 行修复，见该行注释）


# ============ 主循环：依次处理 inputs/1.json ~ inputs/10.json 共 10 个样本 ============
for sample_idx in range(1,11):
    # 打开第 sample_idx 个样本文件
    f = open(f"inputs/{sample_idx}.json")
    # 解析 JSON，得到字典 ex（example 的缩写）
    ex = json.load(f)
    # 这个样本里有几篇文档块（chunk = 检索回来的一段文档）
    chunk_num = ex['chunk_num']
    # 按编号 "0","1","2"... 取出所有文档块文本，组成列表
    doc_prompts = [ex[f'{i}'] for i in range(chunk_num)]
    # 取出用户问题（query prompt）
    q_prompt = ex['query']
    # 把每篇文档转成 token 编号列表；[1:] 是切掉 encode 自动加在开头
    # 的 BOS 起始符（<s>，编号 1）——因为后面拼接时我们会自己加，不去掉就会重复
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    # 同理，把问题转成 token 编号，同样切掉 BOS
    q_ids = tokenizer.encode(q_prompt)[1:]


    # Create a sampling params object.
    # 采样参数：temperature=0 表示贪心解码（每步选概率最大的 token，结果确定可复现）；
    # max_tokens=1 表示只生成 1 个 token——本阶段（collect）只想要 prefill 的副产物
    # （KV 缓存），并不关心生成内容，所以生成 1 个就停，最省时间
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Create an tokenizer and LLM.
    # 往引擎肚子里掏出一个叫 cache_fuse_metadata 的字典（cache_fuse = 缓存融合，
    # 即 CacheBlend 的核心机制；metadata = 元数据/控制开关集合）。
    # 这一长串是在逐层深入引擎内部对象：引擎→执行器→工作进程→模型运行器→模型，
    # 正常用 vLLM 绝不这么写，这是研究原型直接操控内部状态的野路子
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    # 两个总开关先都关掉，确保引擎处于"普通模式"
    cache_fuse_metadata['collect'] = False  # collect=True 时模型每层会把 KV 存到 hack_kv 里
    cache_fuse_metadata['check'] = False    # check=True 时启用 CacheBlend 的选择性重算

    # --- 拼装 Mistral 指令模板对应的特殊 token ---
    # s_start_full：完整指令前缀的 token 编号，相当于 "<s>[INST]"（s = start，
    # full = 带 BOS 起始符的完整版）。Mistral 的 instruct 模板要求输入以它开头
    s_start_full = [733, 4138, 28793]
    # 前缀"长度"：实际 token 数 + 1。+1 是因为引擎内部 encode 时会自动再加一个 BOS，
    # 所以存下来的 KV 比我们的 token 列表多出开头 1 个位置，长度要算上它
    s_start_len = len(s_start_full) + 1

    #s_start = [518, 25580, 29962]   # 注释掉的旧值：这是 LLaMA-2 的 "[INST]" token 编号，作者早期用 LLaMA 调试的遗留
    # s_start：每个文档块自己的前缀，这里是空列表（文档块前不加任何包装）
    s_start = []
    # s_start_1_len：块前缀在 KV 里的占位长度 = 0 + 1（同样是引擎自动加的那个 BOS）。
    # 命名里的 "_1" 是作者为了和上面的 s_start_len 区分
    s_start_1_len = len(s_start) + 1

    # 先赋值为 LLaMA-2 的 "[/INST]" token 编号（作者早期用 LLaMA 调试的遗留），
    # 下一行立刻被 Mistral 的版本覆盖——这一行实际不起作用，但保留原样
    s_end = [518, 29914, 25580, 29962]
    # s_end：指令结束标记的 token 编号，相当于 "[/INST]"（s = sequence/end，序列结尾标记）
    s_end = [733, 28748, 16289, 28793]
    # s_end 的长度（后面其实没用到，属于冗余变量）
    s_end_len = len(s_end)
    # 旧的（缓存的）KV，先初始化为空列表占位（old_kvs = old Key-Values）
    old_kvs = []

    # --- 组装本样本的完整输入结构 ---
    # 给每个文档块前面加上块前缀（这里 s_start 是空的，等于没加）
    doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
    # 在最前面插入指令前缀块：[前缀, 文档1, 文档2, ..., 文档N]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    # 在末尾追加"问题块"：[前缀, 文档1, ..., 文档N, 问题+结束标记]
    doc_chunk_ids = doc_chunk_ids + [s_start+q_ids+s_end]

    # last_len：末尾"问题+结束标记"这段的 token 数。
    # 后面会作为 suffix_len（suffix = 后缀）告诉 CacheBlend：输入末尾这么长的
    # 一段是"问题"，重算时必须保住它。
    # 【已修复的 bug】原作者写的是 len([q_ids+s_end])——外面多套了一层列表，
    # len 恒等于 1，导致问题段没被保护；正确写法是 len(q_ids+s_end)
    last_len = len(q_ids+s_end)

    # 打开"收集模式"，准备进入阶段一
    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata["check"] = False

    # Mistral-7B 的 Transformer 层数。
    # 原来写死 32；改成从引擎里动态数出来，换别的模型也能用
    num_layer = len(llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)
    # 大柜子：用来按层存放拼接好的 KV。
    # 结构是 32 层 × [K, V]，命名沿用 transformers 库的经典术语 past_key_values（过往的 K/V）
    chunk_past_key_values = []

    # Concatenate old KVs
    # ============ 阶段一：逐块喂给模型，把每层的 KV"偷"出来拼起来 ============
    for i in range(len(doc_chunk_ids)):
        # 把 token 编号转回文字（引擎的 generate 接口吃字符串）
        prompts = [tokenizer.decode(doc_chunk_ids[i])]
        # 跑这一块。因为 max_tokens=1 且 collect=True，这次调用真正的目的是：
        # 让 prefill 发生，从而每层把这一块的 KV 存进 hack_kv
        llm.generate(prompts, sampling_params)

        # 掏出模型的 32 层（又是一长串内部访问）
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        # 逐层收割 KV
        for j in range(num_layer):
            # hack_kv：作者在模型代码里开的"后门"，每层 attention 算完会把
            # 本块的 K 和 V 克隆一份存在这里（hack 是作者自嘲的命名）。
            # 注意：存的是 RoPE 旋转位置编码【之前】的 K——位置信息先不写入，
            # 等在线推理时按 token 的实际位置再补旋转，这样块换顺序也不会错
            past_key_values = llm_layers[j].self_attn.hack_kv
            if i == 0:
                # 第 0 块是指令前缀：只保留前 s_start_len 个位置的 KV
                #（BOS + 前缀 token）。temp_k/temp_v = 本层临时的 K/V
                temp_k = past_key_values[0][:s_start_len].clone() # do not chage with s_start_1
                temp_v = past_key_values[1][:s_start_len].clone()
            else:
                # 后续块：切掉开头 1 个 BOS 位置（s_start_1_len=1），
                # 保留正文的 KV。右端 len(...)+1 也是因为 KV 里多了开头那个 BOS，
                # 正文整体右移了一位
                temp_k = past_key_values[0][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
                temp_v = past_key_values[1][s_start_1_len:len(doc_chunk_ids[i])+1].clone()

            if i == 0:
                # 处理第 0 块时，为每一层在柜子里新建一格 [K, V]
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                #pdb.set_trace()   # 作者调试用的断点，已注释掉
                # 之后每块：把新 KV 追加（cat = 拼接）到对应层已有 KV 的末尾，
                # dim=0 表示沿 token 序列维度拼。柜子里逐渐拼出"前缀+文档1+...+文档N+问题"的完整笔记
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=0)
        #print(temp_k.shape[0])
        # 每喂完一块，就把当前拼好的柜子写回模型的 old_kvs 字段——
        # 阶段二（check 模式）的注意力层会从这里读取缓存笔记
        llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values

    # --- 重新拼出完整的输入 token 序列（用于阶段二/三的实际推理）---
    input_ids = []

    for i in range(len(doc_chunk_ids)):
        if i == 0:
            # 第 0 块（指令前缀）原样保留
            temp_ids = doc_chunk_ids[i]
        else:
            # 其余块切掉第 1 个 token——因为序列开头的 BOS 只能出现一次，
            # 各块 decode 时被引擎加过的 BOS 在这里要去掉，否则会重复
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
        # 逐块拼成一条完整的输入
        input_ids += temp_ids

    # 转回文字形式，这就是最终要推理的完整 prompt：[前缀] 文档1 文档2 ... 文档N 问题 [结束]
    input_prompt = tokenizer.decode(input_ids)


    # ============ 阶段二：CacheBlend 缓存融合推理 ============
    # 这次要真正生成答案了，max_tokens=10（最多生成 10 个 token）
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    # check=True：打开 CacheBlend 的选择性重算——模型会复用 old_kvs 里的缓存，
    # 只挑出"新鲜值与缓存值偏差最大"的一小部分 token（默认 16%）重算，
    # 把跨文档注意力补回来
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = False
    # suffix_len：告诉注意力层，输入末尾这么长的"问题段"必须无条件重算
    #（问题从来没被缓存过，且是生成答案的关键）
    cache_fuse_metadata['suffix_len'] = last_len

    # ===== DAF 开关 =====
    # daf=True：启用 DAF 算法；改成 False 就退回原版 CacheBlend（用于对照实验）
    cache_fuse_metadata["daf"] = True
    # sel_layers：打分层，DAF 论文说聚合第 2~4 层的偏差信号，
    # Python 从 0 数层，所以是 [1,2,3]。选择层自动定为 max(sel_layers)+1 = 第 4 层
    cache_fuse_metadata["sel_layers"] = [1, 2, 3]
    # 重要 token 比例 r：论文实验用 0.3；它不是固定常数，可按任务调
    cache_fuse_metadata["recomp_ratio"] = 0.3
    # 同步旧字段 check_layers，避免两套配置互相矛盾
    #（daf 分支实际用 sel_layers 排班，这里只是防止旧代码读到不一致的值）
    cache_fuse_metadata["check_layers"] = [max(cache_fuse_metadata["sel_layers"]) + 1]
    # 用缓存跑推理
    output = llm.generate([input_prompt], sampling_params)
    # 打印缓存模式生成的答案
    print(f"Cached generation: {output[0].outputs[0].text}")
    # 打印缓存模式的 TTFT：第一个 token 生成时刻 - 请求开始调度时刻，
    # 即"从提交到出第一个字"的等待时间
    print(f"TTFT with cache: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")

    # ============ 阶段三：对照组——全量 prefill ============
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    # 关掉所有缓存开关，模型老老实实地把整段输入从头算一遍
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata['collect'] = False
    output = llm.generate([input_prompt], sampling_params)


    # 打印全量重算的答案
    print(f"Normal generation: {output[0].outputs[0].text}")
    # 打印全量重算的 TTFT——和上面的 TTFT with cache 一比，加速比就出来了
    print(f"TTFT with full prefill: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    # 样本之间的分隔线
    print("------------")

