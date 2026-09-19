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
# 读环境变量：用来切换 DAF 开关（DAF=0 跑原版 CacheBlend 对照，默认 1 跑 DAF）
import os
# HuggingFace 的分词器：负责"文字 <-> token 编号"的相互转换
from transformers import AutoTokenizer

# --- 加载推理引擎（把模型权重装进显卡）---
# 模型预设：BLEND_MODEL_PRESET=bf16（默认，6k–10k 档用）或 awq（30k 档用）。
# 30k 档必须用 AWQ：bf16 权重 15GB 在 24GB 卡上 KV 池只有 ~1.9 万 token，
# 装不下 32k 的 max_model_len；AWQ 权重 ~4GB，池子 ~10 万 token，宽裕
if os.environ.get("BLEND_MODEL_PRESET", "bf16") == "awq":
    llm = LLM(model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",  # Mistral-7B 指令版 AWQ 4bit 量化版（~4GB）
              quantization="awq",                            # 量化算法：AWQ（按激活分布感知地把权重压到 4bit）
              # 显存上限只给 75%（18GB）：vllm 预算之外，blend.py 还私占一份 ~4GB 的
              # old_kvs 离线缓存（30k token × 128KB，AWQ 只压权重、KV 仍是 bf16）。
              # 若给 90%（21.6GB），峰值 ~26GB > 24GB，collect 阶段必 OOM
              gpu_memory_utilization=0.75,
              max_model_len=32768,                           # 开满 Mistral-7B-v0.2 原生上限，覆盖 30k 档样本
              enforce_eager=True,                            # 关闭 CUDA Graph（省几百 MB 显存）
              )
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
elif os.environ.get("BLEND_MODEL_PRESET", "bf16") == "qwen":
    # Qwen2.5-7B-Instruct（DAF 论文用的模型）：28 层、GQA 28 Q 头 / 4 KV 头、
    # head_size 128。BF16 权重 ~15GB；max_model_len 默认 8192 覆盖 2–4k 档样本，
    # 长档（如 samsum 9k）用 BLEND_MAX_MODEL_LEN 覆盖（Qwen2.5 不加 YaRN 时
    # 上限 32768，勿超过；24GB 卡 BF16 KV 池约 9 万 token，16384 宽裕）
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct",
              gpu_memory_utilization=0.85,
              max_model_len=int(os.environ.get("BLEND_MAX_MODEL_LEN", "8192")),
              enforce_eager=True,
              )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
else:
    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",  # 模型名：Mistral-7B 指令版官方 bf16 非量化版（~15GB 权重，与 CacheBlend 论文实验配置一致）
              gpu_memory_utilization=0.85,                   # 显存占用上限比例：vLLM 最多用显卡总显存的 85%（24GB × 0.85 ≈ 20.4GB，含权重+KV 缓存）
              max_model_len=16384,                           # 最大上下文长度（token 数）：本档样本最长约 9.6k，16384 留足余量即可。注意 vllm 0.4.1 要求 max_model_len ≤ KV 池容量；bf16 权重 15GB 在 24GB 卡上池子约 1.9 万 token，设 32768（模型原生上限）会启动报错。4096 只是当年 8GB 开发机的保守值
              enforce_eager=True,                            # 关闭 CUDA Graph 优化：CUDA Graph 会把计算图录制下来加速，但要多占几百 MB 显存
              )
    # 单独加载分词器（与模型同名仓库）
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# 强制让引擎使用我们手里这个分词器实例，保证脚本里 encode 的结果和引擎内部完全一致
llm.set_tokenizer(tokenizer)

#TODO (Jiayi): fix last len   # 原作者(Jiayi)留下的待办：last_len 的计算有 bug（已在第 52 行修复，见该行注释）


# ============ 主循环：依次处理样本目录下的所有样本 ============
# 样本目录可用环境变量 BLEND_INPUT_DIR 指定（默认 inputs/，行为与原版一致：
# 只跑 1.json~10.json）；目录里没有纯数字命名的 json 时，跑目录下全部 .json
# （用于 inputs_long_6k10k_official/ 这类带出处信息命名的长上下文样本）
import re
INPUT_DIR = os.environ.get("BLEND_INPUT_DIR", "inputs")
_dir_jsons = [fn for fn in os.listdir(INPUT_DIR) if fn.endswith(".json")]
_numeric = sorted((fn for fn in _dir_jsons if re.fullmatch(r"\d+\.json", fn)),
                  key=lambda fn: int(fn[:-5]))
SAMPLE_FILES = _numeric if _numeric else sorted(_dir_jsons)

for sample_file in SAMPLE_FILES:
    # 打开当前样本文件
    f = open(f"{INPUT_DIR}/{sample_file}")
    # 解析 JSON，得到字典 ex（example 的缩写）
    ex = json.load(f)
    # 这个样本里有几篇文档块（chunk = 检索回来的一段文档）
    chunk_num = ex['chunk_num']
    # 按编号 "0","1","2"... 取出所有文档块文本，组成列表
    doc_prompts = [ex[f'{i}'] for i in range(chunk_num)]
    # 取出用户问题（query prompt）
    q_prompt = ex['query']
    # 把每篇文档转成 token 编号列表；Mistral 的 encode 自动加 BOS 起始符
    #（<s>，编号 1），[1:] 是切掉它——因为后面拼接时我们会自己加，不去掉就会重复。
    # Qwen 的 tokenizer 不加 BOS，_bos_offset=0 不剥首 token
    _bos_offset = 0 if os.environ.get("BLEND_MODEL_PRESET", "bf16") == "qwen" else 1
    doc_chunk_ids = [tokenizer.encode(doc)[_bos_offset:] for doc in doc_prompts]
    # 同理，把问题转成 token 编号
    q_ids = tokenizer.encode(q_prompt)[_bos_offset:]
    # Qwen 的 BPE 会在块边界跨块合并 token：decode→encode 往返不幂等
    #（实测 inputs/1.json 拼接后 3601 个 token 中 3544 个位置错位），
    # 所以 qwen 预设必须直接给引擎传 token id，绕过重新分词；
    # Mistral（SentencePiece）无此问题，保持原来的字符串路径不变
    _use_token_ids = (_bos_offset == 0)

    def _run_prompt(ids, params):
        if _use_token_ids:
            return llm.generate(prompt_token_ids=[list(ids)],
                                sampling_params=params)
        return llm.generate([tokenizer.decode(ids)], params)


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

    # --- 拼装指令模板对应的特殊 token ---
    if os.environ.get("BLEND_MODEL_PRESET", "bf16") == "qwen":
        # Qwen2.5 用 ChatML 模板：<|im_start|>system ... <|im_end|>
        # <|im_start|>user\n{文档...问题}<|im_end|>\n<|im_start|>assistant\n
        s_start_full = tokenizer.encode(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n")
        s_end = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n")
    else:
        # Mistral 指令模板：s_start_full 相当于 "<s>[INST]"（s = start，
        # full = 带 BOS 起始符的完整版）。Mistral 的 instruct 模板要求输入以它开头
        s_start_full = [733, 4138, 28793]
        # s_end：指令结束标记的 token 编号，相当于 "[/INST]"
        s_end = [733, 28748, 16289, 28793]
    # 前缀"长度"：实际 token 数 + BOS 补偿。Mistral 引擎内部 encode 时会自动
    # 再加一个 BOS，所以存下来的 KV 比 token 列表多出开头 1 个位置，要算上它；
    # Qwen 引擎不加 BOS，_bos_offset=0 无补偿
    s_start_len = len(s_start_full) + _bos_offset

    #s_start = [518, 25580, 29962]   # 注释掉的旧值：这是 LLaMA-2 的 "[INST]" token 编号，作者早期用 LLaMA 调试的遗留
    # s_start：每个文档块自己的前缀，这里是空列表（文档块前不加任何包装）
    s_start = []
    # s_start_1_len：块前缀在 KV 里的占位长度 = 0 + BOS 补偿
    s_start_1_len = len(s_start) + _bos_offset
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
        # 跑这一块。因为 max_tokens=1 且 collect=True，这次调用真正的目的是：
        # 让 prefill 发生，从而每层把这一块的 KV 存进 hack_kv
        #（qwen 预设走 token id 直传，避免 BPE 边界错位）
        _run_prompt(doc_chunk_ids[i], sampling_params)

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
            # 各块 decode 时被引擎加过的 BOS 在这里要去掉，否则会重复。
            # 注意 chunk id 列表本身不含 BOS（encode 时已剥），所以实际取 [0:] 全量；
            # Qwen 无 BOS 补偿（s_start_1_len=0），max(...) 防止 [-1:] 只取末位 token
            temp_ids = doc_chunk_ids[i][max(s_start_1_len-1, 0):]
        # 逐块拼成一条完整的输入
        input_ids += temp_ids

    # 转回文字形式，这就是最终要推理的完整 prompt：[前缀] 文档1 文档2 ... 文档N 问题 [结束]
    input_prompt = tokenizer.decode(input_ids)


    # ============ 阶段二：CacheBlend 缓存融合推理 ============
    # 这次要真正生成答案了，max_tokens 默认 32（与 blend_musique.py 一致），
    # 可用 BLEND_MAX_TOKENS 覆盖（摘要类任务用 64，对齐 LongBench 官方上限）。
    # 注意：EM 评测要求答案生成完整，10 token 会把答案截断成残句导致 EM 恒为 0
    _max_tokens = int(os.environ.get("BLEND_MAX_TOKENS", "32"))
    sampling_params = SamplingParams(temperature=0, max_tokens=_max_tokens)
    # check=True：打开 CacheBlend 的选择性重算——模型会复用 old_kvs 里的缓存，
    # 只挑出"新鲜值与缓存值偏差最大"的一小部分 token（默认 16%）重算，
    # 把跨文档注意力补回来
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = False
    # suffix_len：告诉注意力层，输入末尾这么长的"问题段"必须无条件重算
    #（问题从来没被缓存过，且是生成答案的关键）
    cache_fuse_metadata['suffix_len'] = last_len

    # ===== DAF 开关 =====
    # daf=True：启用 DAF 算法；False 就退回原版 CacheBlend（用于对照实验）
    # 用环境变量切换：DAF=0 python example/blend.py 跑原版，不设或 DAF=1 跑 DAF
    use_daf = os.environ.get("DAF", "1") == "1"
    cache_fuse_metadata["daf"] = use_daf
    if use_daf:
        # sel_layers：打分层，DAF 论文说聚合第 2~4 层的偏差信号，
        # Python 从 0 数层，所以是 [1,2,3]。选择层自动定为 max(sel_layers)+1 = 第 4 层
        cache_fuse_metadata["sel_layers"] = [1, 2, 3]
        # 重要 token 比例 r：论文实验用 0.3；它不是固定常数，可按任务调
        cache_fuse_metadata["recomp_ratio"] = 0.3
        # 同步旧字段 check_layers，避免两套配置互相矛盾
        #（daf 分支实际用 sel_layers 排班，这里只是防止旧代码读到不一致的值）
        cache_fuse_metadata["check_layers"] = [max(cache_fuse_metadata["sel_layers"]) + 1]
    else:
        # 论文对比要求 CB 与 DAF 用同一 r（0.3），DAF 论文里 CacheBlend 基线也是 r=0.3；
        # 想复现 CB 原论文配置（r≈0.16，也是 daf1.1~1.3 历史数据用的值）时设 CB_RECOMP_RATIO=0.16
        cache_fuse_metadata["recomp_ratio"] = float(os.environ.get("CB_RECOMP_RATIO", "0.3"))
    # 注意：daf=False 时 check_layers 保持白板默认 [1]（CB 原版单层选择）
    # 用缓存跑推理
    output = _run_prompt(input_ids, sampling_params)
    # 打印当前样本文件名（EM 评分脚本据此把结果对应回样本；长上下文样本名含来源与长度）
    print(f"===== {sample_file} =====")
    # 打印缓存模式生成的答案
    print(f"Cached generation: {output[0].outputs[0].text}")
    # 打印缓存模式的 TTFT：第一个 token 生成时刻 - 请求开始调度时刻，
    # 即"从提交到出第一个字"的等待时间
    print(f"TTFT with cache: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")

    # ============ 阶段三：对照组——全量 prefill ============
    sampling_params = SamplingParams(temperature=0, max_tokens=_max_tokens)  # 与阶段二对齐，保证质量对比公平
    # 关掉所有缓存开关，模型老老实实地把整段输入从头算一遍
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata['collect'] = False
    output = _run_prompt(input_ids, sampling_params)


    # 打印全量重算的答案
    print(f"Normal generation: {output[0].outputs[0].text}")
    # 打印全量重算的 TTFT——和上面的 TTFT with cache 一比，加速比就出来了
    print(f"TTFT with full prefill: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    # 样本之间的分隔线
    print("------------")

