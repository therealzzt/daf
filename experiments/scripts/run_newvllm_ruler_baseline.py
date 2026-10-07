# =============================================================================
# 路线 A 排雷实验：用【原生新版 vLLM】跑 RULER 30k 全量基线
# 位置：experiments/scripts/（用 vllm_new 环境的 python 运行，不用 cacheblend）
#
# 目的：我们的全量基线（vllm 0.4.1 魔改版 + enforce_eager + max_tokens=32）
#       在 RULER 500 样本上是 qa_1 SM 0.716 / qa_2 SM 0.550，低于论文 vLLM
#       基线（0.744/0.765）。两个嫌疑变量：
#         ① 引擎版本：vllm 0.4.1 早于 Qwen2.5 发布，走 Qwen2 兼容路径
#         ② max_tokens=32 截断：RULER 官方 tokens_to_generate=128
#       两臂分离：臂1 = 新引擎 + max_tokens 32（与 0.550 直接可比 → 引擎因素）
#                 臂2 = 新引擎 + max_tokens 128（与臂1 比 → 截断因素）
#
# prompt 构造（对齐 RULER 官方 harness，见 ruler_src/scripts/pred/
#   call_api.py:305 + client_wrappers.py:290）：
#   content = input + answer_prefix（" Answer:" 在 user 消息内部末尾），
#   只发一条 user 消息（system_msg=[]），apply_chat_template 会自动补
#   Qwen2.5 官方默认 system——与我们 blend 管线的信封语义一致
#
# 日志格式与 example/blend.py 完全兼容（===== 样本名 ===== / Cached generation /
# TTFT with cache / Normal generation / TTFT with full prefill），
# 可直接复用 score_em_long_6k10k.py（本基线没有缓存/全量之分，
# 两行打印同一份生成结果，TTFT 同理）：
#   - 样本名沿用 blend 样本文件名（qa_1_000_lenXXXX.json），
#     通过 experiments/inputs_qwen_ruler30k_500/ 的前缀 glob 找回，保证评分时
#     gold 对得上（ruler jsonl 第 i 行 == blend 样本 qa_X_{i:03d}）
#
# 环境变量：
#   NEWVLLM_TASK      qa_1 | qa_2（必填）
#   NEWVLLM_MAX_TOKENS 32 | 128（必填，臂选择）
#   NEWVLLM_RANGE     "lo:hi"（分片，左闭右开；不设=全部 500）
#   NEWVLLM_GPU_UTIL  gpu_memory_utilization（默认 0.90）
# 用法示例见 run_newvllm_baseline_resume.sh
# =============================================================================
import json
import os
import re
import sys
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

TASK = os.environ["NEWVLLM_TASK"]
MAX_TOKENS = int(os.environ["NEWVLLM_MAX_TOKENS"])
MODEL = "Qwen/Qwen2.5-7B-Instruct"
# RULER 原始数据（input 不含 answer_prefix，需拼回）
SRC = f"experiments/ruler_data_gen_500/{TASK}/validation.jsonl"
# blend 样本目录：仅用于把"第 i 行"映射回样本文件名（gold 一致）
BLEND_DIR = "experiments/inputs_qwen_ruler30k_500"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL,
          max_model_len=32768,
          enforce_eager=True,          # 与我们的 0.4.1 管线对齐，排除 cuda graph 变量
          gpu_memory_utilization=float(os.environ.get("NEWVLLM_GPU_UTIL", "0.90")))
# vllm 0.30 移除了 LLM.set_tokenizer；反正下面直传 token id，
# 引擎内部的分词器不会被动用，无需替换

# 第 i 行 → blend 样本文件名（前缀 qa_X_{i:03d}_）
def blend_name(idx):
    hits = [fn for fn in os.listdir(BLEND_DIR)
            if fn.startswith(f"{TASK}_{idx:03d}_") and fn.endswith(".json")]
    assert len(hits) == 1, f"{TASK} 第 {idx} 行找不到唯一样本名: {hits}"
    return hits[0]

rows = [json.loads(line) for line in open(SRC)]
assert len(rows) == 500, f"{SRC} 应有 500 行，实际 {len(rows)}"
_lo, _hi = os.environ.get("NEWVLLM_RANGE", f"0:{len(rows)}").split(":")
rows = rows[int(_lo):int(_hi)]

sampling_params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

for i, ex in enumerate(rows):
    idx = i + int(_lo)
    sample_name = blend_name(idx)
    # RULER 官方口径：input + answer_prefix 作为唯一一条 user 消息
    content = ex["input"] + ex.get("answer_prefix", "")
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True)
    # Qwen 的 BPE 在模板拼接处不幂等（decode→encode 会错位），
    # 直接传 token id 绕过引擎重新分词（与 blend.py 的 qwen 预设同理）
    prompt_ids = tokenizer.encode(prompt_text)
    assert len(prompt_ids) <= 32768, f"{sample_name} prompt 超长: {len(prompt_ids)}"

    t0 = time.perf_counter()
    # vllm 0.30 的 generate 不再接受 prompt_token_ids= 关键字，改传 TokensPrompt
    output = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)],
                          sampling_params=sampling_params)
    t1 = time.perf_counter()
    text = output[0].outputs[0].text
    # 新版 vllm 的 metrics 字段与 0.4.1 不同且常为空，TTFT 用整段墙钟时间
    # 记录（单请求调用，prefill 占绝对主导，与 blend 管线的 TTFT 口径近似可比；
    # 本实验主指标是精度，计时仅参考）
    ttft = t1 - t0

    print(f"===== {sample_name} =====")
    print(f"Cached generation: {text}")
    print(f"TTFT with cache: {ttft}")
    print(f"Normal generation: {text}")
    print(f"TTFT with full prefill: {ttft}")
    print("------------", flush=True)

# vllm 0.30 退出处理的两个坑（都实测踩过）：
#   ① 自然退出时主进程在 atexit 阶段睡死（S 态等 zmq join，>10min 不退）
#   ② 直接 os._exit 会把 EngineCore 子进程变成孤儿（PPID=1），
#      继续占着 ~22GB 显存不放，下一片启动直接 OOM
# 正解：先显式 shutdown（带超时）杀 EngineCore，再 os._exit 兜底
print(f"[全部完成] {len(rows)} 条", flush=True)
try:
    llm.llm_engine.engine_core.shutdown(timeout=60)
except Exception as e:
    print(f"[关停异常，兜底强退] {e}", flush=True)
sys.stdout.flush()
os._exit(0)
