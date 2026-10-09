# =============================================================================
# LMCache CacheBlend 基线冒烟（Phase 0）
#
# 流程：用仓库 inputs/1.json 的已知样本（正确答案 Carnegie Mellon University）
#   1) 先单独 prefill 每个文档块 → LMCache 按块缓存 KV（离线建库的同价物）
#   2) 再把 文档块<sep>文档块…<sep>问题 拼成 blend prompt 请求 → 走 blending 融合路径
#   3) 打印答案 + 墙钟时间，与全量重算对照
#
# 运行（仓库根目录）：
#   LMCACHE_CONFIG_FILE=$PWD/experiments/lmcache/config_lmcache_cb_r03.yaml \
#     python experiments/lmcache/smoke_blend.py
# =============================================================================

import json
import os
import time

from vllm import LLM, SamplingParams

BLEND_SEP = "<|blend_sep|>"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

llm = LLM(model=MODEL,
          gpu_memory_utilization=0.85,
          max_model_len=32768,
          enforce_eager=True,
          kv_transfer_config={
              "kv_connector": "LMCacheConnectorV1",
              "kv_role": "kv_both",
          })

ex = json.load(open("inputs/1.json"))
docs = [ex[f"{i}"] for i in range(ex["chunk_num"])]
query = ex["query"]
print(f"文档块数: {len(docs)}, 问题: {query}")

# 阶段 1：逐块 prefill，把各块 KV 灌进 LMCache 缓存
sp_cache = SamplingParams(temperature=0, max_tokens=1)
t0 = time.perf_counter()
for d in docs:
    llm.generate([d], sp_cache)
t_cache = time.perf_counter() - t0
print(f"[建缓存] 逐块 prefill {len(docs)} 块: {t_cache:.2f}s")

# 阶段 2：blending 融合推理（答案已知：Carnegie Mellon University）
blend_prompt = BLEND_SEP.join(docs) + BLEND_SEP + query
sp = SamplingParams(temperature=0, max_tokens=32)
t0 = time.perf_counter()
out = llm.generate([blend_prompt], sp)
t_blend = time.perf_counter() - t0
ans_blend = out[0].outputs[0].text
print(f"[CacheBlend] {ans_blend!r}  耗时 {t_blend:.2f}s")

# 阶段 3：全量重算对照
full_prompt = "\n\n".join(docs) + "\n\n" + query
t0 = time.perf_counter()
out = llm.generate([full_prompt], sp)
t_full = time.perf_counter() - t0
ans_full = out[0].outputs[0].text
print(f"[全量重算] {ans_full!r}  耗时 {t_full:.2f}s")

print(f"\n加速比: {t_full/t_blend:.2f}×")
ok = "Carnegie Mellon University" in ans_blend and "Carnegie Mellon University" in ans_full
print("答案校验:", "PASS ✅" if ok else "FAIL ❌")
