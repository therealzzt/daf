# DAF 冒烟测试：只跑 inputs/1.json 一个样本
# 用法：DAF=0 跑原版 CacheBlend 回归；DAF=1 跑 DAF
# 需要在仓库根目录下运行（依赖 inputs/ 相对路径）
import os, json, torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

DAF = os.environ.get("DAF", "1") == "1"

llm = LLM(model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
          quantization="awq",
          gpu_memory_utilization=0.9,
          max_model_len=4096,
          enforce_eager=True)
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
llm.set_tokenizer(tokenizer)

ex = json.load(open("inputs/1.json"))
doc_prompts = [ex[f'{i}'] for i in range(ex['chunk_num'])]
q_prompt = ex['query']
doc_chunk_ids = [tokenizer.encode(d)[1:] for d in doc_prompts]
q_ids = tokenizer.encode(q_prompt)[1:]

model_inner = llm.llm_engine.model_executor.driver_worker.model_runner.model.model
cache_fuse_metadata = model_inner.cache_fuse_metadata

s_start_full = [733, 4138, 28793]
s_start_len = len(s_start_full) + 1
s_start = []
s_start_1_len = len(s_start) + 1
s_end = [733, 28748, 16289, 28793]

doc_chunk_ids = [s_start + c for c in doc_chunk_ids]
doc_chunk_ids = [s_start_full] + doc_chunk_ids + [s_start + q_ids + s_end]
last_len = len(q_ids + s_end)
print(f"suffix_len = {last_len}")

num_layer = len(model_inner.layers)
print(f"num_layer = {num_layer}")

# ---- 阶段一：collect ----
chunk_past_key_values = []
sp1 = SamplingParams(temperature=0, max_tokens=1)
cache_fuse_metadata['collect'] = True
cache_fuse_metadata['check'] = False
for i in range(len(doc_chunk_ids)):
    llm.generate([tokenizer.decode(doc_chunk_ids[i])], sp1)
    for j in range(num_layer):
        hkv = model_inner.layers[j].self_attn.hack_kv
        if i == 0:
            tk = hkv[0][:s_start_len].clone()
            tv = hkv[1][:s_start_len].clone()
        else:
            tk = hkv[0][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
            tv = hkv[1][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
        if i == 0:
            chunk_past_key_values.append([tk, tv])
        else:
            chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0], tk), dim=0)
            chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1], tv), dim=0)
    model_inner.old_kvs = chunk_past_key_values

input_ids = []
for i in range(len(doc_chunk_ids)):
    input_ids += doc_chunk_ids[i] if i == 0 else doc_chunk_ids[i][s_start_1_len-1:]
input_prompt = tokenizer.decode(input_ids)
print(f"total_len = {len(input_ids) + 1}")  # +1 是引擎自动加的 BOS

# ---- 阶段二：check（DAF 或原版 CacheBlend）----
sp10 = SamplingParams(temperature=0, max_tokens=10)
cache_fuse_metadata["check"] = True
cache_fuse_metadata['collect'] = False
cache_fuse_metadata['suffix_len'] = last_len
cache_fuse_metadata["daf"] = DAF
cache_fuse_metadata["sel_layers"] = [1, 2, 3]
cache_fuse_metadata["recomp_ratio"] = 0.3
cache_fuse_metadata["check_layers"] = [4]
out = llm.generate([input_prompt], sp10)
print(f"[DAF={DAF}] cached generation: {out[0].outputs[0].text}")
print(f"imp_indices size = {cache_fuse_metadata['imp_indices'].shape if cache_fuse_metadata['imp_indices'] is not None else None}")
if DAF:
    print(f"num_imp_tokens = {cache_fuse_metadata['num_imp_tokens']}")

# ---- 阶段三：全量 prefill 对照 ----
cache_fuse_metadata["check"] = False
cache_fuse_metadata['collect'] = False
out = llm.generate([input_prompt], sp10)
print(f"[DAF={DAF}] full generation: {out[0].outputs[0].text}")
print("SMOKE TEST DONE")
