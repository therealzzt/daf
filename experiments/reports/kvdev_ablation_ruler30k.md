# K+V deviation 消融报告（daf1.4 建议 3）—— RULER 30k 档

日期：2026-09-21。模型：Qwen/Qwen2.5-7B-Instruct（bf16）。数据集：RULER qa_1 / qa_2 各 24 样本（`experiments/inputs_qwen_ruler30k`，~30–32.7k token）。

## 结论（先看这里）

**K+V 联合打分在 30k 档对精度无稳定收益：qa_1 略降（SM 0.375→0.333），qa_2 略升（0.250→0.292），方向相反、幅度都在 ±1~2 题的噪声内；TTFT 不变。维持 V-only 为默认实现，K+V 保留为可选开关（`BLEND_KV_DEV=1`）。** 这是负/中性结果，如实记录。

## 背景与动机

DAF 打分现状是 V-only deviation：`sum((V_new - V_old)^2)`。V 不带位置信息，新旧可直接比；K 带 RoPE 旋转位置编码（RoPE = 按 token 绝对位置对 Q/K 做旋转变换），新旧 K 位置不同、坐标系不一致，直接相减会被位置旋转主导。本实验验证：把旧 K 对齐到新坐标系后，K+V 联合打分是否改善重要 token 选择质量。

## 实现方式（全部中文注释、含 shape 标注）

| 改动 | 文件：行号 | 说明 |
|---|---|---|
| 开关字段 | `vllm_blend/vllm/model_executor/models/qwen2.py:319-323` | 默认 dict 加 `"kv_dev": False, "key_old_rot": None`；check 开头重置（:352-354） |
| 旧 K 对齐 | `qwen2.py:171-190`（status==3 且 kv_dev 分支） | 旧 K 是 collect 阶段存的 RoPE 旋转**前**原始值（collect 分支 :192-193 在 rotary 之前 clone）；按 `org_pos`（check 阶段全序列**绝对位置**）补旋转，与 status∈[1,2] 现有对齐模式一致。**必须 `old_kv[0].clone()` 再旋转**：vllm 0.4.1 的 `ops.rotary_embedding` 是原地算子，原地改会污染共享离线 KV。结果写白板 `key_old_rot` |
| fake_q 缩小 | `qwen2.py:160-166`（status∈[1,2]）与 :185-186 | fake_q 只是给 rotary 算子占位的假 Q（输出丢弃），从 `rand_like(q)` 改为 `rand_like(old_kv[0])`：q 28 头 vs k 4 头（GQA），省 ~200MB。这是必要的——q 形 fake_q 在 32k 下 OOM（util 0.89 余量仅 ~220MB）。数值不变（回归证实） |
| K 偏差打分 | `vllm_blend/vllm/attention/backends/xformers.py:221-233`、`flash_attn.py:220-230`（两后端逐行一致） | `k_diff = sum((key[:doc_len]-key_old_rot[:doc_len])^2, dim=[1,2]).float()`，`dev = dev_V + dev_K` **等权（工程假设，论文未写死 K 是否参与及加权方式）**；含 NaN 断言 |
| 驱动开关 | `example/blend.py:280-282` | DAF 分支读 `BLEND_KV_DEV=0|1`（默认 0） |
| 调试钩子 | `example/blend.py:287-290` + `xformers.py:240-263` | `DAF_DEBUG_KVDEV=1` 时打印第一文档块 K 偏差对齐前/后对照（`[KVDEV-DBG]`），保留 |

`llama.py` 未动：Mistral 不参与本消融，后端 `.get("kv_dev", False)` 兜底缺省关。

## 数值验证（qa_1_000 单样本，3 个打分层，第一文档块）

第一块在 collect 与 check 阶段位置相同、左侧上下文也相同（因果掩码），理论上 K 应几乎不变：

| 打分层 | K 未对齐（直接减原始旧 K） | K 已对齐（RoPE 补旋转后） | V 偏差 | 后续块 K 已对齐（真实信号） |
|---|---|---|---|---|
| 层 1 | 2368 | 2.70 | 0.07 | 45 |
| 层 2 | 789 | 3.50 | 0.18 | 102 |
| 层 3 | 1151 | 2.14 | 0.15 | 78 |

对齐把第一块 K 偏差噪声压低 ~300–880×，对齐后块1噪声 vs 后续块真实信号的信噪比 15–30×，对齐正确。K 噪声底（~2-3）高于 V（~0.1）是 K 幅值更大所致（平方偏差随幅值²缩放），不影响 topk 排名逻辑。

## 消融对照（RULER 30k，分任务均值，n=24/任务）

两臂除 `BLEND_KV_DEV` 外配置完全相同：`DAF=1 BLEND_CHATML=1 BLEND_GPU_MEM_UTIL=0.89 BLEND_MAX_MODEL_LEN=32768 BLEND_MAX_TOKENS=32`，sel_layers=[1,2,3]、recomp_ratio=0.3、贪心解码。V-only 臂复用 ChatML 消融日志。

| 任务 | 打分方式 | EM 均值 | SM 均值 | F1 均值 | TTFT 中位(s) |
|---|---|---|---|---|---|
| qa_1 | V-only（默认） | 0.167 | 0.375 | — | 2.986 |
| qa_1 | **K+V** | 0.125 | 0.333 | 0.218 | 2.977 |
| qa_2 | V-only（默认） | 0.083 | 0.250 | — | 2.939 |
| qa_2 | **K+V** | 0.125 | 0.292 | 0.282 | 2.956 |
| 参考：qa_1 CacheBlend | — | 0.125 | 0.417 | — | 3.172 |
| 参考：qa_2 CacheBlend | — | 0.042 | 0.208 | — | 3.117 |

逐样本看：qa_1 K+V 净 -1 题（SM），qa_2 净 +1 题，方向相反，属噪声范围。TTFT 两臂差 <1.5%（K+V 多一次旧 K 的 rotary + 一次同形平方偏差，开销可忽略）。

## flash 一致性（qa_1，`VLLM_ATTENTION_BACKEND=FLASH_ATTN`）

flash-attn 后端（`flash_attn.py`，复用同一套 DAF 逻辑）与 xformers 后端跑同一批 24 样本：

- SM **24/24 完全一致**；EM/F1 均值一致（0.125/0.333 vs 0.125/0.333，F1 0.218 vs 0.232 的差来自答案措辞）
- 缓存答案逐字一致 21/24；3 个不同样本 SM 均为 0（数值路径差异只影响本就答错的样本的措辞）
- TTFT 中位 2.993s vs 2.977s（<1%）

两后端 K+V 实现行为一致，移植正确。

## 讨论

1. K+V 无收益的可能原因：V-only 的偏差信号已经足以定位"上下文变了"的文档 token；K 偏差与 V 偏差高度相关（同一层同一 token 的新旧差异），叠加后只是重复计数加噪声。等权叠加还让 K 的大噪声底（~2-3 vs V 的 ~0.1）稀释了 V 的干净信号。
2. 工程假设（非论文原方法）：dev = dev_V + dev_K 等权；旧 K 用绝对位置补旋转对齐。论文未写死 K 是否参与打分。
3. 后续可选：加权系数消融（如 0.1·dev_K + dev_V）或 K 偏差按幅值归一后再加。鉴于本档无收益，优先级低。

## 日志与评分文件

- 日志：`experiments/logs/qwen_ruler30k/run_daf_kvdev_chatml_qwen_ruler_qa{1,2}_r1.log`（K+V xformers）、`run_daf_kvdev_flash_chatml_qwen_ruler_qa1_r1.log`（K+V flash）；对照臂 `run_daf_chatml_qwen_ruler_qa{1,2}_r1.log`（V-only）
- 完整性：三个新日志各 24 样本（`===== ` 计数）、零 OOM；后端确认（flash 日志 `Using FlashAttention backend`，xformers 日志 `Using XFormers backend`）
- 评分：`python experiments/scripts/score_em_long_6k10k.py --dir experiments/inputs_qwen_ruler30k --tokenizer Qwen/Qwen2.5-7B-Instruct --raw_parse --logs <log> --out <out>`；分任务均值为对报告第一节逐样本行的 ad-hoc 聚合（报告中位数因多 0 分样本显 0，不代表均值）
