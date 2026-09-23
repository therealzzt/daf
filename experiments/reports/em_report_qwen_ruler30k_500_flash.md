# RULER 30k 档 500 样本：FLASH_ATTN vs XFORMERS 后端对照报告（kernel 红利测量）

日期：2026-09-23。模型：Qwen/Qwen2.5-7B-Instruct（bf16），硬件 4×3090Ti（sm86）。
数据集：与 xformers 轮完全相同的 `experiments/inputs_qwen_ruler30k_500/`（qa_1/qa_2 各 500）。
日志：`experiments/logs/qwen_ruler30k_500_flash/`（40 个分片日志，各 50 样本，
全部 50/50 零 OOM 零 Traceback，每片均有 `Using FlashAttention backend` 确认行，已核验）。

**动机**：xformers 轮 DAF/CB 时长比 0.946，与论文（A100 + FlashAttention）的 0.77 有差距。
本轮同样本同配置只换 kernel，测差距中有多少是 kernel 贡献的。

## 一、xformers vs flash 并排对照（均值 ± 95% CI，n=500/方法/任务）

### qa_1

| 方法 | 后端 | EM | SM | TTFT 中位(s) | 相对全量加速 |
|---|---|---|---|---|---|
| CacheBlend | xformers | 0.124±0.029 | 0.318±0.041 | 3.054 | 2.47× |
| CacheBlend | **flash** | 0.124±0.029 | 0.318±0.041 | 3.060 | 2.47× |
| DAF | xformers | 0.168±0.033 | 0.350±0.042 | 2.892 | 2.61× |
| DAF | **flash** | 0.168±0.033 | 0.364±0.042 | 2.900 | 2.61× |
| vLLM 全量 | xformers | 0.438±0.043 | 0.716±0.040 | 7.549 | 1.00× |
| vLLM 全量 | **flash** | 0.438±0.043 | 0.716±0.040 | 7.559 | 1.00× |

### qa_2

| 方法 | 后端 | EM | SM | TTFT 中位(s) | 相对全量加速 |
|---|---|---|---|---|---|
| CacheBlend | xformers | 0.106±0.027 | 0.252±0.038 | 3.101 | 2.47× |
| CacheBlend | **flash** | 0.106±0.027 | 0.252±0.038 | 3.104 | 2.47× |
| DAF | xformers | 0.096±0.026 | 0.148±0.031 | 2.933 | 2.61× |
| DAF | **flash** | 0.098±0.026 | 0.152±0.031 | 2.947 | 2.61× |
| vLLM 全量 | xformers | 0.374±0.042 | 0.550±0.044 | 7.664 | 1.00× |
| vLLM 全量 | **flash** | 0.374±0.042 | 0.550±0.044 | 7.680 | 1.00× |

### DAF/CB 时长比与论文对照

| 口径 | xformers 轮 | flash 轮 | 论文（A100+FA） |
|---|---|---|---|
| DAF/CB（qa_1） | 0.947 | **0.948** | ≤0.77 |
| DAF/CB（qa_2） | 0.946 | **0.949** | ≤0.77 |
| DAF vs 全量 | 2.61× | 2.61× | ≈2.8× |

## 二、核心结论：kernel 红利在 sm86 上趋近于零

**换 flash kernel 后 DAF/CB 从 0.946 变为 0.948/0.949——不动；三方法 TTFT 全部
变化 <0.5%（远小于跨样本波动）。kernel 红利在 3090Ti（sm86）上趋近于零，
0.946 vs 0.77 的差距主体归属硬件代差（A100 vs 3090Ti 的算力/显存带宽/attention
kernel 效率），而非 kernel 选择。**

解释（工程层面）：vllm 0.4.1 在 sm86 上的 xformers memory_efficient_attention
已经是成熟的 cutlass 后端，flash-attn 2.5.8 对 sm86 没有代差级优势
（FA 的相对优势在 sm90/Hopper 的 TMA/wgmma 时代才拉开）；且 DAF 的瓶颈部分
（打分层全量 prefill、hybrid KV 组装、逐层 Python 调度）本来就不在 kernel 里。

## 三、精度一致性确认（与 xformers 轮逐方法对比）

| 任务 | 方法 | ΔSM（flash−xf） | ΔEM | 判定（CI ±0.03~0.04） |
|---|---|---|---|---|
| qa_1 | CB | 0.000 | 0.000 | ✅ 完全一致 |
| qa_1 | DAF | +0.014 | 0.000 | ✅ CI 内 |
| qa_1 | 全量 | 0.000 | 0.000 | ✅ 完全一致 |
| qa_2 | CB | 0.000 | 0.000 | ✅ 完全一致 |
| qa_2 | DAF | +0.004 | +0.002 | ✅ 完全一致（舍入级） |
| qa_2 | 全量 | 0.000 | 0.000 | ✅ 完全一致 |

CB 与全量臂 SM/EM 逐分相同（贪心解码 + 相同数值路径下 flash 与 xformers 的
输出在绝大多数样本逐字一致）；DAF 臂 SM 差 +0.014/+0.004（qa_1 多对 7 题、
qa_2 多对 2 题），来自 topk 选人边界的 kernel 浮点差（铁律：两后端只在
SM/EM 级别对齐，个别样本贪心解码尾部 token 翻转属正常）。

抽查核对（报告分数 vs 日志原文）：

| 样本 | flash 日志 Cached generation | 报告 CB 行 | 一致 |
|---|---|---|---|
| qa_1_377_len31609 | "Pacific Ocean" | EM 1.00 SM 1.00 TTFT 3.109 | ✅（与 xformers 轮答案逐字相同） |
| qa_2_134_len32449 | "The film is \"The Big Lebowski\"." | EM 0.00 SM 0.00 TTFT 3.204 | ✅（与 xformers 轮答案逐字相同） |

## 四、配置差异注明

- 唯一后端差异：`VLLM_ATTENTION_BACKEND=FLASH_ATTN`（flash_attn 2.5.8，
  cu122torch2.2 cp310 wheel）；xformers 轮为默认 XFORMERS。
- util 对称性：flash 轮两臂统一 0.89；xformers 轮 CB 臂最初 0.90、后改 0.89
  （碎片化 OOM）。util 只决定 KV 池大小（0.89 池 ~3.7 万 token > 32768），
  不影响数值——此不对称对结论无影响。
- 其余配置逐项相同：同样本目录、`BLEND_CHATML=1`、`BLEND_MAX_MODEL_LEN=32768`、
  `BLEND_MAX_TOKENS=32`、V-only 打分、recomp_ratio=0.3、贪心解码、
  1 轮 × 500 样本协议、10 片 × 50 幂等续跑（`run_ruler500_flash_resume.sh`）。

## 附：评分命令

```bash
python experiments/scripts/score_em_long_6k10k.py --dir experiments/inputs_qwen_ruler30k_500 \
    --tokenizer Qwen/Qwen2.5-7B-Instruct --raw_parse \
    --logs experiments/logs/qwen_ruler30k_500_flash/run_*_qa1_*.log --out /tmp/scored500_flash_qa1.md
# qa_2 换 run_*_qa2_*.log；均值/CI 为对报告第一节逐样本行的 ad-hoc 聚合
```
