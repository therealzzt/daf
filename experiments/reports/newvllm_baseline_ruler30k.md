# 路线 A 排雷报告：原生新版 vLLM（0.30.0）RULER 30k 全量基线

日期：2026-10-07（跑批于 2026-09-23 夜完成）。实验性质：**归因排雷**——回答
"我们的 RULER 30k 全量基线精度低于论文（qa_2 SM 0.550 vs 0.765），
引擎版本是不是主因"。结论先行：**不是。引擎版本与 max_tokens 截断两个
嫌疑均被排除，两者合计贡献 ≤0.015 SM，剩余 ~0.20 差距需在别处找。**

## 一、实验设置

- 模型：Qwen/Qwen2.5-7B-Instruct（bf16），与论文及我们既有基线一致。
- 数据：RULER 官方生成 32768 档 qa_1（squad）/ qa_2（hotpotqa）各 500 条
  （`experiments/ruler_data_gen_500/`，seed 42）——**与我们 0.4.1 基线
  （c0163a9）是同一组 500 条**，逐任务严格同样本同分母对照。
  注：每臂总生成条数 1000 = 2 任务 × 500；按任务对照时分母都是 500。
  "前 500 条口径"经脚本显式核对，与全量完全一致（每任务本就恰好 500 条）。
- 引擎（新 conda 环境 `vllm_new`，**不入库**）：
  vllm **0.30.0** + torch **2.13.0+cu130** + transformers 5.17.0，
  CUDA 13 系依赖（清华镜像安装）。`enforce_eager=True`、
  `gpu_memory_utilization=0.90`、`max_model_len=32768`、贪心解码——
  与 0.4.1 基线管线对齐。
- prompt 口径：严格对齐 RULER 官方 harness
  （`ruler_src/scripts/pred/call_api.py:305` + `client_wrappers.py:290`）：
  `input + answer_prefix` 拼成唯一一条 user 消息，apply_chat_template
  （自动补 Qwen2.5 官方默认 system），token id 直传绕过 BPE 边界不幂等。
- 两臂设计：
  - 臂1 `max_tokens=32`：与我们 0.4.1 基线生成配置完全一致 → 与 0.550
    直接可比，隔离**引擎因素**。
  - 臂2 `max_tokens=128`：RULER 官方 tokens_to_generate → 与臂1 比，
    隔离**截断因素**。
- 驱动脚本 `experiments/scripts/run_newvllm_ruler_baseline.py`，
  续跑脚本 `experiments/scripts/run_newvllm_baseline_resume.sh`
  （5 片 × 100，按 `^===== ` 计数幂等，5 轮外层重试）。
  日志 `experiments/logs/newvllm_baseline/run_newvllm_mt{32,128}_{qa1,qa2}_c{00..04}.log`，
  20 片全部 100/100，零报错。
- 评分：`score_em_long_6k10k.py --raw_parse --tokenizer Qwen/Qwen2.5-7B-Instruct`，
  与此前两轮完全同一脚本同一归一化；SM 与 RULER 官方 string_match_part
  等价性此前已核对。逐样本明细见同目录
  `newvllm_baseline_score_{mt32,mt128}_{qa1,qa2}.md`。

## 二、四方对照（均值 ± 95% CI，n=500/任务）

SM = 子串匹配（RULER 主指标）；EM = 精确匹配；TTFT 为 500 样本中位数
（新引擎为单请求墙钟，口径近似，本实验计时仅供参考）。

| 任务 | 基线 | EM | SM | TTFT 中位(s) |
|---|---|---|---|---|
| qa_1 | 我们 0.4.1 魔改（mt32） | 0.438±0.043 | 0.716±0.040 | 7.55 |
| qa_1 | **新引擎 0.30.0 mt32** | 0.430±0.043 | **0.716±0.040** | 8.27 |
| qa_1 | 新引擎 0.30.0 mt128 | 0.430±0.043 | 0.718±0.039 | 8.26 |
| qa_1 | 论文 vLLM 基线 | — | 0.744 | — |
| qa_2 | 我们 0.4.1 魔改（mt32） | 0.374±0.042 | 0.550±0.044 | 7.66 |
| qa_2 | **新引擎 0.30.0 mt32** | 0.376±0.042 | **0.554±0.044** | 8.33 |
| qa_2 | 新引擎 0.30.0 mt128 | 0.376±0.042 | 0.564±0.043 | 8.34 |
| qa_2 | 论文 vLLM 基线 | — | 0.765 | — |

## 三、判定

**① 引擎因素：排除。** 新引擎 mt32 与 0.4.1 基线在**完全相同的 500 条样本**
上给出几乎相同的分数：qa_1 SM 0.716 vs 0.716（逐题一致）、qa_2 SM 0.554
vs 0.550（差 +0.004 = 2 题，远在 ±0.044 的 CI 内）。vllm 0.4.1 早于
Qwen2.5 发布、走 Qwen2 兼容路径——曾被怀疑精度有损，本实验证明该兼容
路径在数值上与原生新引擎**精度等价**。引擎版本对 qa_2 那 0.215 的差距
（0.765 − 0.550）贡献 ≈ 0。

**② 截断因素：≤0.01 SM，基本排除。** mt32 臂生成长度触顶（=32 token）
比例仅 6.4%（qa_1）/ 5.8%（qa_2），且触顶样本里多数已在前 32 token
内给出含金标的片段。mt128 与 mt32 的逐样本配对差：qa_1 均值 +0.002
（翻正 1 题），qa_2 均值 +0.010（翻正 5 题，无翻负）。即便按 RULER 官方
128 生成，qa_2 也只到 0.564，距 0.765 仍有 0.201。

**③ 剩余 ~0.20 差距的嫌疑排序（引擎、截断均已洗清）：**
1. **评测口径/数据版本**：论文未给出其 vLLM 基线的完整生成配置
   （system prompt、stop、数据 seed/版本）。RULER 生成器不同 commit 的
   hotpotqa 干扰段落分布有差异，qa_2 对干扰极度敏感。
2. **硬件数值**：论文 A100 vs 我们 3090Ti，bf16 累加序差异——经验量级
   <0.01，不足以解释 0.20，列为次要。
3. 注意对照组 sanity：qa_1 我们 0.716 vs 论文 0.744 只差 0.028
   （CI 边缘），差距集中在 qa_2 一个任务上，更像数据/口径问题而非
   系统性实现缺陷。

## 四、对路线 B（LMCache 迁移）必要性的影响

**以"精度修复"为动机的迁移：不成立。** 此前把 LMCache/新版 vLLM 迁移
列为路线 B 的前提假设是"0.4.1 引擎太老导致精度受损"。本实验在同一批
样本上证明新旧引擎精度一致（qa_2 0.554 vs 0.550），迁移拿不到精度收益。
若仍要迁移，动机只剩：① 性能（新 kernel/调度器在 30k+ 的 prefill 效率，
本次参考计时 8.3s vs 0.4.1 的 7.6s 反而略慢，sm86 上未见红利）；
② 可维护性（官方对 Qwen2.5 的一等支持）。**优先级建议下调**；
qa_2 差距的下一主攻转为：评测口径逐项 diff（RULER 官方 harness 原样
复跑其 vLLM 基线配置）+ DAF 打分层消融（论文验收线 qa_2 DAF≥0.68
的达成不依赖全量基线绝对值，但 0.148 的现状同样需要先搞清楚
qa_2 任务本身的难度来源）。

## 五、复现要点与踩坑记录（vllm 0.30.0 on 3090Ti/sm86）

1. **flashinfer JIT 两连坑**：0.6.18 的 sampling 模块在引擎启动时 JIT 编译，
   系统 nvcc（CUDA 12.0）不认识 `--compress-mode=size`；换 pip 装的
   CUDA 13.4 nvcc（`nvidia/cu13/bin/nvcc`）又与其自带 CCCL 头文件版本
   互斥（"CUDA compiler and CUDA toolkit headers are incompatible"）。
   解：`VLLM_USE_FLASHINFER_SAMPLER=0` 退回 PyTorch 原生采样即可完全
   绕开 JIT，贪心解码精度路径不受影响。注意 vllm 0.30 已**移除**
   `VLLM_ATTENTION_BACKEND` 环境变量（设了只会报 Unknown env var）。
2. **退出挂死**：自然退出时主进程在 atexit 阶段睡死等 zmq join
   （实测 >10min 不退）。解：脚本末尾
   `llm.llm_engine.engine_core.shutdown(timeout=60)` + `os._exit(0)`。
3. **孤儿 EngineCore**：只 `os._exit` 不显式 shutdown 会把 EngineCore
   变成 PPID=1 的孤儿，继续占 ~22GB 显存，下一个分片直接 OOM 拒启动。
   续跑脚本每片前后清理 PPID=1 的 `VLLM::EngineCore`（正常引擎父进程
   是活的，不会误杀）。
4. **显存检查更严**：0.30 启动时检查"空闲显存 ≥ util×总量"，其他进程
   临时占 2.7GB 就会拒绝启动（0.4.1 没这个检查）。续跑脚本外层 5 轮
   重试吸收瞬时占用。
5. `LLM.set_tokenizer()` 已移除；`generate()` 不再收
   `prompt_token_ids=` 关键字，改传 `TokensPrompt`。
6. pip 官方源直连 ~30KB/s，换 `pypi.tuna.tsinghua.edu.cn` 后正常；
   整套 CUDA 13 依赖 + torch + vllm 约 10GB。
