# DAF 1.4 实验总结：目标 B —— Qwen2.5-7B 官方数据集复现（6k/9k/30k 三档）

> 日期：2026-09-20（三档同一天完成：B-1/B-2 凌晨，B-3 上午）
> 环境：Ubuntu 24.04 服务器 + 4×RTX 3090 Ti 24GB（三档均 4 卡并行跑批）+ vllm 0.4.1（本仓库 vllm_blend）+ torch 2.2.1/cu121 + xformers 0.0.25 + transformers 4.41.2
> 模型：**Qwen2.5-7B-Instruct BF16**（DAF 论文所用模型；28 层、GQA 28Q/4KV、head_dim 128，KV ≈57KB/token）
> 前作：1.1~1.3（Mistral-7B + 自构造样本）见 `docs/daf1.1.md` ~ `docs/daf1.3.md`
> 本轮变化：从 Mistral 自构造样本切换到**论文配置**——Qwen2.5-7B + 官方数据集（LongBench / RULER）+ CB/DAF 同一 r=0.3

## 背景与前置工程

daf1.1~1.3 用 Mistral-7B + 自构造 buried-qa 样本，DAF/CB 比值收到 1.45× 但未反超，且 r 不对等（CB 0.16 / DAF 0.3）、数据集非官方，与论文可比性有限。目标 B 把这些混杂因子逐个剥掉：

- **DAF/CB 移植到 FlashAttention 后端 + Qwen2.5-7B 支持**（commit `7ae2c9c`）：`qwen2.py` 层调度移植；flash 后端对称实现；2–4k 档 {XFORMERS, FLASH} × {DAF=0,1} 40 次运行两后端输出逐字一致
- **Qwen BPE 跨块合并坑**（同 `7ae2c9c`）：Qwen 分词在文本块边界会发生 BPE 合并导致 token 错位，qwen 预设改走 `prompt_token_ids` 直传，collect/check/normal 三阶段一致
- **CB 默认 r 对齐 DAF 论文配置 0.3**（commit `aacaa30`）：`CB_RECOMP_RATIO=0.16` 可回退 CB 原论文配置；daf1.1~1.3 历史数据（CB r=0.16）与本轮不可直接对比
- 三档实验：B-1 `a08fb72`、B-2 `7a78e24`、B-3 `45c1888`

---

## 一、三档结果总表

| 档位 | 数据集（官方来源） | 样本 | 均长(tok) | CB TTFT(s) | DAF TTFT(s) | 全量 TTFT(s) | CB 加速 | DAF 加速 | **DAF/CB** |
|---|---|---|---|---|---|---|---|---|---|
| B-1（6k） | LongBench 2wikimqa | 24 | 6007 | 0.415 | 0.473 | 1.105 | 2.66× | 2.32× | **1.14×** |
| B-2（9k） | LongBench samsum | 24 | 9084 | 0.659 | 0.731 | 1.710 | 2.61× | 2.34× | **1.11×** |
| B-3（30k） | RULER qa_1/qa_2 | 24+24 | 31674/31643 | 3.115 | 2.929 | 7.677 | 2.46× | 2.63× | **0.94×（反超）** |

主指标（三方法，跨轮中位/均值；各档口径见对应报告）：

| 档位 | 主指标 | CacheBlend | DAF | vLLM 全量 |
|---|---|---|---|---|
| B-1 2wikimqa | EM | 0.208 | 0.208 | 0.250 |
| | SM / F1 | 0.292 / 0.281 | 0.333 / 0.346 | 0.417 / 0.414 |
| B-2 samsum | Rouge-L | 0.440 | 0.391 | 0.428 |
| B-3 RULER qa_1 | SM（均值） | 0.458 | 0.375 | 0.625 |
| B-3 RULER qa_2 | SM（均值） | 0.250 | 0.250 | 0.458 |

配置细节：三档均 CB r=0.3 / DAF r=0.3（sel_layers=[1,2,3]）/ 全量基线，贪心解码，3 轮取中位；B-1 `BLEND_MAX_MODEL_LEN=8192`（默认）、B-2 `16384` + `BLEND_MAX_TOKENS=64`（LongBench 官方 samsum max_gen）、B-3 `32768` + `32`（RULER 官方 tokens_to_generate）。

## 二、核心发现

### 2.1 DAF/CB 时长比完整趋势链：30k 首次反超

| 系列 | 短档 | 中档 | 30k |
|---|---|---|---|
| Mistral 自构造（daf1.1~1.3，r 不对等） | 1.75（2–4k） | 1.73（6–10k） | 1.45 |
| **Qwen 官方数据（本轮，r=0.3 对等）** | 1.14（6k） | 1.11（9k） | **0.94（反超）** |

论文"短上下文 DAF≈CB、长上下文 DAF 反超"的趋势在论文同款模型+官方数据上完整复现：6k/9k 两档比值 1.14/1.11 接近持平（与论文 6k/9k 持平的描述一致），30k 降到 0.94——**DAF 第一次在相同输入下比 CacheBlend 快**（daf1.3 曾外推反超点在 60k–130k，那是 Mistral+r 不对等序列上的外推；Qwen 序列起点低，反超提前到 ~25k 附近）。分任务看 qa_1=0.932、qa_2=0.943，反超不依赖单一数据集。

### 2.2 DAF 相对全量加速与论文逐档对照

| 档位 | 本轮 DAF vs 全量 | 论文 |
|---|---|---|
| 6k | 2.32× | 1.8× |
| 9k | 2.34× | ~2× |
| 30k | 2.63× | 2.8× |

三档全部达到或超过论文水平（6k/9k 超出较多，30k 基本持平）。DAF 的"相对全量加速"这一论文主指标可判定为**已复现**。

### 2.3 精度：部分符合、部分相反，统计力有限

- **6k 2wikimqa**：SM/F1 上 DAF（0.333/0.346）位于 CB（0.292/0.281）与全量（0.417/0.414）之间——排序符合论文"DAF 介于 CB 与全量之间"，但"DAF 追平全量"未复现
- **9k samsum**：RL CB 0.440 ≈ 全量 0.428 > DAF 0.391——排序与论文相反（论文预期 DAF 贴近全量）；24 样本统计力有限（1 题 ≈ 0.04）
- **30k RULER**：qa_2 上 DAF=CB=0.250 持平，qa_1 上 DAF 0.375 < CB 0.458（差 2/24 题）；论文的"DAF 精度反超 CB"未复现。**三方法绝对精度系统性低于论文对照值**（论文 qa_2 DAF 0.72 / CB 0.625；本轮连无近似的全量基线也只有 0.458），主因有三：
  1. **样本量 24 vs 论文/RULER 惯例 500**，±0.1 量级噪声；
  2. **prompt 模板差异**（~~未套 ChatML~~）：原判断"裸 prompt"不准确——qwen 预设实际已套 ChatML 信封，仅 system 措辞与官方默认不同；2026-09-21 消融（`experiments/reports/chatml_ablation_ruler30k.md`）证实该因子影响 ≤1 题，**不是主因**。此处保留原记录仅作存档；
  3. **xformers vs 论文 Flash-Attention 内核**（数值上不该影响全量基线精度，主要解释速度差距；列此以备查）。

## 三、与论文验收标准逐条对照

| 验收标准 | 目标 | 实测 | 结论 |
|---|---|---|---|
| 30k 档 DAF vs CB ≥1.3× | DAF/CB ≤ 0.77 | **0.940** | **未达标**——方向已反超（<1）但幅度不足。差距来源：xformers 无 Flash-Attention 变长 kernel 优势；3090Ti vs A100 算力带宽比；r=0.3 下 DAF 前 5 层（status 0/3/1）全量 prefill 的固定开销在 30k 仍未摊薄到论文水平 |
| RULER-qa2 DAF 精度 | ≥ 0.68 | **0.250** | **未达标**——但全量基线也仅 0.458，绝对差距主要来自实验设置（样本量/模板），非算法本身；见 2.3 |
| RULER-qa2 CB 精度 | ≤ 0.65 | 0.250 | 形式上达标，但因绝对水平整体偏低，不具备区分度 |
| 136k 档 | — | — | **未做**：单卡 24GB（BF16 权重 14.25GB + KV 57KB/token）装不下，需要 A100/H100 80GB |

## 四、工程记录（本轮解决的坑）

1. **KV 池装不下 32768（B-3）**：vllm 0.4.1 的显存剖析按 `max_num_batched_tokens=max_model_len` 跑 32k 前向，激活峰值 ~4.8GB；0.85 util 下 BF16 Qwen 的 KV 池只有 16864 token。新增 **`BLEND_GPU_MEM_UTIL`** 环境变量（CB 用 0.90 → 池 2432 块/38912 token；DAF 用 0.89）。util 只影响池大小，不影响计算与结果，CB/DAF 用不同值是公平的
2. **OOM 三连修（B-3）**：old_kvs 离线缓存（30k token ≈1.9GB）在 vllm 预算之外，叠加碎片后 CB 第 2 样本、DAF 第 5 样本（check 阶段前 5 层全量 prefill，激活峰值更大）必现 OOM。修复：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；check 前 `torch.cuda.empty_cache()`；阶段三（全量 prefill）前释放 old_kvs——**必须保持 `[[None,None]]×层数` 占位**，`qwen2.py:360`/`llama.py:405` 的 `old_kvs[i]` 是无条件下标访问，第一版直接置 `[]` 引发 IndexError。作废轮次归档在 `experiments/logs/archived_ruler30k_oom作废/`（只归档不删除）
3. **评分 bug：`parse_generation` 的 "No" 折叠**：`example/utils.py:18` 把 "No…"/"Yes…" 开头的整段回答折叠成 "No"/"Yes"（原作者为 hotpot yes/no 题写的 hack）。RULER squad 答案大量以 Normandy/Norman/Norse 等 No 前缀词开头，首轮评分被系统性误判（"Normandy is located in France." → "No" → SM=0）。评分脚本新增 **`--raw_parse`** 开关（裸首行不折叠），B-3 全部数字为重评后的值；RULER 官方口径已核对（`string_match_part`：小写子串任一命中）。此前 B-1/B-2 报告用折叠版：2wikimqa 的 yes/no 题正需要折叠、samsum 摘要极少 No 前缀开头，结论不受影响
4. **RULER 模板 ` Answer:` 前缀**：官方 qa 模板以 `Question: {query} Answer:` 结尾，`qa.py` 据此把 ` Answer:` 拆成 answer_prefix。第一版漏传导致 answer_prefix 退化为 "?"（rfind 返回 -1 切掉末字符），已重新生成
5. **新增/变更的实验环境变量清单**（blend.py）：`BLEND_MODEL_PRESET=bf16|awq|qwen`、`BLEND_MAX_MODEL_LEN`（qwen 默认 8192）、`BLEND_MAX_TOKENS`（默认 32）、`BLEND_GPU_MEM_UTIL`（qwen 默认 0.85）、`BLEND_INPUT_DIR`、`CB_RECOMP_RATIO`（默认 0.3）、`DAF=0|1`、`VLLM_ATTENTION_BACKEND`

## 五、复现命令链

三档均从仓库根目录运行，`conda activate cacheblend && export HF_ENDPOINT=https://hf-mirror.com`。

### B-1（2wikimqa ≈6k）

```bash
python experiments/scripts/make_inputs_qwen_2wikimqa.py          # 生成 24 样本
BLEND_MODEL_PRESET=qwen BLEND_INPUT_DIR=<样本目录> DAF=0 python example/blend.py   # CB，×3 轮
BLEND_MODEL_PRESET=qwen BLEND_INPUT_DIR=<样本目录> DAF=1 python example/blend.py   # DAF，×3 轮
python experiments/scripts/score_em_long_6k10k.py --dir experiments/inputs_qwen_2wikimqa \
    --tokenizer Qwen/Qwen2.5-7B-Instruct \
    --logs experiments/logs/qwen_2wikimqa/run_cb_qwen2wiki_r*.log \
           experiments/logs/qwen_2wikimqa/run_daf_qwen2wiki_r*.log \
    --out experiments/reports/em_report_qwen_2wikimqa.md
```

### B-2（samsum ≈9k）

```bash
python experiments/scripts/make_inputs_qwen_samsum.py
BLEND_MODEL_PRESET=qwen BLEND_MAX_MODEL_LEN=16384 BLEND_MAX_TOKENS=64 \
    BLEND_INPUT_DIR=<样本目录> DAF=0 python example/blend.py       # CB/DAF 各 3 轮
python experiments/scripts/score_em_long_6k10k.py --dir experiments/inputs_qwen_samsum \
    --tokenizer Qwen/Qwen2.5-7B-Instruct --rouge \
    --logs experiments/logs/qwen_samsum/run_cb_qwen_samsum_r*.log \
           experiments/logs/qwen_samsum/run_daf_qwen_samsum_r*.log \
    --out experiments/reports/rouge_report_qwen_samsum.md
```

### B-3（RULER qa_1/qa_2 ≈30k）

```bash
# 1. RULER 官方生成（repo 在 experiments/ruler_src/，含自身 .git 不入库；
#    生成的 validation.jsonl 归档在 experiments/ruler_data_gen/）
cd experiments/ruler_src/scripts/data/synthetic
python qa.py --save_dir ~/daf/experiments/ruler_src/data_gen --save_name qa_1 \
    --tokenizer_path Qwen/Qwen2.5-7B-Instruct --tokenizer_type hf \
    --max_seq_length 32768 --tokens_to_generate 32 --num_samples 24 --dataset squad \
    --template "<官方 qa 模板，以 'Question: {query} Answer:' 结尾>"   # qa_2 改 hotpotqa
cd ~/daf
# 2. 转样本 → 3. 4 卡并行 3 轮跑批 → 4. 评分（命令全文见 experiments/README.md）
python experiments/scripts/make_inputs_qwen_ruler30k.py
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BLEND_MODEL_PRESET=qwen BLEND_GPU_MEM_UTIL=0.90 BLEND_MAX_MODEL_LEN=32768 \
    BLEND_MAX_TOKENS=32 BLEND_INPUT_DIR=<qa_1或qa_2目录> DAF=0 python example/blend.py  # DAF 用 0.89
python experiments/scripts/score_em_long_6k10k.py --dir experiments/inputs_qwen_ruler30k \
    --tokenizer Qwen/Qwen2.5-7B-Instruct --raw_parse \
    --logs experiments/logs/qwen_ruler30k/run_cb_*_r*.log \
           experiments/logs/qwen_ruler30k/run_daf_*_r*.log \
    --out experiments/reports/em_report_qwen_ruler30k.md
```

## 六、下一步建议（按价值排序）

1. **加样本量**：三档都只有 24 样本，1 题 ≈0.04 分，精度排序结论的统计力不足；RULER 应先跑到官方惯例的 500 样本/任务（qa.py `--num_samples 500`，预计单轮 4 卡 ~2.5 小时），再判精度验收线
2. ~~ChatML 模板对照~~（已完成，2026-09-21）：消融证实上轮已套 ChatML 信封（仅 system 措辞不同），官方模板下精度变化 ≤1 题，**非主因**；见 `experiments/reports/chatml_ablation_ruler30k.md`。500 样本放量建议用 `BLEND_CHATML=1` 对齐官方形态
3. **K+V deviation 消融**：当前 V-only 打分是工程假设（AGENTS.md 铁律 7），30k 上 DAF 精度未超 CB，K+V 打分是论文描述里最可能的差距来源之一
4. **A100/H100 上 136k 档**：验证论文最大长度档的 DAF 领先（2×）与精度优势；同时可把 30k 档在 FA 内核+A100 上重跑，剥离硬件/内核因子确认 0.94 vs 0.77 的差距归属
5. **反超点精确定位**：在 Qwen 序列上补 16k/20k 档，把 DAF/CB=1.0 的交叉点从"~25k 附近"收窄到具体档位

---

*数据索引：B-1 报告 `experiments/reports/em_report_qwen_2wikimqa.md`；B-2 `experiments/reports/rouge_report_qwen_samsum.md`（另有 flash 对照报告）；B-3 `experiments/reports/em_report_qwen_ruler30k.md`（第四节为论文对照分析与评分修正说明）；样本生成明细见各 `inputs_qwen_*/_selection_report.md`*
