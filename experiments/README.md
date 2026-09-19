# experiments/ —— DAF 长上下文实验工作区

> 2026-09-16 整理：把此前散落在仓库根目录的实验脚本、样本、日志、报告统一收入此处。
> 所有脚本**从仓库根目录（~/daf）运行**，路径均为相对仓库根。

## 目录结构

```
experiments/
├── README.md                     ← 本文件
├── scripts/                      ← 辅助工具（生成样本 / 评分 / 幂等续跑 / GPU 看门狗）
│   ├── make_inputs_long_6k10k_official.py   6k–10k 样本生成（从 inputs/ 官方数据集挑选转换）
│   ├── make_inputs_long_30k_buriedqa.py     30k 样本生成（RULER 风格埋答案构造）
│   ├── score_em_long_6k10k.py               EM/SM/F1 评分 + 按方法/数据集聚合（两档通用）
│   ├── run_long6k10k_resume.sh              6k–10k 档幂等续跑（每轮 24 样本，nohup 可断线）
│   ├── run_long30k_resume.sh                30k 档幂等续跑（每轮 12 样本，AWQ 预设）
│   └── run_long30k_wait_and_start.sh        GPU 看门狗：卡被占满时排队，空出自动开跑
│
├── inputs_long_6k10k_official/   ← 6k–10k 档 24 样本（musique/wikimqa/samsum 各 8，带标准答案）
│   └── _selection_report.md                 挑选明细（出处、长度、跳过原因）
├── inputs_long_30k_buriedqa/     ← 30k 档 12 样本（4 题 × 埋深 20%/50%/80%，带标准答案）
│   └── _selection_report.md                 构造明细
│
├── logs/                         ← 全部运行日志（按档分目录；含 .out 为续跑/看门狗日志）
│   ├── short_2k4k_awq/                      2–4k 档 AWQ（daf1.0 复现，对应 docs/daf_experiment_10samples.md）
│   ├── short_2k4k_bf16/                     2–4k 档 bf16（daf1.1）
│   ├── long6k10k/                           6k–10k 档 6 轮（daf1.2）
│   ├── long30k/                             30k 档 6 轮 + 看门狗日志（daf1.3）
│   └── archived_maxtok10截断作废/           ⚠️ 作废数据：max_tokens=10 截断答案致 EM 失效，
│                                              归档留档勿用于结论（见 docs/daf1.2.md 第三节教训）
│
└── reports/                      ← 评分报告（score 脚本生成）
    ├── em_report_long_6k10k.md
    └── em_report_long_30k.md
```

## 各档结论速查（详细分析见 docs/daf1.1~1.3.md）

| 档位 | 样本 | 模型 | CacheBlend | DAF | DAF/CB 时长比 |
|---|---|---|---|---|---|
| 2–4k（1.1） | inputs/1~10.json | bf16 | 加速 3.8× | 加速 2.2× | 1.75× |
| 6–10k（1.2） | inputs_long_6k10k_official | bf16 | 加速 4.0× | 加速 2.3× | 1.73× |
| 30k（1.3） | inputs_long_30k_buriedqa | AWQ | 加速 3.9× | 加速 2.7× | **1.45×（随长度收窄，趋势符合论文）** |
| 6k（B-1） | inputs_qwen_2wikimqa（LongBench 官方） | qwen | 加速 2.66× | 加速 2.32× | 1.14× |
| 9k（B-2） | inputs_qwen_samsum（LongBench 官方） | qwen | 加速 2.61× | 加速 2.34× | 1.11× |
| 30k（B-3） | inputs_qwen_ruler30k（RULER 官方 qa_1/qa_2 各 24） | qwen | 加速 2.46× | 加速 2.63× | **0.94×（DAF 首次反超 CB）** |

## 目标 B 第三档（RULER qa_1/qa_2 ≈30k，Qwen2.5）命令链

```bash
conda activate cacheblend && cd ~/daf
# 1. 生成 RULER 数据（repo 已 clone 在 experiments/ruler_src/（含自身 .git，不入库），
#    源数据 squad.json/hotpotqa.json 在 scripts/data/synthetic/json/；
#    生成命令见报告第四节；生成的 validation.jsonl 归档副本在 experiments/ruler_data_gen/）
python experiments/scripts/make_inputs_qwen_ruler30k.py            # 2. 转成 blend 样本
# 3. 跑批（4 卡并行 × 3 轮；qa_1/qa_2 各 24 样本放 /tmp/blend_ruler_qa{1,2} 软链）
export HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 BLEND_MODEL_PRESET=qwen BLEND_GPU_MEM_UTIL=0.90 \
    BLEND_MAX_MODEL_LEN=32768 BLEND_MAX_TOKENS=32 \
    BLEND_INPUT_DIR=/tmp/blend_ruler_qa1 DAF=0 python example/blend.py   # CB 用 0.90，DAF 用 0.89
# 4. 评分（RULER 必须 --raw_parse，否则 Normandy 等 No 前缀答案被误判）
python experiments/scripts/score_em_long_6k10k.py --dir experiments/inputs_qwen_ruler30k \
    --tokenizer Qwen/Qwen2.5-7B-Instruct --raw_parse \
    --logs experiments/logs/qwen_ruler30k/run_cb_*_r*.log \
           experiments/logs/qwen_ruler30k/run_daf_*_r*.log \
    --out experiments/reports/em_report_qwen_ruler30k.md
```

## 复现一条命令链（以 30k 档为例）

```bash
conda activate cacheblend && cd ~/daf
python experiments/scripts/make_inputs_long_30k_buriedqa.py     # 1. 生成样本
nohup bash experiments/scripts/run_long30k_resume.sh \
    > experiments/logs/long30k/resume.out 2>&1 &                # 2. 跑 6 轮（可断线）
python experiments/scripts/score_em_long_6k10k.py \
    --dir experiments/inputs_long_30k_buriedqa \
    --logs experiments/logs/long30k/run_cb_long30k_r*.log \
           experiments/logs/long30k/run_daf_long30k_r*.log \
    --out experiments/reports/em_report_long_30k.md             # 3. 评分
```

## 相关文档

- `docs/daf1.1.md` —— bf16 改造 + 2–4k 结果（与论文差别分析）
- `docs/daf1.2.md` —— 6k–10k 档：样本管线、EM/SM/F1 指标定义、作废数据教训
- `docs/daf1.3.md` —— 30k 档：DAF/CB 比值首次收窄、与论文 30k 档对比
- `docs/daf1.4.md` —— 目标 B 三档（Qwen2.5 官方数据集 6k/9k/30k）总记录：DAF/CB 0.94 首次反超
- `docs/daf1.0.md` / `docs/daf_experiment_10samples.md` —— DAF 算法实现与 AWQ 基线
- `docs/服务器复现指南.md` —— 环境搭建权威指南
