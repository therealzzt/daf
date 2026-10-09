# 6k–10k 长上下文实验评分报告

- 样本目录：`experiments/inputs_long_30k_buriedqa`（12 个样本）
- 评分：EM（精确匹配，多答案取最大）+ token-F1（多答案取最大）
- 归一化：小写/去冠词/去标点/Yes-No 规整（`example/utils.py`）
- 输入日志：`run_cb_long30k_r1.log`, `run_cb_long30k_r2.log`, `run_cb_long30k_r3.log`, `run_daf_long30k_r1.log`, `run_daf_long30k_r2.log`, `run_daf_long30k_r3.log`

## 一、逐样本明细（质量指标为多轮均值，TTFT 为多轮中位数）

| 样本 | 来源 | 长度 | 方法 | EM | SM | F1 | TTFT缓存(s) | TTFT全量(s) | 缓存答案 | 全量答案 |
|---|---|---|---|---|---|---|---|---|---|---|
| musique_s_000_at020_len27946.json | musique_s.json | 27946 | CacheBlend | 0.00 | 1.00 | 0.75 | 1.916 | 7.494 |  Exeter College, Oxford. |  Exeter College, Oxford. |
| musique_s_000_at020_len27946.json | musique_s.json | 27946 | DAF | 0.00 | 1.00 | 0.75 | 2.843 | 7.493 |  Exeter College, Oxford |  Exeter College, Oxford. |
| musique_s_000_at050_len31299.json | musique_s.json | 31299 | CacheBlend | 0.00 | 1.00 | 0.75 | 2.249 | 8.754 |  Exeter College, Oxford. |  Exeter College, Oxford |
| musique_s_000_at050_len31299.json | musique_s.json | 31299 | DAF | 0.00 | 0.00 | 0.00 | 3.225 | 8.768 |  Guadalajara, Spain. |  Exeter College, Oxford |
| musique_s_000_at080_len30856.json | musique_s.json | 30856 | CacheBlend | 0.00 | 0.00 | 0.00 | 2.221 | 8.625 |  Eton and Trinity Cambridge. |  Exeter College, Oxford. |
| musique_s_000_at080_len30856.json | musique_s.json | 30856 | DAF | 0.00 | 0.00 | 0.00 | 3.152 | 8.634 |  Eton and Cambridge University |  Exeter College, Oxford. |
| musique_s_001_at020_len29206.json | musique_s.json | 29206 | CacheBlend | 1.00 | 1.00 | 1.00 | 2.024 | 7.981 |  Charles county |  Charles county |
| musique_s_001_at020_len29206.json | musique_s.json | 29206 | DAF | 0.00 | 0.00 | 0.00 | 2.956 | 8.000 |  Southern Maryland |  Charles county |
| musique_s_001_at050_len28660.json | musique_s.json | 28660 | CacheBlend | 0.00 | 0.00 | 0.00 | 1.983 | 7.776 |  Hughesville, Maryland |  Hughesville, Maryland. |
| musique_s_001_at050_len28660.json | musique_s.json | 28660 | DAF | 0.00 | 0.00 | 0.00 | 2.874 | 7.803 |  Southern Maryland |  Hughesville, Maryland. |
| musique_s_001_at080_len28562.json | musique_s.json | 28562 | CacheBlend | 1.00 | 1.00 | 1.00 | 1.982 | 7.736 |  Charles County |  Charles county |
| musique_s_001_at080_len28562.json | musique_s.json | 28562 | DAF | 0.00 | 0.00 | 0.09 | 2.868 | 7.740 |  The answer is in the United States, spe |  Charles county |
| wikimqa_s_000_at020_len30613.json | wikimqa_s.json | 30613 | CacheBlend | 0.00 | 0.00 | 0.00 | 2.169 | 8.510 |  Marie de Medici, born in Paris. |  Marie de Medici, daughter of Henry IV o |
| wikimqa_s_000_at020_len30613.json | wikimqa_s.json | 30613 | DAF | 0.00 | 0.00 | 0.00 | 3.139 | 8.513 |  Born in milan. |  Marie de Medici, daughter of Henry IV o |
| wikimqa_s_000_at050_len31156.json | wikimqa_s.json | 31156 | CacheBlend | 0.00 | 0.00 | 0.00 | 2.242 | 8.750 |  Born in Austria. |  She was born in Naples. |
| wikimqa_s_000_at050_len31156.json | wikimqa_s.json | 31156 | DAF | 0.00 | 0.00 | 0.00 | 3.223 | 8.743 |  Born in Hungary. |  She was born in Naples. |
| wikimqa_s_000_at080_len29920.json | wikimqa_s.json | 29920 | CacheBlend | 0.00 | 0.00 | 0.00 | 2.090 | 8.261 |  Born in Vienna. |  Born in Naples. |
| wikimqa_s_000_at080_len29920.json | wikimqa_s.json | 29920 | DAF | 0.00 | 0.00 | 0.00 | 3.063 | 8.254 |  Born in islamabad capital territory. |  Born in Naples. |
| wikimqa_s_001_at020_len28335.json | wikimqa_s.json | 28335 | CacheBlend | 0.00 | 0.00 | 0.24 | 1.933 | 7.665 |  John Bourchier, 1st Baron Berners. |  Angelitha Wass is Sobe's grandchild. |
| wikimqa_s_001_at020_len28335.json | wikimqa_s.json | 28335 | DAF | 0.00 | 0.00 | 0.00 | 2.842 | 7.667 |  Matthieu Chedid (Matthieu is the French |  Angelitha Wass is Sobe's grandchild. |
| wikimqa_s_001_at050_len31074.json | wikimqa_s.json | 31074 | CacheBlend | 0.00 | 0.00 | 0.00 | 2.234 | 8.693 |  Ferdinand Maximilian, Count of Isenburg |  Elizabeth, daughter of Celje, is Sobe's |
| wikimqa_s_001_at050_len31074.json | wikimqa_s.json | 31074 | DAF | 0.00 | 0.00 | 0.00 | 3.195 | 8.689 |  Queen Victoria of the United Kingdom. |  Elizabeth, daughter of Celje, is Sobe's |
| wikimqa_s_001_at080_len28565.json | wikimqa_s.json | 28565 | CacheBlend | 0.00 | 0.00 | 0.00 | 1.983 | 7.743 |  The grandchild of Sobe (Sister Of Saint |  John Angelos, gallo-Roman senator, is S |
| wikimqa_s_001_at080_len28565.json | wikimqa_s.json | 28565 | DAF | 0.00 | 0.00 | 0.00 | 2.864 | 7.745 |  The grandchild of Sobe (Sister Of Saint |  John Angelos, gallo-Roman senator, is S |

## 二、按方法聚合（每轮一行，再给出跨轮中位数）

| 方法 | 轮次 | EM | SM | F1 | TTFT缓存均值(s) | TTFT全量均值(s) | 缓存加速比 |
|---|---|---|---|---|---|---|---|
| CacheBlend | r1 | 0.167 | 0.333 | 0.311 | 2.083 | 8.164 | 3.92× |
| CacheBlend | r2 | 0.167 | 0.333 | 0.311 | 2.084 | 8.164 | 3.92× |
| CacheBlend | r3 | 0.167 | 0.333 | 0.311 | 2.088 | 8.175 | 3.91× |
| **CacheBlend** | **中位** | **0.167** | **0.333** | **0.311** | **2.084** | **8.164** | **3.92×** |
| DAF | r1 | 0.000 | 0.083 | 0.070 | 3.021 | 8.180 | 2.71× |
| DAF | r2 | 0.000 | 0.083 | 0.070 | 3.018 | 8.170 | 2.71× |
| DAF | r3 | 0.000 | 0.083 | 0.070 | 3.020 | 8.170 | 2.71× |
| **DAF** | **中位** | **0.000** | **0.083** | **0.070** | **3.020** | **8.170** | **2.71×** |

## 三、按数据集×方法聚合（跨轮中位数；QA 数据集看 SM/F1，samsum 摘要看 F1）

| 数据集 | 样本数 | 方法 | EM中位 | SM中位 | F1中位 | TTFT缓存中位(s) |
|---|---|---|---|---|---|---|
| musique_s.json | 6 | CacheBlend | 0.000 | 1.000 | 0.750 | 2.004 |
| musique_s.json | 6 | DAF | 0.000 | 0.000 | 0.000 | 2.912 |
| wikimqa_s.json | 6 | CacheBlend | 0.000 | 0.000 | 0.000 | 2.130 |
| wikimqa_s.json | 6 | DAF | 0.000 | 0.000 | 0.000 | 3.100 |
