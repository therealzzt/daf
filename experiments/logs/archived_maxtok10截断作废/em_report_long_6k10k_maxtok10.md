# 6k–10k 长上下文实验评分报告

- 样本目录：`inputs_long_6k10k_official`（24 个样本）
- 评分：EM（精确匹配，多答案取最大）+ token-F1（多答案取最大）
- 归一化：小写/去冠词/去标点/Yes-No 规整（`example/utils.py`）
- 输入日志：`run_cb_long6k10k_r1.log`, `run_cb_long6k10k_r2.log`, `run_cb_long6k10k_r3.log`, `run_daf_long6k10k_r1.log`, `run_daf_long6k10k_r2.log`, `run_daf_long6k10k_r3.log`

## 一、逐样本明细（按日志聚合：EM/F1 为多轮均值，TTFT 为多轮中位数）

| 样本 | 来源 | 长度 | 方法 | EM | F1 | TTFT缓存(s) | TTFT全量(s) | 缓存答案 | 全量答案 |
|---|---|---|---|---|---|---|---|---|---|
| musique_s_019_len6734.json | musique_s.json | 6734 | CacheBlend | 0.00 | 0.00 | 0.355 | 1.369 |  The New Orleans Voodoo were the team An |  Major League Baseball. |
| musique_s_019_len6734.json | musique_s.json | 6734 | DAF | 0.00 | 0.00 | 0.652 | 1.369 |  Kris Benson was Anna Benson's |  Major League Baseball. |
| musique_s_040_len6457.json | musique_s.json | 6457 | CacheBlend | 0.00 | 0.10 | 0.314 | 1.285 |  The death of John Curtin caused Robert  |  Robert Menzies became Prime Minister af |
| musique_s_040_len6457.json | musique_s.json | 6457 | DAF | 0.00 | 0.10 | 0.549 | 1.286 |  The death of John Curtin caused Robert  |  Robert Menzies became Prime Minister af |
| musique_s_045_len6408.json | musique_s.json | 6408 | CacheBlend | 0.00 | 0.00 | 0.313 | 1.281 |  Defence Research and Development Organi |  Defence research and development organi |
| musique_s_045_len6408.json | musique_s.json | 6408 | DAF | 0.00 | 0.00 | 0.549 | 1.283 |  The manufacturer of Agni-V was founded  |  Defence research and development organi |
| musique_s_063_len6522.json | musique_s.json | 6522 | CacheBlend | 0.00 | 0.00 | 0.316 | 1.305 |  Caceres, Extremadura, Spain |  Portugal (Vila Franca de Xira) |
| musique_s_063_len6522.json | musique_s.json | 6522 | DAF | 0.00 | 0.00 | 0.555 | 1.311 |  Caceres, Extremadura, Spain |  Portugal (Vila Franca de Xira) |
| musique_s_070_len6009.json | musique_s.json | 6009 | CacheBlend | 0.00 | 0.00 | 0.295 | 1.207 |  The person who wrote about the rioting  |  James Watt was educated in Birmingham. |
| musique_s_070_len6009.json | musique_s.json | 6009 | DAF | 0.00 | 0.00 | 0.507 | 1.216 |  The person who wrote about the rioting  |  James Watt was educated in Birmingham. |
| musique_s_093_len6354.json | musique_s.json | 6354 | CacheBlend | 0.00 | 0.67 | 0.312 | 1.269 |  Charles university in Prague. |  Charles University |
| musique_s_093_len6354.json | musique_s.json | 6354 | DAF | 0.00 | 0.29 | 0.549 | 1.278 |  University of Prague |  Charles University |
| musique_s_110_len6604.json | musique_s.json | 6604 | CacheBlend | 0.00 | 0.44 | 0.318 | 1.309 |  Lillian Disney |  Ub Iwerks created Mickey Mouse' |
| musique_s_110_len6604.json | musique_s.json | 6604 | DAF | 0.00 | 0.29 | 0.560 | 1.317 |  Lillian Disney created Mickey Mouse's |  Ub Iwerks created Mickey Mouse' |
| musique_s_138_len6288.json | musique_s.json | 6288 | CacheBlend | 0.00 | 0.10 | 0.312 | 1.266 |  IEEE Ernst Weber Managerial Leadership  |  Universal acclaim from critics. |
| musique_s_138_len6288.json | musique_s.json | 6288 | DAF | 0.00 | 0.10 | 0.541 | 1.275 |  IEEE Ernst Weber Managerial Leadership  |  Universal acclaim from critics. |
| samsum_032_len8287.json | samsum.json | 8287 | CacheBlend | 0.00 | 0.10 | 0.481 | 1.695 |  Dreams reflect subconsciousness. Strang |  Dreams reflect subconsciousness. |
| samsum_032_len8287.json | samsum.json | 8287 | DAF | 0.00 | 0.10 | 0.762 | 1.705 |  Dreams reflect subconsciousness. Strang |  Dreams reflect subconsciousness. |
| samsum_062_len7444.json | samsum.json | 7444 | CacheBlend | 0.00 | 0.26 | 0.395 | 1.522 |  Natalie wants to go to the new club on |  Natalie wants to go to the new club on |
| samsum_062_len7444.json | samsum.json | 7444 | DAF | 0.00 | 0.26 | 0.663 | 1.532 |  Natalie wants to go to the new club on |  Natalie wants to go to the new club on |
| samsum_080_len8753.json | samsum.json | 8753 | CacheBlend | 0.00 | 0.42 | 0.462 | 1.814 |  New song of Charlie Puth shared by Jasm |  New song of Charlie Puth. |
| samsum_080_len8753.json | samsum.json | 8753 | DAF | 0.00 | 0.29 | 0.774 | 1.824 |  New song of Charlie Puth shared. Both l |  New song of Charlie Puth. |
| samsum_096_len7109.json | samsum.json | 7109 | CacheBlend | 0.00 | 0.30 | 0.348 | 1.425 |  Tom wears a red jacket. |  Tom and Elena are near Jeffrey, wearing |
| samsum_096_len7109.json | samsum.json | 7109 | DAF | 0.00 | 0.35 | 0.625 | 1.434 |  Elena points out Tom's red jacket. |  Tom and Elena are near Jeffrey, wearing |
| samsum_106_len6352.json | samsum.json | 6352 | CacheBlend | 0.00 | 0.18 | 0.314 | 1.270 |  Package leaving tomorrow. |  Package leaving tomorrow. |
| samsum_106_len6352.json | samsum.json | 6352 | DAF | 0.00 | 0.25 | 0.559 | 1.278 |  Package leaves warehouse, delivered tom |  Package leaving tomorrow. |
| samsum_108_len9546.json | samsum.json | 9546 | CacheBlend | 0.00 | 0.08 | 0.530 | 1.999 |  Road blocked by work vans. Avoid if pos |  Road repairs blocking route to Waitrose |
| samsum_108_len9546.json | samsum.json | 9546 | DAF | 0.00 | 0.12 | 0.874 | 2.003 |  Road blocked by work vans. Avoid it. |  Road repairs blocking route to Waitrose |
| samsum_124_len6001.json | samsum.json | 6001 | CacheBlend | 0.00 | 0.21 | 0.328 | 1.178 |  Pegah will be back at midnight. Miriam |  Pegah returns at 10:00 |
| samsum_124_len6001.json | samsum.json | 6001 | DAF | 0.00 | 0.23 | 0.543 | 1.180 |  Pegah will be back at 10: |  Pegah returns at 10:00 |
| samsum_168_len6663.json | samsum.json | 6663 | CacheBlend | 0.00 | 0.33 | 0.342 | 1.370 |  They wish Adam good luck before an exam |  Good luck, Adam. |
| samsum_168_len6663.json | samsum.json | 6663 | DAF | 0.00 | 0.25 | 0.590 | 1.374 |  They wish Adam good luck. |  Good luck, Adam. |
| wikimqa_s_010_len6665.json | wikimqa_s.json | 6665 | CacheBlend | 0.00 | 0.00 | 0.321 | 1.372 |  The Secret Invasion. (born 18 |  The Secret Invasion (1964) |
| wikimqa_s_010_len6665.json | wikimqa_s.json | 6665 | DAF | 0.00 | 0.00 | 0.571 | 1.373 |  The Secret Invasion (1941) |  The Secret Invasion (1964) |
| wikimqa_s_032_len6517.json | wikimqa_s.json | 6517 | CacheBlend | 0.00 | 0.77 | 0.316 | 1.292 |  Goring-on-Thames. |  At his home in Goring-on-Th |
| wikimqa_s_032_len6517.json | wikimqa_s.json | 6517 | DAF | 0.00 | 0.00 | 0.557 | 1.293 |  Christmas Day 2016. |  At his home in Goring-on-Th |
| wikimqa_s_060_len6451.json | wikimqa_s.json | 6451 | CacheBlend | 0.00 | 0.14 | 0.315 | 1.289 |  The spouse of Waylon Jennings for the s |  Jessi Colter |
| wikimqa_s_060_len6451.json | wikimqa_s.json | 6451 | DAF | 0.00 | 0.14 | 0.551 | 1.290 |  The spouse of Waylon Jennings during th |  Jessi Colter |
| wikimqa_s_088_len6078.json | wikimqa_s.json | 6078 | CacheBlend | 0.00 | 0.18 | 0.298 | 1.219 |  Abu Dhabi |  Abu Dhabi, UAE |
| wikimqa_s_088_len6078.json | wikimqa_s.json | 6078 | DAF | 0.00 | 0.50 | 0.510 | 1.218 |  Hawaii. (Note: Keōpūol |  Abu Dhabi, UAE |
| wikimqa_s_108_len6910.json | wikimqa_s.json | 6910 | CacheBlend | 0.00 | 0.00 | 0.347 | 1.414 |  University College - Dublin |  University of Texas |
| wikimqa_s_108_len6910.json | wikimqa_s.json | 6910 | DAF | 0.00 | 0.00 | 0.602 | 1.413 |  The University of Texas graduated the B |  University of Texas |
| wikimqa_s_110_len6584.json | wikimqa_s.json | 6584 | CacheBlend | 0.00 | 0.67 | 0.319 | 1.314 |  Moment Of Danger was released earlier. |  Moment Of Danger (1960 |
| wikimqa_s_110_len6584.json | wikimqa_s.json | 6584 | DAF | 0.00 | 0.67 | 0.559 | 1.312 |  Moment Of Danger was released earlier. |  Moment Of Danger (1960 |
| wikimqa_s_123_len6367.json | wikimqa_s.json | 6367 | CacheBlend | 0.00 | 0.73 | 0.314 | 1.278 |  The Comedians Of Comedy came out first |  The Comedians Of Comedy came out first |
| wikimqa_s_123_len6367.json | wikimqa_s.json | 6367 | DAF | 0.00 | 0.73 | 0.549 | 1.278 |  The Comedians Of Comedy came out first |  The Comedians Of Comedy came out first |
| wikimqa_s_196_len6767.json | wikimqa_s.json | 6767 | CacheBlend | 0.00 | 0.40 | 0.323 | 1.378 |  Duke Ferdinand of Wurttemberg. |  Frederick II, Duke of Wurttemberg |
| wikimqa_s_196_len6767.json | wikimqa_s.json | 6767 | DAF | 0.00 | 0.00 | 0.587 | 1.377 |  Marie Louise's maternal grandfather is  |  Frederick II, Duke of Wurttemberg |

## 二、按方法聚合（每轮一行，再给出跨轮中位数）

| 方法 | 轮次 | EM | F1 | TTFT缓存均值(s) | TTFT全量均值(s) | 缓存加速比 |
|---|---|---|---|---|---|---|
| CacheBlend | r1 | 0.000 | 0.253 | 0.346 | 1.382 | 4.00× |
| CacheBlend | r2 | 0.000 | 0.253 | 0.345 | 1.377 | 3.99× |
| CacheBlend | r3 | 0.000 | 0.253 | 0.345 | 1.380 | 4.00× |
| **CacheBlend** | **中位** | **0.000** | **0.253** | **0.345** | **1.380** | **4.00×** |
| DAF | r1 | 0.000 | 0.194 | 0.597 | 1.384 | 2.32× |
| DAF | r2 | 0.000 | 0.194 | 0.597 | 1.385 | 2.32× |
| DAF | r3 | 0.000 | 0.194 | 0.597 | 1.384 | 2.32× |
| **DAF** | **中位** | **0.000** | **0.194** | **0.597** | **1.384** | **2.32×** |
