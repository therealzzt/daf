# 6k–10k 长上下文实验评分报告

- 样本目录：`experiments/inputs_qwen_2wikimqa`（24 个样本）
- 评分：EM（精确匹配，多答案取最大）+ token-F1（多答案取最大）
- 归一化：小写/去冠词/去标点/Yes-No 规整（`example/utils.py`）
- 输入日志：`run_cb_flash_qwen2wiki_r1.log`, `run_daf_flash_qwen2wiki_r1.log`

## 一、逐样本明细（质量指标为多轮均值，TTFT 为多轮中位数）

| 样本 | 来源 | 长度 | 方法 | EM | SM | F1 | TTFT缓存(s) | TTFT全量(s) | 缓存答案 | 全量答案 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2wikimqa_008_len6111.json | 2wikimqa.jsonl | 6111 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.446 | 1.087 | Not specified | Abbey Theatre |
| 2wikimqa_008_len6111.json | 2wikimqa.jsonl | 6111 | DAF | 0.00 | 1.00 | 0.80 | 0.552 | 1.094 | St Patrick's College, Dublin. | Abbey Theatre |
| 2wikimqa_013_len5407.json | 2wikimqa.jsonl | 5407 | CacheBlend | 0.00 | 0.00 | 0.50 | 0.363 | 0.979 | 1806 | 1872 |
| 2wikimqa_013_len5407.json | 2wikimqa.jsonl | 5407 | DAF | 0.00 | 0.00 | 0.50 | 0.430 | 0.984 | 1846 | 1872 |
| 2wikimqa_019_len5555.json | 2wikimqa.jsonl | 5555 | CacheBlend | 0.00 | 1.00 | 0.80 | 0.379 | 1.000 | Mysore, India | Mysore, India |
| 2wikimqa_019_len5555.json | 2wikimqa.jsonl | 5555 | DAF | 0.00 | 1.00 | 0.80 | 0.449 | 1.007 | Mysore, India | Mysore, India |
| 2wikimqa_034_len6441.json | 2wikimqa.jsonl | 6441 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.439 | 1.170 | Ringo-en no shōjo | Wooden Crosses |
| 2wikimqa_034_len6441.json | 2wikimqa.jsonl | 6441 | DAF | 0.00 | 0.00 | 0.00 | 0.520 | 1.180 | The Abduction Club | Wooden Crosses |
| 2wikimqa_046_len6386.json | 2wikimqa.jsonl | 6386 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.438 | 1.173 | Burghausen | Worms |
| 2wikimqa_046_len6386.json | 2wikimqa.jsonl | 6386 | DAF | 0.00 | 0.00 | 0.00 | 0.512 | 1.179 | He died in 1253. | Worms |
| 2wikimqa_052_len5626.json | 2wikimqa.jsonl | 5626 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.382 | 1.019 | In Castile. | Pamplona |
| 2wikimqa_052_len5626.json | 2wikimqa.jsonl | 5626 | DAF | 0.00 | 0.00 | 0.00 | 0.450 | 1.027 | In Saldaña. | Pamplona |
| 2wikimqa_068_len6050.json | 2wikimqa.jsonl | 6050 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.422 | 1.092 | The Great Man'S Lady | The Great Man's Lady |
| 2wikimqa_068_len6050.json | 2wikimqa.jsonl | 6050 | DAF | 1.00 | 1.00 | 1.00 | 0.490 | 1.100 | La Belle Américaine | The Great Man's Lady |
| 2wikimqa_069_len5077.json | 2wikimqa.jsonl | 5077 | CacheBlend | 1.00 | 1.00 | 1.00 | 0.357 | 0.911 | Oklahoma City, Oklahoma | Oklahoma City, Oklahoma |
| 2wikimqa_069_len5077.json | 2wikimqa.jsonl | 5077 | DAF | 1.00 | 1.00 | 1.00 | 0.418 | 0.918 | Oklahoma City, Oklahoma | Oklahoma City, Oklahoma |
| 2wikimqa_072_len6952.json | 2wikimqa.jsonl | 6952 | CacheBlend | 1.00 | 1.00 | 1.00 | 0.479 | 1.300 | Prince Albert of Prussia | Prince Albert of Prussia |
| 2wikimqa_072_len6952.json | 2wikimqa.jsonl | 6952 | DAF | 0.00 | 0.00 | 0.00 | 0.557 | 1.299 | Duke Christian Ludwig II | Prince Albert of Prussia |
| 2wikimqa_074_len5686.json | 2wikimqa.jsonl | 5686 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.393 | 1.025 | Beaulieu-sur-Loire | Beaulieu-sur-Loire |
| 2wikimqa_074_len5686.json | 2wikimqa.jsonl | 5686 | DAF | 0.00 | 0.00 | 0.00 | 0.454 | 1.031 | Beaulieu-sur-Loire | Beaulieu-sur-Loire |
| 2wikimqa_081_len6688.json | 2wikimqa.jsonl | 6688 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.469 | 1.220 | Closely Watched Trains | Closely Watched Trains |
| 2wikimqa_081_len6688.json | 2wikimqa.jsonl | 6688 | DAF | 0.00 | 0.00 | 0.00 | 0.546 | 1.230 | Closely Watched Trains | Closely Watched Trains |
| 2wikimqa_102_len6161.json | 2wikimqa.jsonl | 6161 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.424 | 1.132 | Denmark | Denmark |
| 2wikimqa_102_len6161.json | 2wikimqa.jsonl | 6161 | DAF | 0.00 | 0.00 | 0.00 | 0.498 | 1.142 | Denmark | Denmark |
| 2wikimqa_103_len6864.json | 2wikimqa.jsonl | 6864 | CacheBlend | 0.00 | 1.00 | 0.00 | 0.473 | 1.241 | Seonjong's illness. | Not specified in text. |
| 2wikimqa_103_len6864.json | 2wikimqa.jsonl | 6864 | DAF | 0.00 | 1.00 | 0.00 | 0.548 | 1.251 | He died of illness. | Not specified in text. |
| 2wikimqa_123_len6755.json | 2wikimqa.jsonl | 6755 | CacheBlend | 1.00 | 1.00 | 1.00 | 0.470 | 1.223 | The Comedians Of Comedy | The Comedians Of Comedy |
| 2wikimqa_123_len6755.json | 2wikimqa.jsonl | 6755 | DAF | 1.00 | 1.00 | 1.00 | 0.545 | 1.233 | The Comedians Of Comedy | The Comedians Of Comedy |
| 2wikimqa_129_len5744.json | 2wikimqa.jsonl | 5744 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.387 | 1.055 | Vytautas Straižys | Vytautas Straižys |
| 2wikimqa_129_len5744.json | 2wikimqa.jsonl | 5744 | DAF | 0.00 | 0.00 | 0.00 | 0.458 | 1.064 | Vytautas Straižys | Vytautas Straižys |
| 2wikimqa_135_len6729.json | 2wikimqa.jsonl | 6729 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.471 | 1.223 | No, they have different directors. | No, they are from different countries. |
| 2wikimqa_135_len6729.json | 2wikimqa.jsonl | 6729 | DAF | 1.00 | 1.00 | 1.00 | 0.545 | 1.233 | Yes, they are both French. | No, they are from different countries. |
| 2wikimqa_139_len5487.json | 2wikimqa.jsonl | 5487 | CacheBlend | 1.00 | 1.00 | 1.00 | 0.380 | 1.004 | Wolf Warrior | Wolf Warrior |
| 2wikimqa_139_len5487.json | 2wikimqa.jsonl | 5487 | DAF | 1.00 | 1.00 | 1.00 | 0.450 | 1.013 | Wolf Warrior | Wolf Warrior |
| 2wikimqa_140_len6288.json | 2wikimqa.jsonl | 6288 | CacheBlend | 0.00 | 0.00 | 0.44 | 0.454 | 1.164 | Trịnh Cương is. | Trịnh Cương |
| 2wikimqa_140_len6288.json | 2wikimqa.jsonl | 6288 | DAF | 0.00 | 0.00 | 0.50 | 0.513 | 1.174 | Trịnh Cương | Trịnh Cương |
| 2wikimqa_146_len5300.json | 2wikimqa.jsonl | 5300 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.362 | 0.948 | The passage does not provide the directo | Werner Abrolat, Germany |
| 2wikimqa_146_len5300.json | 2wikimqa.jsonl | 5300 | DAF | 0.00 | 0.00 | 0.00 | 0.428 | 0.956 | West Germany | Werner Abrolat, Germany |
| 2wikimqa_152_len6599.json | 2wikimqa.jsonl | 6599 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.448 | 1.213 | Peter Greenaway, UK | Newport, Monmouthshire, Wales |
| 2wikimqa_152_len6599.json | 2wikimqa.jsonl | 6599 | DAF | 0.00 | 0.00 | 0.00 | 0.525 | 1.219 | Ireland | Newport, Monmouthshire, Wales |
| 2wikimqa_158_len5914.json | 2wikimqa.jsonl | 5914 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.421 | 1.082 | Queen Ding of Zhou | Queen Yi Jiang |
| 2wikimqa_158_len5914.json | 2wikimqa.jsonl | 5914 | DAF | 0.00 | 0.00 | 0.00 | 0.487 | 1.087 | Wang Si, Queen of Zhou | Queen Yi Jiang |
| 2wikimqa_182_len5003.json | 2wikimqa.jsonl | 5003 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.358 | 0.919 | London, England. | Toronto, Ontario |
| 2wikimqa_182_len5003.json | 2wikimqa.jsonl | 5003 | DAF | 0.00 | 0.00 | 0.00 | 0.418 | 0.923 | Paris, France | Toronto, Ontario |
| 2wikimqa_183_len6222.json | 2wikimqa.jsonl | 6222 | CacheBlend | 1.00 | 1.00 | 1.00 | 0.428 | 1.157 | Forbidden Daughters | The Third Kiss |
| 2wikimqa_183_len6222.json | 2wikimqa.jsonl | 6222 | DAF | 0.00 | 0.00 | 0.00 | 0.504 | 1.166 | The Third Kiss | The Third Kiss |
| 2wikimqa_198_len5112.json | 2wikimqa.jsonl | 5112 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.360 | 0.934 | Fredrikstad | Milan, Italy |
| 2wikimqa_198_len5112.json | 2wikimqa.jsonl | 5112 | DAF | 0.00 | 0.00 | 0.00 | 0.423 | 0.940 | Rosario, Santa Fe Province | Milan, Italy |

## 二、按方法聚合（每轮一行，再给出跨轮中位数）

| 方法 | 轮次 | EM | SM | F1 | TTFT缓存均值(s) | TTFT全量均值(s) | 缓存加速比 |
|---|---|---|---|---|---|---|---|
| CacheBlend | r1 | 0.208 | 0.292 | 0.281 | 0.417 | 1.095 | 2.63× |
| **CacheBlend** | **中位** | **0.208** | **0.292** | **0.281** | **0.417** | **1.095** | **2.63×** |
| DAF | r1 | 0.208 | 0.333 | 0.317 | 0.488 | 1.102 | 2.26× |
| **DAF** | **中位** | **0.208** | **0.333** | **0.317** | **0.488** | **1.102** | **2.26×** |
| vLLM全量 | r1 | 0.250 | 0.417 | 0.414 | - | 1.095 | 1.00× |
| vLLM全量 | r2 | 0.250 | 0.417 | 0.414 | - | 1.102 | 1.00× |
| **vLLM全量** | **中位** | **0.250** | **0.417** | **0.414** | - | **1.098** | **1.00×** |

## 三、按数据集×方法聚合（跨轮中位数；QA 数据集看 SM/F1，samsum 摘要看 F1）

| 数据集 | 样本数 | 方法 | EM中位 | SM中位 | F1中位 | TTFT缓存中位(s) |
|---|---|---|---|---|---|---|
| 2wikimqa.jsonl | 24 | CacheBlend | 0.000 | 0.000 | 0.000 | 0.423 |
| 2wikimqa.jsonl | 24 | DAF | 0.000 | 0.000 | 0.000 | 0.494 |
