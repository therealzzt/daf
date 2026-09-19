# 6k–10k 长上下文实验评分报告

- 样本目录：`experiments/inputs_qwen_2wikimqa`（24 个样本）
- 评分：EM（精确匹配，多答案取最大）+ token-F1（多答案取最大）
- 归一化：小写/去冠词/去标点/Yes-No 规整（`example/utils.py`）
- 输入日志：`run_cb_qwen2wiki_r1.log`, `run_cb_qwen2wiki_r2.log`, `run_cb_qwen2wiki_r3.log`, `run_daf_qwen2wiki_r1.log`, `run_daf_qwen2wiki_r2.log`, `run_daf_qwen2wiki_r3.log`

## 一、逐样本明细（质量指标为多轮均值，TTFT 为多轮中位数）

| 样本 | 来源 | 长度 | 方法 | EM | SM | F1 | TTFT缓存(s) | TTFT全量(s) | 缓存答案 | 全量答案 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2wikimqa_008_len6111.json | 2wikimqa.jsonl | 6111 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.442 | 1.101 | Not specified | Abbey Theatre |
| 2wikimqa_008_len6111.json | 2wikimqa.jsonl | 6111 | DAF | 0.00 | 1.00 | 0.80 | 0.523 | 1.099 | St Patrick's College, Dublin. | Abbey Theatre |
| 2wikimqa_013_len5407.json | 2wikimqa.jsonl | 5407 | CacheBlend | 0.00 | 0.00 | 0.50 | 0.362 | 0.990 | 1806 | 1872 |
| 2wikimqa_013_len5407.json | 2wikimqa.jsonl | 5407 | DAF | 0.00 | 0.00 | 0.50 | 0.420 | 0.988 | 1846 | 1872 |
| 2wikimqa_019_len5555.json | 2wikimqa.jsonl | 5555 | CacheBlend | 0.00 | 1.00 | 0.80 | 0.379 | 1.011 | Mysore, India | Mysore, India |
| 2wikimqa_019_len5555.json | 2wikimqa.jsonl | 5555 | DAF | 0.00 | 1.00 | 0.80 | 0.436 | 1.009 | Mysore, India | Mysore, India |
| 2wikimqa_034_len6441.json | 2wikimqa.jsonl | 6441 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.440 | 1.183 | Ringo-en no shōjo | Wooden Crosses |
| 2wikimqa_034_len6441.json | 2wikimqa.jsonl | 6441 | DAF | 0.00 | 0.00 | 0.00 | 0.503 | 1.181 | The Abduction Club | Wooden Crosses |
| 2wikimqa_046_len6386.json | 2wikimqa.jsonl | 6386 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.439 | 1.185 | Burghausen | Worms |
| 2wikimqa_046_len6386.json | 2wikimqa.jsonl | 6386 | DAF | 0.00 | 0.00 | 0.00 | 0.499 | 1.181 | He died in 1253. | Worms |
| 2wikimqa_052_len5626.json | 2wikimqa.jsonl | 5626 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.382 | 1.031 | In Castile. | Pamplona |
| 2wikimqa_052_len5626.json | 2wikimqa.jsonl | 5626 | DAF | 0.00 | 0.00 | 0.00 | 0.439 | 1.026 | In Saldaña. | Pamplona |
| 2wikimqa_068_len6050.json | 2wikimqa.jsonl | 6050 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.422 | 1.103 | The Great Man'S Lady | The Great Man's Lady |
| 2wikimqa_068_len6050.json | 2wikimqa.jsonl | 6050 | DAF | 1.00 | 1.00 | 1.00 | 0.477 | 1.097 | La Belle Américaine | The Great Man's Lady |
| 2wikimqa_069_len5077.json | 2wikimqa.jsonl | 5077 | CacheBlend | 1.00 | 1.00 | 1.00 | 0.356 | 0.920 | Oklahoma City, Oklahoma | Oklahoma City, Oklahoma |
| 2wikimqa_069_len5077.json | 2wikimqa.jsonl | 5077 | DAF | 1.00 | 1.00 | 1.00 | 0.405 | 0.915 | Oklahoma City, Oklahoma | Oklahoma City, Oklahoma |
| 2wikimqa_072_len6952.json | 2wikimqa.jsonl | 6952 | CacheBlend | 1.00 | 1.00 | 1.00 | 0.478 | 1.306 | Prince Albert of Prussia | Prince Albert of Prussia |
| 2wikimqa_072_len6952.json | 2wikimqa.jsonl | 6952 | DAF | 0.00 | 0.00 | 0.71 | 0.543 | 1.296 | Prince Wilhelm of Prussia | Prince Albert of Prussia |
| 2wikimqa_074_len5686.json | 2wikimqa.jsonl | 5686 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.384 | 1.036 | Beaulieu-sur-Loire | Beaulieu-sur-Loire |
| 2wikimqa_074_len5686.json | 2wikimqa.jsonl | 5686 | DAF | 0.00 | 0.00 | 0.00 | 0.440 | 1.028 | Beaulieu-sur-Loire | Beaulieu-sur-Loire |
| 2wikimqa_081_len6688.json | 2wikimqa.jsonl | 6688 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.470 | 1.234 | Closely Watched Trains | Closely Watched Trains |
| 2wikimqa_081_len6688.json | 2wikimqa.jsonl | 6688 | DAF | 0.00 | 0.00 | 0.00 | 0.531 | 1.225 | Closely Watched Trains | Closely Watched Trains |
| 2wikimqa_102_len6161.json | 2wikimqa.jsonl | 6161 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.425 | 1.146 | Denmark | Denmark |
| 2wikimqa_102_len6161.json | 2wikimqa.jsonl | 6161 | DAF | 0.00 | 0.00 | 0.00 | 0.482 | 1.137 | Denmark | Denmark |
| 2wikimqa_103_len6864.json | 2wikimqa.jsonl | 6864 | CacheBlend | 0.00 | 1.00 | 0.00 | 0.473 | 1.255 | Seonjong's illness. | Not specified in text. |
| 2wikimqa_103_len6864.json | 2wikimqa.jsonl | 6864 | DAF | 0.00 | 1.00 | 0.00 | 0.534 | 1.244 | He died of illness. | Not specified in text. |
| 2wikimqa_123_len6755.json | 2wikimqa.jsonl | 6755 | CacheBlend | 1.00 | 1.00 | 1.00 | 0.471 | 1.237 | The Comedians Of Comedy | The Comedians Of Comedy |
| 2wikimqa_123_len6755.json | 2wikimqa.jsonl | 6755 | DAF | 1.00 | 1.00 | 1.00 | 0.529 | 1.228 | The Comedians Of Comedy | The Comedians Of Comedy |
| 2wikimqa_129_len5744.json | 2wikimqa.jsonl | 5744 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.386 | 1.067 | Vytautas Straižys | Vytautas Straižys |
| 2wikimqa_129_len5744.json | 2wikimqa.jsonl | 5744 | DAF | 0.00 | 0.00 | 0.00 | 0.444 | 1.059 | Vytautas Straižys | Vytautas Straižys |
| 2wikimqa_135_len6729.json | 2wikimqa.jsonl | 6729 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.471 | 1.237 | No, they have different directors. | No, they are from different countries. |
| 2wikimqa_135_len6729.json | 2wikimqa.jsonl | 6729 | DAF | 1.00 | 1.00 | 1.00 | 0.533 | 1.228 | Yes, they are both French. | No, they are from different countries. |
| 2wikimqa_139_len5487.json | 2wikimqa.jsonl | 5487 | CacheBlend | 1.00 | 1.00 | 1.00 | 0.380 | 1.015 | Wolf Warrior | Wolf Warrior |
| 2wikimqa_139_len5487.json | 2wikimqa.jsonl | 5487 | DAF | 1.00 | 1.00 | 1.00 | 0.435 | 1.008 | Wolf Warrior | Wolf Warrior |
| 2wikimqa_140_len6288.json | 2wikimqa.jsonl | 6288 | CacheBlend | 0.00 | 0.00 | 0.44 | 0.440 | 1.178 | Trịnh Cương is. | Trịnh Cương |
| 2wikimqa_140_len6288.json | 2wikimqa.jsonl | 6288 | DAF | 0.00 | 0.00 | 0.50 | 0.499 | 1.169 | Trịnh Cương | Trịnh Cương |
| 2wikimqa_146_len5300.json | 2wikimqa.jsonl | 5300 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.360 | 0.953 | The passage does not provide the directo | Werner Abrolat, Germany |
| 2wikimqa_146_len5300.json | 2wikimqa.jsonl | 5300 | DAF | 0.00 | 0.00 | 0.00 | 0.415 | 0.955 | West Germany | Werner Abrolat, Germany |
| 2wikimqa_152_len6599.json | 2wikimqa.jsonl | 6599 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.448 | 1.218 | Peter Greenaway, UK | Newport, Monmouthshire, Wales |
| 2wikimqa_152_len6599.json | 2wikimqa.jsonl | 6599 | DAF | 0.00 | 0.00 | 0.00 | 0.511 | 1.209 | Ireland | Newport, Monmouthshire, Wales |
| 2wikimqa_158_len5914.json | 2wikimqa.jsonl | 5914 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.419 | 1.087 | Queen Ding of Zhou | Queen Yi Jiang |
| 2wikimqa_158_len5914.json | 2wikimqa.jsonl | 5914 | DAF | 0.00 | 0.00 | 0.00 | 0.473 | 1.079 | Wang Si, Queen of Zhou | Queen Yi Jiang |
| 2wikimqa_182_len5003.json | 2wikimqa.jsonl | 5003 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.356 | 0.923 | London, England. | Toronto, Ontario |
| 2wikimqa_182_len5003.json | 2wikimqa.jsonl | 5003 | DAF | 0.00 | 0.00 | 0.00 | 0.404 | 0.916 | Paris, France | Toronto, Ontario |
| 2wikimqa_183_len6222.json | 2wikimqa.jsonl | 6222 | CacheBlend | 1.00 | 1.00 | 1.00 | 0.426 | 1.163 | Forbidden Daughters | The Third Kiss |
| 2wikimqa_183_len6222.json | 2wikimqa.jsonl | 6222 | DAF | 0.00 | 0.00 | 0.00 | 0.486 | 1.154 | The Third Kiss | The Third Kiss |
| 2wikimqa_198_len5112.json | 2wikimqa.jsonl | 5112 | CacheBlend | 0.00 | 0.00 | 0.00 | 0.357 | 0.937 | Fredrikstad | Milan, Italy |
| 2wikimqa_198_len5112.json | 2wikimqa.jsonl | 5112 | DAF | 0.00 | 0.00 | 0.00 | 0.414 | 0.938 | Rosario, Santa Fe Province | Milan, Italy |

## 二、按方法聚合（每轮一行，再给出跨轮中位数）

| 方法 | 轮次 | EM | SM | F1 | TTFT缓存均值(s) | TTFT全量均值(s) | 缓存加速比 |
|---|---|---|---|---|---|---|---|
| CacheBlend | r1 | 0.208 | 0.292 | 0.281 | 0.415 | 1.104 | 2.66× |
| CacheBlend | r2 | 0.208 | 0.292 | 0.281 | 0.416 | 1.105 | 2.66× |
| CacheBlend | r3 | 0.208 | 0.292 | 0.281 | 0.415 | 1.105 | 2.66× |
| **CacheBlend** | **中位** | **0.208** | **0.292** | **0.281** | **0.415** | **1.105** | **2.66×** |
| DAF | r1 | 0.208 | 0.333 | 0.346 | 0.472 | 1.097 | 2.32× |
| DAF | r2 | 0.208 | 0.333 | 0.346 | 0.475 | 1.100 | 2.31× |
| DAF | r3 | 0.208 | 0.333 | 0.346 | 0.473 | 1.099 | 2.32× |
| **DAF** | **中位** | **0.208** | **0.333** | **0.346** | **0.473** | **1.099** | **2.32×** |
| vLLM全量 | r1 | 0.250 | 0.417 | 0.414 | - | 1.104 | 1.00× |
| vLLM全量 | r2 | 0.250 | 0.417 | 0.414 | - | 1.105 | 1.00× |
| vLLM全量 | r3 | 0.250 | 0.417 | 0.414 | - | 1.105 | 1.00× |
| vLLM全量 | r4 | 0.250 | 0.417 | 0.414 | - | 1.097 | 1.00× |
| vLLM全量 | r5 | 0.250 | 0.417 | 0.414 | - | 1.100 | 1.00× |
| vLLM全量 | r6 | 0.250 | 0.417 | 0.414 | - | 1.099 | 1.00× |
| **vLLM全量** | **中位** | **0.250** | **0.417** | **0.414** | - | **1.102** | **1.00×** |

## 三、按数据集×方法聚合（跨轮中位数；QA 数据集看 SM/F1，samsum 摘要看 F1）

| 数据集 | 样本数 | 方法 | EM中位 | SM中位 | F1中位 | TTFT缓存中位(s) |
|---|---|---|---|---|---|---|
| 2wikimqa.jsonl | 24 | CacheBlend | 0.000 | 0.000 | 0.000 | 0.423 |
| 2wikimqa.jsonl | 24 | DAF | 0.000 | 0.000 | 0.000 | 0.480 |

## 四、与论文 6k 档对照（分析）

**实验配置**：Qwen/Qwen2.5-7B-Instruct（BF16）/ 3090Ti / xformers 后端（FLASH 对照见
`em_report_qwen_2wikimqa_flash.md`）；LongBench 官方 2wikimqa 24 样本（5.0k~7.0k token，
均值 6007，中位 6111）；CB 与 DAF 均 r=0.3；3 轮贪心解码。

| 指标 | CacheBlend | DAF | vLLM 全量 | 论文 6k 档预期 |
|---|---|---|---|---|
| EM（跨轮中位） | 0.208 | 0.208 | 0.250 | DAF 贴近全量、CB 略掉 |
| SM（跨轮中位） | 0.292 | **0.333** | 0.417 | 同上 |
| F1（跨轮中位） | 0.281 | **0.346** | 0.414 | 同上 |
| TTFT（跨轮中位） | 0.415s | 0.473s | 1.099~1.105s | 6k 档 DAF≈CB 持平 |
| 相对全量加速 | 2.66× | 2.32× | 1.00× | DAF ≈1.8× |

**发现**：
1. **TTFT 持平复现成功**：DAF/CB 时长比 = 0.473/0.415 ≈ **1.14×**，接近论文"6k 档
   DAF 与 CB 基本持平"的结论（此前 Mistral 2–4k 档为 1.75×，Qwen 6k 档明显收窄）。
2. **加速比超过论文**：DAF 相对全量 2.32×（论文 6k 档 ≈1.8×）。可能原因：3090Ti 上
   全量 prefill 相对更贵、xformers 实现差异；属环境差异，方向一致。
3. **精度趋势与论文一致但未完全复现**：SM/F1 上 DAF（0.333/0.346）位于 CB
   （0.292/0.281）与全量（0.417/0.414）之间，符合"DAF 贴近全量、CB 略掉"的排序；
   但 EM 上 DAF 与 CB 打平（0.208），且全量仍明显最优——"DAF 追平全量"未复现。
   注意 Qwen 输出常带补充（如 "St Patrick's College, Dublin." vs gold
   "St Patrick's College"），EM 偏严，SM/F1 更能反映语义正确率。
4. **两后端一致性**：FLASH 对照（各 1 轮）CB/DAF 的 EM/SM 与 xformers 完全相同，
   TTFT 差异 <5%（kernel 噪声内），flash 移植在 6k 规模下同样成立。

**工程实现假设**（论文未写死，非论文原方法）：多层偏差分数取平均聚合、选择层固定在
第 4 层（sel_layers=[1,2,3] 之后）、状态融合为 [重要文档|问题] 顺序拼接。
环境差异：论文为 A100 + Flash-Attention，本实验为 3090Ti + xformers/flash_attn 2.5.8。
