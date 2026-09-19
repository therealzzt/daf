# 长上下文实验评分报告

- 样本目录：`experiments/inputs_qwen_samsum`（24 个样本）
- 评分：EM（精确匹配，多答案取最大）+ SM（子串匹配）+ token-F1（多答案取最大） + Rouge-L（摘要任务主指标，LongBench 官方口径）
- 归一化：小写/去冠词/去标点/Yes-No 规整（`example/utils.py`）
- 输入日志：`run_cb_qwen_samsum_r1.log`, `run_cb_qwen_samsum_r2.log`, `run_cb_qwen_samsum_r3.log`, `run_daf_qwen_samsum_r1.log`, `run_daf_qwen_samsum_r2.log`, `run_daf_qwen_samsum_r3.log`

## 一、逐样本明细（质量指标为多轮均值，TTFT 为多轮中位数）

| 样本 | 来源 | 长度 | 方法 | EM | SM | F1 | RL | TTFT缓存(s) | TTFT全量(s) | 缓存答案 | 全量答案 |
|---|---|---|---|---|---|---|---|---|----|---|---|
| samsum_000_len9508.json | samsum.jsonl | 9508 | CacheBlend | 0.00 | 1.00 | 0.77 | 0.80 | 0.692 | 1.808 | Emilia is still angry at William.  | Emilia is still angry with William. |
| samsum_000_len9508.json | samsum.jsonl | 9508 | DAF | 0.00 | 1.00 | 0.77 | 0.80 | 0.790 | 1.804 | Emilia is still angry at William.  | Emilia is still angry with William. |
| samsum_003_len8351.json | samsum.jsonl | 8351 | CacheBlend | 0.00 | 0.00 | 0.24 | 0.26 | 0.610 | 1.551 | Israel hasn't sewn anything yet, but Aut | Autumn has sewn a dress and shows Israel |
| samsum_003_len8351.json | samsum.jsonl | 8351 | DAF | 0.00 | 0.00 | 0.21 | 0.22 | 0.682 | 1.548 | Israel hasn't sewn anything yet, while A | Autumn has sewn a dress and shows Israel |
| samsum_004_len9891.json | samsum.jsonl | 9891 | CacheBlend | 0.00 | 0.00 | 0.47 | 0.43 | 0.771 | 1.865 | Brenda and Sandra, who used to work toge | Brenda and Sandra, who used to work toge |
| samsum_004_len9891.json | samsum.jsonl | 9891 | DAF | 0.00 | 0.00 | 0.49 | 0.40 | 0.843 | 1.865 | Brenda and Sandra, who used to work toge | Brenda and Sandra, who used to work toge |
| samsum_010_len9875.json | samsum.jsonl | 9875 | CacheBlend | 0.00 | 0.00 | 0.44 | 0.41 | 0.703 | 1.866 | Celine is not at home, but Cara still wa | Celine is not at home and can drop by in |
| samsum_010_len9875.json | samsum.jsonl | 9875 | DAF | 0.00 | 0.00 | 0.32 | 0.30 | 0.770 | 1.860 | Cara wanted to drop by Celine's place, b | Celine is not at home and can drop by in |
| samsum_011_len9677.json | samsum.jsonl | 9677 | CacheBlend | 0.00 | 0.00 | 0.66 | 0.38 | 0.700 | 1.837 | Janice has watched almost all of the top | Janice has watched almost all of the top |
| samsum_011_len9677.json | samsum.jsonl | 9677 | DAF | 0.00 | 0.00 | 0.55 | 0.31 | 0.768 | 1.821 | Janice has watched almost all of the top | Janice has watched almost all of the top |
| samsum_015_len9811.json | samsum.jsonl | 9811 | CacheBlend | 0.00 | 0.00 | 0.22 | 0.21 | 0.701 | 1.847 | Max shared his latest music project with | Max shared his latest music project with |
| samsum_015_len9811.json | samsum.jsonl | 9811 | DAF | 0.00 | 0.00 | 0.29 | 0.24 | 0.769 | 1.844 | Max shared a link to his latest music pr | Max shared his latest music project with |
| samsum_028_len9462.json | samsum.jsonl | 9462 | CacheBlend | 0.00 | 0.00 | 0.58 | 0.55 | 0.691 | 1.818 | Iris is trying to remove herself from an | Iris is trying to remove herself from an |
| samsum_028_len9462.json | samsum.jsonl | 9462 | DAF | 0.00 | 0.00 | 0.46 | 0.51 | 0.769 | 1.802 | Iris is trying to remove herself from an | Iris is trying to remove herself from an |
| samsum_063_len8439.json | samsum.jsonl | 8439 | CacheBlend | 0.00 | 1.00 | 0.78 | 0.75 | 0.614 | 1.618 | Railey will buy Tiffany a burger on his  | Tiffany asked Railey to buy her a burger |
| samsum_063_len8439.json | samsum.jsonl | 8439 | DAF | 0.00 | 0.00 | 0.35 | 0.38 | 0.679 | 1.603 | Tiffany asked Railey to buy her a burger | Tiffany asked Railey to buy her a burger |
| samsum_068_len9121.json | samsum.jsonl | 9121 | CacheBlend | 0.00 | 0.00 | 0.38 | 0.35 | 0.662 | 1.713 | Olivia and Jake are discussing their pla | Olivia has some tasks planned for the da |
| samsum_068_len9121.json | samsum.jsonl | 9121 | DAF | 0.00 | 0.00 | 0.42 | 0.36 | 0.733 | 1.701 | Olivia and Jake are discussing their pla | Olivia has some tasks planned for the da |
| samsum_071_len8973.json | samsum.jsonl | 8973 | CacheBlend | 0.00 | 0.00 | 0.63 | 0.57 | 0.640 | 1.695 | Paul will be home later than expected an | Paul will be home later than expected an |
| samsum_071_len8973.json | samsum.jsonl | 8973 | DAF | 0.00 | 0.00 | 0.55 | 0.51 | 0.710 | 1.682 | Paul will be home later than expected an | Paul will be home later than expected an |
| samsum_073_len8489.json | samsum.jsonl | 8489 | CacheBlend | 0.00 | 0.00 | 0.27 | 0.26 | 0.615 | 1.617 | Iris's husband got an interview, making  | Iris's husband got an interview, which m |
| samsum_073_len8489.json | samsum.jsonl | 8489 | DAF | 0.00 | 0.00 | 0.22 | 0.25 | 0.682 | 1.605 | Iris's husband has an interview, which m | Iris's husband got an interview, which m |
| samsum_074_len9713.json | samsum.jsonl | 9713 | CacheBlend | 0.00 | 0.00 | 0.64 | 0.63 | 0.698 | 1.853 | Derek wants Judy to feed his animals on  | Derek asks Judy to feed his animals on F |
| samsum_074_len9713.json | samsum.jsonl | 9713 | DAF | 0.00 | 0.00 | 0.67 | 0.62 | 0.770 | 1.839 | Derek wants Judy to feed his animals on  | Derek asks Judy to feed his animals on F |
| samsum_078_len8183.json | samsum.jsonl | 8183 | CacheBlend | 0.00 | 0.00 | 0.71 | 0.63 | 0.573 | 1.532 | William is making spaghetti and needs Ol | William is making spaghetti and asks Oli |
| samsum_078_len8183.json | samsum.jsonl | 8183 | DAF | 0.00 | 0.00 | 0.64 | 0.43 | 0.643 | 1.522 | William is making spaghetti and needs fr | William is making spaghetti and asks Oli |
| samsum_091_len9297.json | samsum.jsonl | 9297 | CacheBlend | 0.00 | 1.00 | 0.44 | 0.36 | 0.666 | 1.774 | Aimee is looking for Maryam and has trie | Aimee is looking for Maryam but couldn't |
| samsum_091_len9297.json | samsum.jsonl | 9297 | DAF | 0.00 | 1.00 | 0.44 | 0.36 | 0.741 | 1.762 | Aimee is looking for Maryam and has trie | Aimee is looking for Maryam but couldn't |
| samsum_093_len9159.json | samsum.jsonl | 9159 | CacheBlend | 0.00 | 0.00 | 0.56 | 0.52 | 0.645 | 1.714 | Patricia needs to talk to George in an h | Patricia wants to talk to George in an h |
| samsum_093_len9159.json | samsum.jsonl | 9159 | DAF | 0.00 | 0.00 | 0.56 | 0.52 | 0.714 | 1.704 | Patricia needs to talk to George in an h | Patricia wants to talk to George in an h |
| samsum_107_len8135.json | samsum.jsonl | 8135 | CacheBlend | 0.00 | 0.00 | 0.42 | 0.42 | 0.571 | 1.514 | Ben and Tom are meeting at the Oval Room | Tom and Ben are meeting at 2pm in the Ov |
| samsum_107_len8135.json | samsum.jsonl | 8135 | DAF | 0.00 | 0.00 | 0.44 | 0.30 | 0.637 | 1.510 | Tom and Ben are going to a meeting at 2p | Tom and Ben are meeting at 2pm in the Ov |
| samsum_112_len8429.json | samsum.jsonl | 8429 | CacheBlend | 0.00 | 0.00 | 0.28 | 0.35 | 0.609 | 1.564 | Marta thinks Jay is lying about having a | Marta thinks Jay is lying about having a |
| samsum_112_len8429.json | samsum.jsonl | 8429 | DAF | 0.00 | 0.00 | 0.22 | 0.26 | 0.675 | 1.553 | Marta thinks Jay is a pathological liar  | Marta thinks Jay is lying about having a |
| samsum_131_len9346.json | samsum.jsonl | 9346 | CacheBlend | 0.00 | 0.00 | 0.48 | 0.27 | 0.692 | 1.787 | Inez, Gosia, Alicja, and Patrycja are pl | Inez proposes a food evening and shares  |
| samsum_131_len9346.json | samsum.jsonl | 9346 | DAF | 0.00 | 0.00 | 0.54 | 0.24 | 0.769 | 1.780 | Inez, Gosia, Alicja, and Patrycja are pl | Inez proposes a food evening and shares  |
| samsum_134_len9592.json | samsum.jsonl | 9592 | CacheBlend | 0.00 | 0.00 | 0.21 | 0.13 | 0.696 | 1.829 | Kaylin and Amir are making fun of a pers | Kaylin and Amir are laughing at funny im |
| samsum_134_len9592.json | samsum.jsonl | 9592 | DAF | 0.00 | 0.00 | 0.23 | 0.21 | 0.777 | 1.821 | Kaylin and Amir are making fun of a pers | Kaylin and Amir are laughing at funny im |
| samsum_146_len9769.json | samsum.jsonl | 9769 | CacheBlend | 0.00 | 0.00 | 0.82 | 0.80 | 0.698 | 1.850 | Lilly will be late and Gabriel will orde | Lilly will be late and Gabriel will orde |
| samsum_146_len9769.json | samsum.jsonl | 9769 | DAF | 0.00 | 0.00 | 0.78 | 0.65 | 0.770 | 1.841 | Lilly will be late and Gabriel will orde | Lilly will be late and Gabriel will orde |
| samsum_153_len9318.json | samsum.jsonl | 9318 | CacheBlend | 0.00 | 0.00 | 0.31 | 0.29 | 0.688 | 1.774 | Simon is considering having Freddy sing  | Simon asked Freddy if he would be willin |
| samsum_153_len9318.json | samsum.jsonl | 9318 | DAF | 0.00 | 0.00 | 0.29 | 0.27 | 0.765 | 1.764 | Simon heard that Freddy has a good singi | Simon asked Freddy if he would be willin |
| samsum_187_len8519.json | samsum.jsonl | 8519 | CacheBlend | 0.00 | 0.00 | 0.43 | 0.41 | 0.630 | 1.621 | Chloe wants to go to Megan's house after | Lesley asks Chloe to walk the dog when s |
| samsum_187_len8519.json | samsum.jsonl | 8519 | DAF | 0.00 | 0.00 | 0.43 | 0.39 | 0.704 | 1.607 | Chloe wants to go to Megan's house after | Lesley asks Chloe to walk the dog when s |
| samsum_193_len8683.json | samsum.jsonl | 8683 | CacheBlend | 0.00 | 0.00 | 0.48 | 0.42 | 0.635 | 1.633 | Max's sister is studying in Shanghai, Ch | Max's sister is studying in Shanghai, Ch |
| samsum_193_len8683.json | samsum.jsonl | 8683 | DAF | 0.00 | 0.00 | 0.49 | 0.42 | 0.708 | 1.625 | Max's sister is studying in Shanghai, Ch | Max's sister is studying in Shanghai, Ch |
| samsum_198_len8272.json | samsum.jsonl | 8272 | CacheBlend | 0.00 | 0.00 | 0.60 | 0.36 | 0.609 | 1.536 | Lucy turned down Johnny's invitation to  | Lucy turned down Johnny's offer to go to |
| samsum_198_len8272.json | samsum.jsonl | 8272 | DAF | 0.00 | 0.00 | 0.65 | 0.42 | 0.679 | 1.526 | Lucy turned down Johnny's invitation to  | Lucy turned down Johnny's offer to go to |

## 二、按方法聚合（每轮一行，再给出跨轮中位数）

| 方法 | 轮次 | EM | SM | F1 | RL | TTFT缓存均值(s) | TTFT全量均值(s) | 缓存加速比 |
|---|---|---|---|---|---|---|------|---|
| CacheBlend | r1 | 0.000 | 0.125 | 0.492 | 0.440 | 0.661 | 1.722 | 2.60× |
| CacheBlend | r2 | 0.000 | 0.125 | 0.492 | 0.440 | 0.659 | 1.717 | 2.61× |
| CacheBlend | r3 | 0.000 | 0.125 | 0.492 | 0.440 | 0.657 | 1.711 | 2.60× |
| **CacheBlend** | **中位** | **0.000** | **0.125** | **0.492** | **0.440** | **0.659** | **1.717** | **2.61×** |
| DAF | r1 | 0.000 | 0.083 | 0.458 | 0.391 | 0.731 | 1.706 | 2.33× |
| DAF | r2 | 0.000 | 0.083 | 0.458 | 0.391 | 0.733 | 1.709 | 2.33× |
| DAF | r3 | 0.000 | 0.083 | 0.458 | 0.391 | 0.731 | 1.708 | 2.34× |
| **DAF** | **中位** | **0.000** | **0.083** | **0.458** | **0.391** | **0.731** | **1.708** | **2.34×** |
| vLLM全量 | r1 | 0.000 | 0.125 | 0.484 | 0.428 | - | 1.722 | 1.00× |
| vLLM全量 | r2 | 0.000 | 0.125 | 0.484 | 0.428 | - | 1.717 | 1.00× |
| vLLM全量 | r3 | 0.000 | 0.125 | 0.484 | 0.428 | - | 1.711 | 1.00× |
| vLLM全量 | r4 | 0.000 | 0.125 | 0.484 | 0.428 | - | 1.706 | 1.00× |
| vLLM全量 | r5 | 0.000 | 0.125 | 0.484 | 0.428 | - | 1.709 | 1.00× |
| vLLM全量 | r6 | 0.000 | 0.125 | 0.484 | 0.428 | - | 1.708 | 1.00× |
| **vLLM全量** | **中位** | **0.000** | **0.125** | **0.484** | **0.428** | - | **1.710** | **1.00×** |

## 三、按数据集×方法聚合（跨轮中位数；QA 数据集看 SM/F1，samsum 摘要看 F1）

| 数据集 | 样本数 | 方法 | EM中位 | SM中位 | F1中位 | TTFT缓存中位(s) |
|---|---|---|---|---|---|---|
| samsum.jsonl | 24 | CacheBlend | 0.000 | 0.000 | 0.475 | 0.664 |
| samsum.jsonl | 24 | DAF | 0.000 | 0.000 | 0.450 | 0.737 |

## 三、与论文 9k 档对照（分析）

**实验配置**：Qwen/Qwen2.5-7B-Instruct（BF16）/ 3090Ti / xformers 后端（FLASH 对照见
`rouge_report_qwen_samsum_flash.md`）；LongBench 官方 samsum 24 样本（8.1k~9.9k token，
均值 9084，中位 9297）；prompt 版式对齐官方 `dataset2prompt.json` 的 samsum 模板
（指令行 + ~53 个 Dialogue+Summary few-shot 示例块 + 待摘要对话）；生成上限 64 token
（LongBench 官方 max_gen）；CB 与 DAF 均 r=0.3；3 轮贪心解码。
主指标 Rouge-L（rouge_score rougeL fmeasure，先取首行——LongBench 官方 eval.py 口径）。

| 指标（跨轮中位） | CacheBlend | DAF | vLLM 全量 | 论文 9k 档预期 |
|---|---|---|---|---|
| Rouge-L | **0.440** | 0.391 | 0.428 | DAF 贴近全量、CB 略掉 |
| F1（参考） | 0.492 | 0.458 | 0.484 | — |
| EM/SM（参考） | 0.000/0.125 | 0.000/0.083 | 0.000/0.125 | 摘要任务预期≈0，非 bug |
| TTFT | 0.659s | 0.731s | 1.708~1.717s | 9k 档 DAF≈CB 持平 |
| 相对全量加速 | 2.61× | 2.34× | 1.00× | DAF ≈2× |

**发现**：
1. **TTFT 持平复现成功**：DAF/CB 时长比 = 0.731/0.659 ≈ **1.11×**，比 2wikimqa 6k 档
   的 1.14× 更接近持平，符合论文"9k 档 DAF 与 CB TTFT 基本持平"。
2. **加速比与论文一致**：DAF 相对全量 2.34×（论文 9k 档 ≈2×）。
3. **精度排序与论文相反（未复现）**：Rouge-L 上 CB（0.440）≈ 全量（0.428）> DAF
   （0.391），论文预期 DAF 贴近全量、CB 略掉。差距 0.04~0.05，24 样本下单样本
   ±1 题 ≈ ±0.02，部分属统计噪声；但方向上 DAF 确实最低。可能原因（工程假设，
   需更多样本验证）：samsum 的 few-shot 结构（~53 个独立示例块）与 DAF 的
   "重要 token 互看"假设相性较差——跨块注意力本来就只发生在少数分隔/指代 token
   上，CB 的单层 16%→30% 重算已足够补回；DAF 丢掉 70% 文档 token 的跨层注意力
   对摘要连贯性的损害在 Rouge-L 这种弱指标上被放大。
4. **两后端一致性**：FLASH 对照（各 1 轮）Rouge-L 与 xformers 逐位相同
   （0.440/0.391/0.428），TTFT 差 <2%。

**工程实现假设**（论文未写死，非论文原方法）：多层偏差分数取平均聚合、选择层固定
第 4 层、状态融合 [重要文档|问题] 顺序拼接。环境差异：论文为 A100 +
Flash-Attention，本实验为 3090Ti + xformers/flash_attn 2.5.8。官方 pred.py 对
samsum 还有 "\n" 早停 trick（防模型循环复述 Dialogue），本实验未实现该 trick，
用 64 token 上限兜底，三方法条件相同。
