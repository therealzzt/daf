# 长上下文实验评分报告

- 样本目录：`experiments/inputs_qwen_samsum`（24 个样本）
- 评分：EM（精确匹配，多答案取最大）+ SM（子串匹配）+ token-F1（多答案取最大） + Rouge-L（摘要任务主指标，LongBench 官方口径）
- 归一化：小写/去冠词/去标点/Yes-No 规整（`example/utils.py`）
- 输入日志：`run_cb_flash_qwen_samsum_r1.log`, `run_daf_flash_qwen_samsum_r1.log`

## 一、逐样本明细（质量指标为多轮均值，TTFT 为多轮中位数）

| 样本 | 来源 | 长度 | 方法 | EM | SM | F1 | RL | TTFT缓存(s) | TTFT全量(s) | 缓存答案 | 全量答案 |
|---|---|---|---|---|---|---|---|---|----|---|---|
| samsum_000_len9508.json | samsum.jsonl | 9508 | CacheBlend | 0.00 | 1.00 | 0.77 | 0.80 | 0.686 | 1.789 | Emilia is still angry at William.  | Emilia is still angry with William. |
| samsum_000_len9508.json | samsum.jsonl | 9508 | DAF | 0.00 | 1.00 | 0.77 | 0.80 | 0.798 | 1.794 | Emilia is still angry at William.  | Emilia is still angry with William. |
| samsum_003_len8351.json | samsum.jsonl | 8351 | CacheBlend | 0.00 | 0.00 | 0.24 | 0.26 | 0.613 | 1.534 | Israel hasn't sewn anything yet, but Aut | Autumn has sewn a dress and shows Israel |
| samsum_003_len8351.json | samsum.jsonl | 8351 | DAF | 0.00 | 0.00 | 0.21 | 0.22 | 0.689 | 1.545 | Israel hasn't sewn anything yet, while A | Autumn has sewn a dress and shows Israel |
| samsum_004_len9891.json | samsum.jsonl | 9891 | CacheBlend | 0.00 | 0.00 | 0.47 | 0.43 | 0.763 | 1.847 | Brenda and Sandra, who used to work toge | Brenda and Sandra, who used to work toge |
| samsum_004_len9891.json | samsum.jsonl | 9891 | DAF | 0.00 | 0.00 | 0.49 | 0.40 | 0.844 | 1.862 | Brenda and Sandra, who used to work toge | Brenda and Sandra, who used to work toge |
| samsum_010_len9875.json | samsum.jsonl | 9875 | CacheBlend | 0.00 | 0.00 | 0.44 | 0.41 | 0.697 | 1.848 | Celine is not at home, but Cara still wa | Celine is not at home and can drop by in |
| samsum_010_len9875.json | samsum.jsonl | 9875 | DAF | 0.00 | 0.00 | 0.36 | 0.30 | 0.782 | 1.863 | Celine is not at home and Cara wanted to | Celine is not at home and can drop by in |
| samsum_011_len9677.json | samsum.jsonl | 9677 | CacheBlend | 0.00 | 0.00 | 0.66 | 0.38 | 0.698 | 1.815 | Janice has watched almost all of the top | Janice has watched almost all of the top |
| samsum_011_len9677.json | samsum.jsonl | 9677 | DAF | 0.00 | 0.00 | 0.55 | 0.31 | 0.778 | 1.831 | Janice has watched almost all of the top | Janice has watched almost all of the top |
| samsum_015_len9811.json | samsum.jsonl | 9811 | CacheBlend | 0.00 | 0.00 | 0.22 | 0.21 | 0.697 | 1.844 | Max shared his latest music project with | Max shared his latest music project with |
| samsum_015_len9811.json | samsum.jsonl | 9811 | DAF | 0.00 | 0.00 | 0.29 | 0.24 | 0.785 | 1.853 | Max shared a link to his latest music pr | Max shared his latest music project with |
| samsum_028_len9462.json | samsum.jsonl | 9462 | CacheBlend | 0.00 | 0.00 | 0.58 | 0.55 | 0.694 | 1.805 | Iris is trying to remove herself from an | Iris is trying to remove herself from an |
| samsum_028_len9462.json | samsum.jsonl | 9462 | DAF | 0.00 | 0.00 | 0.46 | 0.51 | 0.780 | 1.811 | Iris is trying to remove herself from an | Iris is trying to remove herself from an |
| samsum_063_len8439.json | samsum.jsonl | 8439 | CacheBlend | 0.00 | 1.00 | 0.78 | 0.75 | 0.613 | 1.609 | Railey will buy Tiffany a burger on his  | Tiffany asked Railey to buy her a burger |
| samsum_063_len8439.json | samsum.jsonl | 8439 | DAF | 0.00 | 0.00 | 0.35 | 0.38 | 0.695 | 1.611 | Tiffany asked Railey to buy her a burger | Tiffany asked Railey to buy her a burger |
| samsum_068_len9121.json | samsum.jsonl | 9121 | CacheBlend | 0.00 | 0.00 | 0.38 | 0.35 | 0.664 | 1.702 | Olivia and Jake are discussing their pla | Olivia has some tasks planned for the da |
| samsum_068_len9121.json | samsum.jsonl | 9121 | DAF | 0.00 | 0.00 | 0.42 | 0.36 | 0.741 | 1.708 | Olivia and Jake are discussing their pla | Olivia has some tasks planned for the da |
| samsum_071_len8973.json | samsum.jsonl | 8973 | CacheBlend | 0.00 | 0.00 | 0.63 | 0.57 | 0.642 | 1.686 | Paul will be home later than expected an | Paul will be home later than expected an |
| samsum_071_len8973.json | samsum.jsonl | 8973 | DAF | 0.00 | 0.00 | 0.55 | 0.51 | 0.725 | 1.691 | Paul will be home later than expected an | Paul will be home later than expected an |
| samsum_073_len8489.json | samsum.jsonl | 8489 | CacheBlend | 0.00 | 0.00 | 0.27 | 0.26 | 0.616 | 1.607 | Iris's husband got an interview, making  | Iris's husband got an interview, which m |
| samsum_073_len8489.json | samsum.jsonl | 8489 | DAF | 0.00 | 0.00 | 0.22 | 0.25 | 0.698 | 1.611 | Iris's husband has an interview, which m | Iris's husband got an interview, which m |
| samsum_074_len9713.json | samsum.jsonl | 9713 | CacheBlend | 0.00 | 0.00 | 0.64 | 0.63 | 0.699 | 1.843 | Derek wants Judy to feed his animals on  | Derek asks Judy to feed his animals on F |
| samsum_074_len9713.json | samsum.jsonl | 9713 | DAF | 0.00 | 0.00 | 0.67 | 0.62 | 0.784 | 1.853 | Derek wants Judy to feed his animals on  | Derek asks Judy to feed his animals on F |
| samsum_078_len8183.json | samsum.jsonl | 8183 | CacheBlend | 0.00 | 0.00 | 0.71 | 0.63 | 0.575 | 1.527 | William is making spaghetti and needs Ol | William is making spaghetti and asks Oli |
| samsum_078_len8183.json | samsum.jsonl | 8183 | DAF | 0.00 | 0.00 | 0.64 | 0.43 | 0.655 | 1.530 | William is making spaghetti and needs fr | William is making spaghetti and asks Oli |
| samsum_091_len9297.json | samsum.jsonl | 9297 | CacheBlend | 0.00 | 1.00 | 0.44 | 0.36 | 0.670 | 1.766 | Aimee is looking for Maryam and has trie | Aimee is looking for Maryam but couldn't |
| samsum_091_len9297.json | samsum.jsonl | 9297 | DAF | 0.00 | 1.00 | 0.44 | 0.36 | 0.753 | 1.769 | Aimee is looking for Maryam and has trie | Aimee is looking for Maryam but couldn't |
| samsum_093_len9159.json | samsum.jsonl | 9159 | CacheBlend | 0.00 | 0.00 | 0.56 | 0.52 | 0.649 | 1.708 | Patricia needs to talk to George in an h | Patricia wants to talk to George in an h |
| samsum_093_len9159.json | samsum.jsonl | 9159 | DAF | 0.00 | 0.00 | 0.56 | 0.52 | 0.730 | 1.713 | Patricia needs to talk to George in an h | Patricia wants to talk to George in an h |
| samsum_107_len8135.json | samsum.jsonl | 8135 | CacheBlend | 0.00 | 0.00 | 0.42 | 0.42 | 0.574 | 1.512 | Ben and Tom are meeting at the Oval Room | Tom and Ben are meeting at 2pm in the Ov |
| samsum_107_len8135.json | samsum.jsonl | 8135 | DAF | 0.00 | 0.00 | 0.44 | 0.30 | 0.649 | 1.516 | Tom and Ben are going to a meeting at 2p | Tom and Ben are meeting at 2pm in the Ov |
| samsum_112_len8429.json | samsum.jsonl | 8429 | CacheBlend | 0.00 | 0.00 | 0.28 | 0.35 | 0.612 | 1.555 | Marta thinks Jay is lying about having a | Marta thinks Jay is lying about having a |
| samsum_112_len8429.json | samsum.jsonl | 8429 | DAF | 0.00 | 0.00 | 0.22 | 0.26 | 0.686 | 1.561 | Marta thinks Jay is a pathological liar  | Marta thinks Jay is lying about having a |
| samsum_131_len9346.json | samsum.jsonl | 9346 | CacheBlend | 0.00 | 0.00 | 0.48 | 0.27 | 0.692 | 1.778 | Inez, Gosia, Alicja, and Patrycja are pl | Inez proposes a food evening and shares  |
| samsum_131_len9346.json | samsum.jsonl | 9346 | DAF | 0.00 | 0.00 | 0.54 | 0.24 | 0.776 | 1.787 | Inez, Gosia, Alicja, and Patrycja are pl | Inez proposes a food evening and shares  |
| samsum_134_len9592.json | samsum.jsonl | 9592 | CacheBlend | 0.00 | 0.00 | 0.21 | 0.13 | 0.697 | 1.823 | Kaylin and Amir are making fun of a pers | Kaylin and Amir are laughing at funny im |
| samsum_134_len9592.json | samsum.jsonl | 9592 | DAF | 0.00 | 0.00 | 0.23 | 0.21 | 0.786 | 1.828 | Kaylin and Amir are making fun of a pers | Kaylin and Amir are laughing at funny im |
| samsum_146_len9769.json | samsum.jsonl | 9769 | CacheBlend | 0.00 | 0.00 | 0.82 | 0.80 | 0.699 | 1.842 | Lilly will be late and Gabriel will orde | Lilly will be late and Gabriel will orde |
| samsum_146_len9769.json | samsum.jsonl | 9769 | DAF | 0.00 | 0.00 | 0.78 | 0.65 | 0.784 | 1.848 | Lilly will be late and Gabriel will orde | Lilly will be late and Gabriel will orde |
| samsum_153_len9318.json | samsum.jsonl | 9318 | CacheBlend | 0.00 | 0.00 | 0.31 | 0.29 | 0.690 | 1.766 | Simon is considering having Freddy sing  | Simon asked Freddy if he would be willin |
| samsum_153_len9318.json | samsum.jsonl | 9318 | DAF | 0.00 | 0.00 | 0.29 | 0.27 | 0.775 | 1.772 | Simon heard that Freddy has a good singi | Simon asked Freddy if he would be willin |
| samsum_187_len8519.json | samsum.jsonl | 8519 | CacheBlend | 0.00 | 0.00 | 0.43 | 0.41 | 0.632 | 1.612 | Chloe wants to go to Megan's house after | Lesley asks Chloe to walk the dog when s |
| samsum_187_len8519.json | samsum.jsonl | 8519 | DAF | 0.00 | 0.00 | 0.43 | 0.39 | 0.715 | 1.614 | Chloe wants to go to Megan's house after | Lesley asks Chloe to walk the dog when s |
| samsum_193_len8683.json | samsum.jsonl | 8683 | CacheBlend | 0.00 | 0.00 | 0.48 | 0.42 | 0.637 | 1.627 | Max's sister is studying in Shanghai, Ch | Max's sister is studying in Shanghai, Ch |
| samsum_193_len8683.json | samsum.jsonl | 8683 | DAF | 0.00 | 0.00 | 0.49 | 0.42 | 0.717 | 1.636 | Max's sister is studying in Shanghai, Ch | Max's sister is studying in Shanghai, Ch |
| samsum_198_len8272.json | samsum.jsonl | 8272 | CacheBlend | 0.00 | 0.00 | 0.60 | 0.36 | 0.609 | 1.531 | Lucy turned down Johnny's invitation to  | Lucy turned down Johnny's offer to go to |
| samsum_198_len8272.json | samsum.jsonl | 8272 | DAF | 0.00 | 0.00 | 0.65 | 0.42 | 0.696 | 1.533 | Lucy turned down Johnny's invitation to  | Lucy turned down Johnny's offer to go to |

## 二、按方法聚合（每轮一行，再给出跨轮中位数）

| 方法 | 轮次 | EM | SM | F1 | RL | TTFT缓存均值(s) | TTFT全量均值(s) | 缓存加速比 |
|---|---|---|---|---|---|---|------|---|
| CacheBlend | r1 | 0.000 | 0.125 | 0.492 | 0.440 | 0.659 | 1.707 | 2.59× |
| **CacheBlend** | **中位** | **0.000** | **0.125** | **0.492** | **0.440** | **0.659** | **1.707** | **2.59×** |
| DAF | r1 | 0.000 | 0.083 | 0.460 | 0.391 | 0.743 | 1.714 | 2.31× |
| **DAF** | **中位** | **0.000** | **0.083** | **0.460** | **0.391** | **0.743** | **1.714** | **2.31×** |
| vLLM全量 | r1 | 0.000 | 0.125 | 0.484 | 0.428 | - | 1.707 | 1.00× |
| vLLM全量 | r2 | 0.000 | 0.125 | 0.484 | 0.428 | - | 1.714 | 1.00× |
| **vLLM全量** | **中位** | **0.000** | **0.125** | **0.484** | **0.428** | - | **1.711** | **1.00×** |

## 三、按数据集×方法聚合（跨轮中位数；QA 数据集看 SM/F1，samsum 摘要看 F1）

| 数据集 | 样本数 | 方法 | EM中位 | SM中位 | F1中位 | TTFT缓存中位(s) |
|---|---|---|---|---|---|---|
| samsum.jsonl | 24 | CacheBlend | 0.000 | 0.000 | 0.475 | 0.667 |
| samsum.jsonl | 24 | DAF | 0.000 | 0.000 | 0.450 | 0.747 |
