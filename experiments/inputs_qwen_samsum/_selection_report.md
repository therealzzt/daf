# LongBench samsum（Qwen2.5，≈9k 档）样本挑选报告

- 分词器：`Qwen/Qwen2.5-7B-Instruct`
- 数据来源：`experiments/longbench_src/data/samsum.jsonl`（LongBench 官方 jsonl）
- 挑选区间：总 token 数 [8000, 10000]（含问题）
- 最多取 24 个（按长度均匀抽样）
- 结构：context = ~53 个 Dialogue+Summary few-shot 示例（按 `Dialogue: ` 切块）；query = input 原文（`Dialogue: <待摘要对话>\nSummary: `）；官方指令行拼在第 0 块开头
- 评分主指标：Rouge-L（LongBench 官方口径，rouge_score rougeL fmeasure）

## 入选样本

| 输出文件 | 原行号 | 文档块数 | 总token数 | 标准摘要（截断） |
|---|---|---|---|---|
| samsum_107_len8135.json | 107 | 50 | 8135 | Tom will meet Ben in the Oval Room at 2pm and tells him to b |
| samsum_078_len8183.json | 78 | 41 | 8183 | William is making spaghetti. Olivia will buy fresh tomatoes  |
| samsum_198_len8272.json | 198 | 47 | 8272 | Johny wants to go for a gig with Lucy. She prefers to go wit |
| samsum_003_len8351.json | 3 | 59 | 8351 | Autumn has sewn a dress. |
| samsum_112_len8429.json | 112 | 53 | 8429 | Randy has been lying about being rich and travelling around  |
| samsum_063_len8439.json | 63 | 43 | 8439 | Railey will buy Tiffany a burger. |
| samsum_073_len8489.json | 73 | 44 | 8489 | Pete has got an interview. His wife Iris is happy because he |
| samsum_187_len8519.json | 187 | 50 | 8519 | Somebody has to take the dog out. Chloe wants to go to Megan |
| samsum_193_len8683.json | 193 | 44 | 8683 | Max's sister is studying in Shanghai and she already speaks  |
| samsum_071_len8973.json | 71 | 50 | 8973 | Lena has her keys. Paul will be home later than he thought.  |
| samsum_068_len9121.json | 68 | 51 | 9121 | Olivia has to sort out her accounts and upload a few videos  |
| samsum_093_len9159.json | 93 | 44 | 9159 | Patricia cannot talk now. George will call her in about an h |
| samsum_091_len9297.json | 91 | 56 | 9297 | Aimee is looking for Maryam. |
| samsum_153_len9318.json | 153 | 50 | 9318 | Freddy will sing at the school concert and send Simon 3 back |
| samsum_131_len9346.json | 131 | 50 | 9346 | Patrycja and Inez enjoyed the Italian evening. Gosia chose a |
| samsum_028_len9462.json | 28 | 56 | 9462 | Iris is trying to remove herself from an expense. Ken has fi |
| samsum_000_len9508.json | 0 | 53 | 9508 | Emilia is still angry. |
| samsum_134_len9592.json | 134 | 56 | 9592 | Kaylin has set her alarm for tomorrow. |
| samsum_011_len9677.json | 11 | 53 | 9677 | Rachel sends a list of Top 50 films of 2018. Janice watched  |
| samsum_074_len9713.json | 74 | 55 | 9713 | Judy is staying for the weekend. Derek asks Judy to feed his |
| samsum_146_len9769.json | 146 | 57 | 9769 | Lilly will be late. Gabriel will order pasta with salmon and |
| samsum_015_len9811.json | 15 | 56 | 9811 | Jim will check out Max's latest music project when he gets h |
| samsum_010_len9875.json | 10 | 58 | 9875 | Celine is not at home, but she will call Cara before visitin |
| samsum_004_len9891.json | 4 | 53 | 9891 | Sandra and Brenda used to work together in the clothes facto |

候选 26 个（24 入选 / 低于下限 84 / 超上限 90 / 缺字段 0）

候选长度分布：min=8135, 中位=9297, max=9930
入选长度分布：min=8135, 中位=9297, max=9891, 均值=9084
