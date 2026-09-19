# RULER qa_1/qa_2（Qwen2.5，32768 档）样本转换报告

- 分词器：`Qwen/Qwen2.5-7B-Instruct`
- 数据来源：`experiments/ruler_src/data_gen/{qa_1,qa_2}/validation.jsonl`（RULER 官方 qa.py 生成，--max_seq_length 32768，seed 42，各 24 条）
- qa_1 = squad（单文档问答，干扰文档为随机 squad 段落）
- qa_2 = hotpotqa（多跳问答，干扰文档为 hotpotqa distractor）
- 结构：context 按 `Document N:` 切块；指令前缀拼在第 0 块开头；query = 尾部指令 + `Question: {query} Answer:`
- 评分主指标：SM（RULER string_match，答案列表任一命中即对），EM 记录

## 入选样本

| 输出文件 | 任务 | 原行号 | 文档块数 | 总token数 | gold（截断） |
|---|---|---|---|---|---|
| qa_1_000_len32420.json | qa_1 | 0 | 193 | 32420 | France |
| qa_1_001_len31728.json | qa_1 | 1 | 193 | 31728 | 10th and 11th centuries |
| qa_1_002_len32716.json | qa_1 | 2 | 193 | 32716 | Denmark, Iceland and Norway |
| qa_1_003_len30149.json | qa_1 | 3 | 183 | 30149 | Rollo |
| qa_1_004_len30471.json | qa_1 | 4 | 183 | 30471 | 10th century |
| qa_1_005_len32253.json | qa_1 | 5 | 193 | 32253 | William the Conqueror |
| qa_1_006_len32195.json | qa_1 | 6 | 193 | 32195 | Richard I |
| qa_1_007_len32600.json | qa_1 | 7 | 193 | 32600 | Catholic |
| qa_1_008_len30678.json | qa_1 | 8 | 183 | 30678 | Viking |
| qa_1_009_len32681.json | qa_1 | 9 | 193 | 32681 | 9th century |
| qa_1_010_len32585.json | qa_1 | 10 | 193 | 32585 | 911 |
| qa_1_011_len31585.json | qa_1 | 11 | 183 | 31585 | King Charles III |
| qa_1_012_len31552.json | qa_1 | 12 | 193 | 31552 | Seine |
| qa_1_013_len32635.json | qa_1 | 13 | 193 | 32635 | Rollo |
| qa_1_014_len32483.json | qa_1 | 14 | 183 | 32483 | Catholicism |
| qa_1_015_len30396.json | qa_1 | 15 | 183 | 30396 | north |
| qa_1_016_len32051.json | qa_1 | 16 | 183 | 32051 | fighting horsemen |
| qa_1_017_len30626.json | qa_1 | 17 | 173 | 30626 | Seljuk Turks |
| qa_1_018_len30128.json | qa_1 | 18 | 183 | 30128 | 1050s |
| qa_1_019_len32473.json | qa_1 | 19 | 193 | 32473 | 1060s |
| qa_1_020_len30589.json | qa_1 | 20 | 183 | 30589 | Alexius Komnenos |
| qa_1_021_len32038.json | qa_1 | 21 | 183 | 32038 | Afranji |
| qa_1_022_len30751.json | qa_1 | 22 | 183 | 30751 | Oursel |
| qa_1_023_len32382.json | qa_1 | 23 | 193 | 32382 | Turkish forces |

qa_1 长度分布：min=30128, 中位=32051, max=32716, 均值=31674

| qa_2_000_len31234.json | qa_2 | 0 | 228 | 31234 | yes |
| qa_2_001_len31871.json | qa_2 | 1 | 228 | 31871 | Chief of Protocol |
| qa_2_002_len30167.json | qa_2 | 2 | 228 | 30167 | Animorphs |
| qa_2_003_len31814.json | qa_2 | 3 | 228 | 31814 | no |
| qa_2_004_len32050.json | qa_2 | 4 | 228 | 32050 | Greenwich Village, New York City |
| qa_2_005_len32340.json | qa_2 | 5 | 228 | 32340 | YG Entertainment |
| qa_2_006_len31806.json | qa_2 | 6 | 228 | 31806 | Eenasul Fateh |
| qa_2_007_len31346.json | qa_2 | 7 | 218 | 31346 | 3,677 seated |
| qa_2_008_len32138.json | qa_2 | 8 | 218 | 32138 | Terry Richardson |
| qa_2_009_len31131.json | qa_2 | 9 | 228 | 31131 | yes |
| qa_2_010_len31807.json | qa_2 | 10 | 218 | 31807 | Kansas Song |
| qa_2_011_len31916.json | qa_2 | 11 | 228 | 31916 | David Weissman |
| qa_2_012_len31839.json | qa_2 | 12 | 238 | 31839 | 1999 |
| qa_2_013_len32311.json | qa_2 | 13 | 218 | 32311 | no |
| qa_2_014_len32118.json | qa_2 | 14 | 238 | 32118 | from 1986 to 2013 |
| qa_2_015_len32037.json | qa_2 | 15 | 228 | 32037 | 9,984 |
| qa_2_016_len32351.json | qa_2 | 16 | 238 | 32351 | the North Atlantic Conference |
| qa_2_017_len31167.json | qa_2 | 17 | 228 | 31167 | yes |
| qa_2_018_len31835.json | qa_2 | 18 | 238 | 31835 | 1969 until 1974 |
| qa_2_019_len30786.json | qa_2 | 19 | 228 | 30786 | Robert Erskine Childers DSC |
| qa_2_020_len30327.json | qa_2 | 20 | 218 | 30327 | Pedro Rodríguez |
| qa_2_021_len32159.json | qa_2 | 21 | 238 | 32159 | Sonic |
| qa_2_022_len31724.json | qa_2 | 22 | 238 | 31724 | keyboard function keys |
| qa_2_023_len31158.json | qa_2 | 23 | 228 | 31158 | Badly Drawn Boy |

qa_2 长度分布：min=30167, 中位=31835, max=32351, 均值=31643

