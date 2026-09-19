# LongBench 2wikimqa（Qwen2.5，≈6k 档）样本挑选报告

- 分词器：`Qwen/Qwen2.5-7B-Instruct`
- 数据来源：`experiments/longbench_src/data/2wikimqa.jsonl`（LongBench 官方 jsonl）
- 挑选区间：总 token 数 [5000, 7000]（含问题）
- 最多取 24 个（按长度均匀抽样）
- 切块：按 `Passage N:` 边界，每篇 Passage 一个文档块；问题拼法与 `example/blend_musique.py` 的指令一致

## 入选样本

| 输出文件 | 原行号 | 文档块数 | 总token数 | 问题 | 标准答案 |
|---|---|---|---|---|---|
| 2wikimqa_182_len5003.json | 182 | 10 | 5003 | Where was the husband of Maude Smith born? | Toronto, Ontario |
| 2wikimqa_069_len5077.json | 69 | 10 | 5077 | What is the place of birth of the director of film | Oklahoma City, Oklahoma |
| 2wikimqa_198_len5112.json | 198 | 10 | 5112 | Where was the director of film Love At First Sight | Milan |
| 2wikimqa_146_len5300.json | 146 | 10 | 5300 | Where was the director of film 00 Schneider – Jagd | Mülheim an der Ruhr |
| 2wikimqa_013_len5407.json | 13 | 10 | 5407 | When did William Le Poer Trench's father die? | 26 April 1872 |
| 2wikimqa_139_len5487.json | 139 | 10 | 5487 | Which film has the director who was born later, Th | Wolf Warrior |
| 2wikimqa_019_len5555.json | 19 | 10 | 5555 | Where was the director of film Kanneshwara Rama bo | Mysore |
| 2wikimqa_052_len5626.json | 52 | 10 | 5626 | Where was the place of death of Sancha Of Castile, | Palencia |
| 2wikimqa_074_len5686.json | 74 | 10 | 5686 | Where was the place of death of Anne Devereux's hu | Banbury |
| 2wikimqa_129_len5744.json | 129 | 10 | 5744 | Who was born first, Vytautas Straižys or Mirjam Po | Mirjam Polkunen |
| 2wikimqa_158_len5914.json | 158 | 10 | 5914 | Who is the paternal grandmother of King Kang Of Zh | Yi Jiang |
| 2wikimqa_068_len6050.json | 68 | 10 | 6050 | Which film has the director who died later, The Gr | La Belle Américaine |
| 2wikimqa_008_len6111.json | 8 | 10 | 6111 | Where did Helena Carroll's father study? | St Patrick's College |
| 2wikimqa_102_len6161.json | 102 | 10 | 6161 | Which country Dorothea Of Denmark, Duchess Of Meck | Norway |
| 2wikimqa_183_len6222.json | 183 | 10 | 6222 | Which film was released earlier, Forbidden Daughte | Forbidden Daughters |
| 2wikimqa_140_len6288.json | 140 | 10 | 6288 | Who is the paternal grandfather of Trịnh Doanh? | Trịnh Bính |
| 2wikimqa_046_len6386.json | 46 | 10 | 6386 | Where was the place of death of Agnes Of Hohenstau | Brunswick |
| 2wikimqa_034_len6441.json | 34 | 10 | 6441 | Which film whose director was born first, The Abdu | Wooden Crosses |
| 2wikimqa_152_len6599.json | 152 | 10 | 6599 | Where was the director of film Eisenstein In Guana | Newport, Wales |
| 2wikimqa_081_len6688.json | 81 | 10 | 6688 | Which film came out earlier, Closely Watched Train | Det Sande Ansigt |
| 2wikimqa_135_len6729.json | 135 | 10 | 6729 | Do both films, The Glass Castle (1950 Film) and Th | yes |
| 2wikimqa_123_len6755.json | 123 | 10 | 6755 | Which film came out first, F The Prom or The Comed | The Comedians Of Comedy |
| 2wikimqa_103_len6864.json | 103 | 10 | 6864 | What is the cause of death of Heonjong Of Goryeo's | illness |
| 2wikimqa_072_len6952.json | 72 | 10 | 6952 | Who is the father-in-law of Duke William Of Meckle | Prince Albert of Prussia |

候选 71 个（24 入选 / 低于下限 45 / 超上限 84 / 缺字段 0）

候选长度分布：min=5003, 中位=6111, max=6972
入选长度分布：min=5003, 中位=6111, max=6952, 均值=6007
