# 6k–10k 长上下文样本挑选报告

- 分词器：`mistralai/Mistral-7B-Instruct-v0.2`
- 挑选区间：总 token 数 [6000, 10000]（含问题）
- 入选总数：**24** 个
- 每个数据集最多取 8 个（按长度均匀抽样）
- 文档块拼法：`title\n\ntext`；问题拼法：与 `example/blend_musique.py` 的指令一致

## 入选样本

| 输出文件 | 来源 | 原下标 | 文档块数 | 总token数 | 问题 | 标准答案 |
|---|---|---|---|---|---|---|
| musique_s_070_len6009.json | musique_s.json | 70 | 10 | 6009 | Where was the person who wrote about the rioting b | University of Glasgow |
| musique_s_138_len6288.json | musique_s.json | 138 | 10 | 6288 | What recognition did the Jack Goes Boating directo | nominated for an Academy Award for Best Supporting Actor |
| musique_s_093_len6354.json | musique_s.json | 93 | 10 | 6354 | Which university employed the person that Mach ban | Charles University |
| musique_s_045_len6408.json | musique_s.json | 45 | 10 | 6408 | During what era was the manufacturer of Agni-V fou | From the 1950s to the 1970s |
| musique_s_040_len6457.json | musique_s.json | 40 | 10 | 6457 | What event caused the Liberal Party of Australia's | Lyons' death in 1939 |
| musique_s_063_len6522.json | musique_s.json | 63 | 10 | 6522 | In which state is Vera Barbosa's place of birth lo | Lisbon District |
| musique_s_110_len6604.json | musique_s.json | 110 | 10 | 6604 | Who is the creator of Mickey Mouse's spouse? | Walt Disney |
| musique_s_019_len6734.json | musique_s.json | 19 | 10 | 6734 | What team was Anna Benson's husband on? | Pittsburgh Pirates |

### musique_s.json：候选 149 个（8 入选 / 低于下限 1 / 超上限 0 / 缺字段 0）

候选长度分布：min=6009, 中位=6457, max=7898

| wikimqa_s_088_len6078.json | wikimqa_s.json | 88 | 10 | 6078 | Which country Keōpūolani's husband is from? | Kingdom of Hawaii |
| wikimqa_s_123_len6367.json | wikimqa_s.json | 123 | 10 | 6367 | Which film came out first, F The Prom or The Comed | The Comedians Of Comedy |
| wikimqa_s_060_len6451.json | wikimqa_s.json | 60 | 10 | 6451 | Who is the spouse of the performer of song This Ti | Jessi Colter |
| wikimqa_s_032_len6517.json | wikimqa_s.json | 32 | 10 | 6517 | Where did the performer of song Fantasy (George Mi | Goring-on-Thames, Oxfordshire |
| wikimqa_s_110_len6584.json | wikimqa_s.json | 110 | 10 | 6584 | Which film was released earlier, Moment Of Danger  | Moment Of Danger |
| wikimqa_s_010_len6665.json | wikimqa_s.json | 10 | 10 | 6665 | Which film has the director who was born earlier,  | The House Of The Seven Hawks |
| wikimqa_s_196_len6767.json | wikimqa_s.json | 196 | 10 | 6767 | Who is the maternal grandfather of Marie Louise, D | Ferdinand I of the Two Sicilies |
| wikimqa_s_108_len6910.json | wikimqa_s.json | 108 | 10 | 6910 | Where did the director of film Balls Out (2014 Fil | Tisch |

### wikimqa_s.json：候选 196 个（8 入选 / 低于下限 4 / 超上限 0 / 缺字段 0）

候选长度分布：min=6078, 中位=6584, max=7550

| samsum_124_len6001.json | samsum.json | 124 | 33 | 6001 | Dialogue: Miriam: heyo
Miriam: when do you get ba | Pegah is in class till 15:00. She will work from 17:00 till  |
| samsum_106_len6352.json | samsum.json | 106 | 25 | 6352 | Dialogue: Jim: Hello, welcome to the store, how ca | Jim tracks Finn's package number 45678 which will be deliver |
| samsum_168_len6663.json | samsum.json | 168 | 40 | 6663 | Dialogue: Adam: 10 minutes and I'm going in
Adam: | Adam is going to take an exam in ten minutes. |
| samsum_096_len7109.json | samsum.json | 96 | 29 | 7109 | Dialogue: Tom: we're few meters from you, can't yo | Elena is wearing the red jacket and Jeffrey can't see her no |
| samsum_062_len7444.json | samsum.json | 62 | 34 | 7444 | Dialogue: Natalie: Have you been to this new club  | Natalie is checking if it's worth going to the new club at R |
| samsum_032_len8287.json | samsum.json | 32 | 36 | 8287 | Dialogue: Jess: Guys, last night I dreamt I was a  | Jess dreamt she was a lion tamer. Lynn believes that dreams  |
| samsum_080_len8753.json | samsum.json | 80 | 43 | 8753 | Dialogue: Jasmine: Have you heard this song?
Jasm | Jasmine loves Charlie Puth and his new song. Paola thinks Ch |
| samsum_108_len9546.json | samsum.json | 108 | 41 | 9546 | Dialogue: Karen: you know that road from the swimm | Karen got stuck on the road from the swimming pool to Waitro |

### samsum.json：候选 40 个（8 入选 / 低于下限 47 / 超上限 113 / 缺字段 0）

候选长度分布：min=6001, 中位=7444, max=9995
