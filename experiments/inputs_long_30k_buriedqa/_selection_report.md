# 30k 长上下文样本构造报告（RULER 风格埋答案）

- 构造：种子题的 10 篇文档为答案块，干扰段来自其他样本的文档池，答案块埋于干扰池的 [0.2, 0.5, 0.8] 深度处
- 长度目标：[28000, 31500] token（含问题）
- 生成总数：**12**
- 种子：4 题 × 3 深度 = 12 个样本；已排除 6k–10k 档用过的样本
- 随机种子 42（重复生成结果一致）

| 输出文件 | 种子 | 埋深 | 块数 | 总token | 问题 | 标准答案 |
|---|---|---|---|---|---|---|
| wikimqa_s_000_at020_len30613.json | wikimqa_s.json#0 | 0.2 | 46 | 30613 | Where was the wife of Francis I Rákóczi born? | Ozalj |
| wikimqa_s_000_at050_len31156.json | wikimqa_s.json#0 | 0.5 | 48 | 31156 | Where was the wife of Francis I Rákóczi born? | Ozalj |
| wikimqa_s_000_at080_len29920.json | wikimqa_s.json#0 | 0.8 | 44 | 29920 | Where was the wife of Francis I Rákóczi born? | Ozalj |
| wikimqa_s_001_at020_len28335.json | wikimqa_s.json#1 | 0.2 | 44 | 28335 | Who is Sobe (Sister Of Saint Anne)'s grandchild? | John the Baptist |
| wikimqa_s_001_at050_len31074.json | wikimqa_s.json#1 | 0.5 | 49 | 31074 | Who is Sobe (Sister Of Saint Anne)'s grandchild? | John the Baptist |
| wikimqa_s_001_at080_len28565.json | wikimqa_s.json#1 | 0.8 | 44 | 28565 | Who is Sobe (Sister Of Saint Anne)'s grandchild? | John the Baptist |
| musique_s_000_at020_len27946.json | musique_s.json#0 | 0.2 | 44 | 27946 | Where was the author of Hannibal and Scipio educat | Exeter College |
| musique_s_000_at050_len31299.json | musique_s.json#0 | 0.5 | 49 | 31299 | Where was the author of Hannibal and Scipio educat | Exeter College |
| musique_s_000_at080_len30856.json | musique_s.json#0 | 0.8 | 47 | 30856 | Where was the author of Hannibal and Scipio educat | Exeter College |
| musique_s_001_at020_len29206.json | musique_s.json#1 | 0.2 | 48 | 29206 | In which county is Southern Maryland Electric Coop | Charles County |
| musique_s_001_at050_len28660.json | musique_s.json#1 | 0.5 | 44 | 28660 | In which county is Southern Maryland Electric Coop | Charles County |
| musique_s_001_at080_len28562.json | musique_s.json#1 | 0.8 | 45 | 28562 | In which county is Southern Maryland Electric Coop | Charles County |
