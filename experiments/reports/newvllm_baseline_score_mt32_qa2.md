# 长上下文实验评分报告

- 样本目录：`experiments/inputs_qwen_ruler30k_500`（1000 个样本）
- 评分：EM（精确匹配，多答案取最大）+ SM（子串匹配）+ token-F1（多答案取最大）
- 归一化：小写/去冠词/去标点/Yes-No 规整（`example/utils.py`）
- 输入日志：`run_newvllm_mt32_qa2_c00.log`, `run_newvllm_mt32_qa2_c01.log`, `run_newvllm_mt32_qa2_c02.log`, `run_newvllm_mt32_qa2_c03.log`, `run_newvllm_mt32_qa2_c04.log`

## 一、逐样本明细（质量指标为多轮均值，TTFT 为多轮中位数）

| 样本 | 来源 | 长度 | 方法 | EM | SM | F1 | TTFT缓存(s) | TTFT全量(s) | 缓存答案 | 全量答案 |
|---|---|---|---|---|---|---|---|----|---|---|
| qa_2_000_len31234.json | validation.jsonl (qa_2) | 31234 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.057 | 8.057 | No | No |
| qa_2_001_len31871.json | validation.jsonl (qa_2) | 31871 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.334 | 8.334 | Shirley Temple | Shirley Temple |
| qa_2_002_len30167.json | validation.jsonl (qa_2) | 30167 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.853 | 7.853 | The Hork-Bajir Chronicles | The Hork-Bajir Chronicles |
| qa_2_003_len31814.json | validation.jsonl (qa_2) | 31814 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.194 | 8.194 | No | No |
| qa_2_004_len32050.json | validation.jsonl (qa_2) | 32050 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.339 | 8.339 | Greenwich Village | Greenwich Village |
| qa_2_005_len32340.json | validation.jsonl (qa_2) | 32340 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.422 | 8.422 | YG Entertainment | YG Entertainment |
| qa_2_006_len31806.json | validation.jsonl (qa_2) | 31806 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.351 | 8.351 | Eenasul Fateh | Eenasul Fateh |
| qa_2_007_len31346.json | validation.jsonl (qa_2) | 31346 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.180 | 8.180 | 1,400 | 1,400 |
| qa_2_008_len32138.json | validation.jsonl (qa_2) | 32138 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.364 | 8.364 | Annie Morton | Annie Morton |
| qa_2_009_len31131.json | validation.jsonl (qa_2) | 31131 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.983 | 7.983 | Yes | Yes |
| qa_2_010_len31807.json | validation.jsonl (qa_2) | 31807 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.243 | 8.243 | Kansas Song | Kansas Song |
| qa_2_011_len31916.json | validation.jsonl (qa_2) | 31916 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.323 | 8.323 | David Weissman | David Weissman |
| qa_2_012_len31839.json | validation.jsonl (qa_2) | 31839 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.327 | 8.327 | 1999 | 1999 |
| qa_2_013_len32311.json | validation.jsonl (qa_2) | 32311 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.400 | 8.400 | Yes | Yes |
| qa_2_014_len32118.json | validation.jsonl (qa_2) | 32118 | CacheBlend | 0.00 | 0.00 | 0.40 | 9.074 | 9.074 | 1903–1912 and then again from 1912–1924 | 1903–1912 and then again from 1912–1924 |
| qa_2_015_len32037.json | validation.jsonl (qa_2) | 32037 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.891 | 8.891 | The document does not provide the popula | The document does not provide the popula |
| qa_2_016_len32351.json | validation.jsonl (qa_2) | 32351 | CacheBlend | 0.00 | 0.00 | 0.25 | 8.484 | 8.484 | America East Conference | America East Conference |
| qa_2_017_len31167.json | validation.jsonl (qa_2) | 31167 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.984 | 7.984 | Yes | Yes |
| qa_2_018_len31835.json | validation.jsonl (qa_2) | 31835 | CacheBlend | 0.00 | 0.00 | 0.89 | 8.507 | 8.507 | 1969-1974 | 1969-1974 |
| qa_2_019_len30786.json | validation.jsonl (qa_2) | 30786 | CacheBlend | 0.00 | 1.00 | 0.88 | 8.032 | 8.032 | Robert Erskine Childers | Robert Erskine Childers |
| qa_2_020_len30327.json | validation.jsonl (qa_2) | 30327 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.820 | 7.820 | Sergio Pérez | Sergio Pérez |
| qa_2_021_len32159.json | validation.jsonl (qa_2) | 32159 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.351 | 8.351 | Pooh | Pooh |
| qa_2_022_len31724.json | validation.jsonl (qa_2) | 31724 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.214 | 8.214 | Apple TV | Apple TV |
| qa_2_023_len31158.json | validation.jsonl (qa_2) | 31158 | CacheBlend | 0.00 | 1.00 | 0.15 | 9.004 | 9.004 | Based on the information provided, Badly | Based on the information provided, Badly |
| qa_2_024_len30318.json | validation.jsonl (qa_2) | 30318 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.822 | 7.822 | World's Best Goalkeeper | World's Best Goalkeeper |
| qa_2_025_len30613.json | validation.jsonl (qa_2) | 30613 | CacheBlend | 0.00 | 0.00 | 0.23 | 8.573 | 8.573 | Lee Hazlewood died in 2007 and wrote "Th | Lee Hazlewood died in 2007 and wrote "Th |
| qa_2_026_len32716.json | validation.jsonl (qa_2) | 32716 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.616 | 8.616 | 1838 | 1838 |
| qa_2_027_len32343.json | validation.jsonl (qa_2) | 32343 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.411 | 8.411 | yes | yes |
| qa_2_028_len30034.json | validation.jsonl (qa_2) | 30034 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.699 | 7.699 | Henry J. Kaiser | Henry J. Kaiser |
| qa_2_029_len29006.json | validation.jsonl (qa_2) | 29006 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.391 | 7.391 | Arena of Khazan | Arena of Khazan |
| qa_2_030_len32408.json | validation.jsonl (qa_2) | 32408 | CacheBlend | 0.00 | 1.00 | 0.57 | 8.754 | 8.754 | March 14, 2000 | March 14, 2000 |
| qa_2_031_len31794.json | validation.jsonl (qa_2) | 31794 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.490 | 8.490 | Buck-Tick hails from Japan. | Buck-Tick hails from Japan. |
| qa_2_032_len31191.json | validation.jsonl (qa_2) | 31191 | CacheBlend | 0.00 | 0.00 | 0.29 | 8.241 | 8.241 | Charles Nungesser and François Coli | Charles Nungesser and François Coli |
| qa_2_033_len31857.json | validation.jsonl (qa_2) | 31857 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.222 | 8.222 | Yes | Yes |
| qa_2_034_len30981.json | validation.jsonl (qa_2) | 30981 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.027 | 8.027 | Letters to Cleo | Letters to Cleo |
| qa_2_035_len31102.json | validation.jsonl (qa_2) | 31102 | CacheBlend | 0.00 | 0.00 | 0.55 | 8.059 | 8.059 | 1924 | 1924 |
| qa_2_036_len32115.json | validation.jsonl (qa_2) | 32115 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.401 | 8.401 | 2000 | 2000 |
| qa_2_037_len31683.json | validation.jsonl (qa_2) | 31683 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.237 | 8.237 | World War II | World War II |
| qa_2_038_len32687.json | validation.jsonl (qa_2) | 32687 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.508 | 8.508 | No | No |
| qa_2_039_len32205.json | validation.jsonl (qa_2) | 32205 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.388 | 8.388 | Ithaca | Ithaca |
| qa_2_040_len29707.json | validation.jsonl (qa_2) | 29707 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.592 | 7.592 | Scotch Collie | Scotch Collie |
| qa_2_041_len30563.json | validation.jsonl (qa_2) | 30563 | CacheBlend | 0.00 | 1.00 | 0.06 | 8.797 | 8.797 | Tata Consultancy Services is headquarter | Tata Consultancy Services is headquarter |
| qa_2_042_len32525.json | validation.jsonl (qa_2) | 32525 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.559 | 8.559 | 1962 | 1962 |
| qa_2_043_len30954.json | validation.jsonl (qa_2) | 30954 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.700 | 8.700 | The battle in which Giuseppe Arimondi lo | The battle in which Giuseppe Arimondi lo |
| qa_2_044_len31499.json | validation.jsonl (qa_2) | 31499 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.143 | 8.143 | Nelson Rockefeller | Nelson Rockefeller |
| qa_2_045_len32664.json | validation.jsonl (qa_2) | 32664 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.039 | 9.039 | Dirleton Castle borders on the south sid | Dirleton Castle borders on the south sid |
| qa_2_046_len31525.json | validation.jsonl (qa_2) | 31525 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.201 | 8.201 | Phil Spector | Phil Spector |
| qa_2_047_len31157.json | validation.jsonl (qa_2) | 31157 | CacheBlend | 0.00 | 1.00 | 0.50 | 8.766 | 8.766 | a forum in which the Nobel Peace Laureat | a forum in which the Nobel Peace Laureat |
| qa_2_048_len30718.json | validation.jsonl (qa_2) | 30718 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.809 | 7.809 | Yes | Yes |
| qa_2_049_len32426.json | validation.jsonl (qa_2) | 32426 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.509 | 8.509 | English Electric Canberra | English Electric Canberra |
| qa_2_050_len30583.json | validation.jsonl (qa_2) | 30583 | CacheBlend | 0.00 | 1.00 | 0.90 | 8.125 | 8.125 | 2009 Big 12 Conference football season | 2009 Big 12 Conference football season |
| qa_2_051_len29134.json | validation.jsonl (qa_2) | 29134 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.375 | 7.375 | 539 | 539 |
| qa_2_052_len32148.json | validation.jsonl (qa_2) | 32148 | CacheBlend | 0.00 | 0.00 | 0.20 | 8.443 | 8.443 | Darlington Raceway | Darlington Raceway |
| qa_2_053_len30874.json | validation.jsonl (qa_2) | 30874 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.016 | 8.016 | New York City | New York City |
| qa_2_054_len31684.json | validation.jsonl (qa_2) | 31684 | CacheBlend | 0.00 | 0.00 | 0.97 | 8.721 | 8.721 | Max Martin, Savan Kotecha, Ilya Salmanza | Max Martin, Savan Kotecha, Ilya Salmanza |
| qa_2_055_len32186.json | validation.jsonl (qa_2) | 32186 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.355 | 8.355 | Princeton | Princeton |
| qa_2_056_len32706.json | validation.jsonl (qa_2) | 32706 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.548 | 8.548 | Drifting | Drifting |
| qa_2_057_len31681.json | validation.jsonl (qa_2) | 31681 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.282 | 8.282 | Keith Bostic | Keith Bostic |
| qa_2_058_len32641.json | validation.jsonl (qa_2) | 32641 | CacheBlend | 0.00 | 0.00 | 0.44 | 8.631 | 8.631 | 1,146 | 1,146 |
| qa_2_059_len31590.json | validation.jsonl (qa_2) | 31590 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.142 | 8.142 | No | No |
| qa_2_060_len31574.json | validation.jsonl (qa_2) | 31574 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.178 | 8.178 | Larry Johnson | Larry Johnson |
| qa_2_061_len30328.json | validation.jsonl (qa_2) | 30328 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.828 | 7.828 | Ronald Shusett | Ronald Shusett |
| qa_2_062_len32496.json | validation.jsonl (qa_2) | 32496 | CacheBlend | 0.00 | 1.00 | 0.55 | 8.524 | 8.524 | Virginia Woolf | Virginia Woolf |
| qa_2_063_len30735.json | validation.jsonl (qa_2) | 30735 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.881 | 7.881 | 821 | 821 |
| qa_2_064_len31090.json | validation.jsonl (qa_2) | 31090 | CacheBlend | 0.00 | 1.00 | 0.56 | 8.387 | 8.387 | ExxonMobil operates in more than 70 coun | ExxonMobil operates in more than 70 coun |
| qa_2_065_len30489.json | validation.jsonl (qa_2) | 30489 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.800 | 7.800 | Charmed | Charmed |
| qa_2_066_len31221.json | validation.jsonl (qa_2) | 31221 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.139 | 8.139 | International Boxing Hall of Fame | International Boxing Hall of Fame |
| qa_2_067_len31901.json | validation.jsonl (qa_2) | 31901 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.285 | 8.285 | Usher | Usher |
| qa_2_068_len29747.json | validation.jsonl (qa_2) | 29747 | CacheBlend | 0.00 | 0.00 | 0.22 | 7.618 | 7.618 | Brian Doyle-Murray | Brian Doyle-Murray |
| qa_2_069_len31334.json | validation.jsonl (qa_2) | 31334 | CacheBlend | 0.00 | 0.00 | 0.07 | 9.071 | 9.071 | The 2017–18 Wigan Athletic F.C. season w | The 2017–18 Wigan Athletic F.C. season w |
| qa_2_070_len30409.json | validation.jsonl (qa_2) | 30409 | CacheBlend | 0.00 | 1.00 | 0.86 | 7.756 | 7.756 | Teen Titans | Teen Titans |
| qa_2_071_len30940.json | validation.jsonl (qa_2) | 30940 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.957 | 7.957 | Strasbourg | Strasbourg |
| qa_2_072_len32001.json | validation.jsonl (qa_2) | 32001 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.259 | 8.259 | orange | orange |
| qa_2_073_len32695.json | validation.jsonl (qa_2) | 32695 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.651 | 8.651 | Tromeo and Juliet | Tromeo and Juliet |
| qa_2_074_len28897.json | validation.jsonl (qa_2) | 28897 | CacheBlend | 0.00 | 0.00 | 0.40 | 7.249 | 7.249 | Bill Clinton | Bill Clinton |
| qa_2_075_len30720.json | validation.jsonl (qa_2) | 30720 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.881 | 7.881 | John John Florence | John John Florence |
| qa_2_076_len31469.json | validation.jsonl (qa_2) | 31469 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.106 | 8.106 | Sela | Sela |
| qa_2_077_len31196.json | validation.jsonl (qa_2) | 31196 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.029 | 8.029 | Ais | Ais |
| qa_2_078_len30936.json | validation.jsonl (qa_2) | 30936 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.920 | 7.920 | British | British |
| qa_2_079_len32456.json | validation.jsonl (qa_2) | 32456 | CacheBlend | 0.00 | 0.00 | 0.75 | 8.544 | 8.544 | 1895 | 1895 |
| qa_2_080_len30443.json | validation.jsonl (qa_2) | 30443 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.802 | 7.802 | Lempster | Lempster |
| qa_2_081_len30467.json | validation.jsonl (qa_2) | 30467 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.802 | 7.802 | Bob Seger | Bob Seger |
| qa_2_082_len32651.json | validation.jsonl (qa_2) | 32651 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.832 | 8.832 | the practice of requiring only men to re | the practice of requiring only men to re |
| qa_2_083_len31347.json | validation.jsonl (qa_2) | 31347 | CacheBlend | 0.00 | 1.00 | 0.89 | 8.139 | 8.139 | Mondelez International | Mondelez International |
| qa_2_084_len31142.json | validation.jsonl (qa_2) | 31142 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.100 | 8.100 | Mimi Alford | Mimi Alford |
| qa_2_085_len31063.json | validation.jsonl (qa_2) | 31063 | CacheBlend | 0.00 | 1.00 | 0.58 | 8.449 | 8.449 | Paul Manafort was born on April 1, 1949. | Paul Manafort was born on April 1, 1949. |
| qa_2_086_len32396.json | validation.jsonl (qa_2) | 32396 | CacheBlend | 0.00 | 0.00 | 0.25 | 8.517 | 8.517 | 1970 | 1970 |
| qa_2_087_len30960.json | validation.jsonl (qa_2) | 30960 | CacheBlend | 0.00 | 1.00 | 0.24 | 8.950 | 8.950 | Teide National Park is located in Teneri | Teide National Park is located in Teneri |
| qa_2_088_len32564.json | validation.jsonl (qa_2) | 32564 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.170 | 9.170 | The given documents do not provide speci | The given documents do not provide speci |
| qa_2_089_len29446.json | validation.jsonl (qa_2) | 29446 | CacheBlend | 0.00 | 1.00 | 0.00 | 7.430 | 7.430 | film director | film director |
| qa_2_090_len31022.json | validation.jsonl (qa_2) | 31022 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.975 | 7.975 | The Conversation | The Conversation |
| qa_2_091_len29529.json | validation.jsonl (qa_2) | 29529 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.465 | 7.465 | John Waters | John Waters |
| qa_2_092_len32060.json | validation.jsonl (qa_2) | 32060 | CacheBlend | 0.00 | 1.00 | 0.57 | 8.317 | 8.317 | Las Vegas | Las Vegas |
| qa_2_093_len31803.json | validation.jsonl (qa_2) | 31803 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.208 | 8.208 | No | No |
| qa_2_094_len32728.json | validation.jsonl (qa_2) | 32728 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.521 | 8.521 | November | November |
| qa_2_095_len30382.json | validation.jsonl (qa_2) | 30382 | CacheBlend | 0.00 | 1.00 | 0.25 | 8.146 | 8.146 | Tysons Galleria is located in Fairfax Co | Tysons Galleria is located in Fairfax Co |
| qa_2_096_len30281.json | validation.jsonl (qa_2) | 30281 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.306 | 8.306 | Bordan Tkachuk was the CEO of Viglen, wh | Bordan Tkachuk was the CEO of Viglen, wh |
| qa_2_097_len31183.json | validation.jsonl (qa_2) | 31183 | CacheBlend | 0.00 | 0.00 | 0.89 | 8.118 | 8.118 | Lev Yilmaz | Lev Yilmaz |
| qa_2_098_len31177.json | validation.jsonl (qa_2) | 31177 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.041 | 8.041 | Rabat | Rabat |
| qa_2_099_len31084.json | validation.jsonl (qa_2) | 31084 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.938 | 7.938 | no | no |
| qa_2_100_len29933.json | validation.jsonl (qa_2) | 29933 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.654 | 8.654 | The Black Seminoles lived mostly in dist | The Black Seminoles lived mostly in dist |
| qa_2_101_len32041.json | validation.jsonl (qa_2) | 32041 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.365 | 8.365 | Kathy Troccoli | Kathy Troccoli |
| qa_2_102_len31969.json | validation.jsonl (qa_2) | 31969 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.344 | 8.344 | 1999 | 1999 |
| qa_2_103_len29605.json | validation.jsonl (qa_2) | 29605 | CacheBlend | 0.00 | 1.00 | 0.33 | 8.297 | 8.297 | The stadium is located at the corner of  | The stadium is located at the corner of  |
| qa_2_104_len31540.json | validation.jsonl (qa_2) | 31540 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.210 | 8.210 | Leo Varadkar | Leo Varadkar |
| qa_2_105_len31867.json | validation.jsonl (qa_2) | 31867 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.280 | 8.280 | George Raft | George Raft |
| qa_2_106_len31791.json | validation.jsonl (qa_2) | 31791 | CacheBlend | 0.00 | 0.00 | 0.25 | 8.306 | 8.306 | 1946 | 1946 |
| qa_2_107_len31973.json | validation.jsonl (qa_2) | 31973 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.358 | 8.358 | valuation and investment analysis | valuation and investment analysis |
| qa_2_108_len32698.json | validation.jsonl (qa_2) | 32698 | CacheBlend | 0.00 | 1.00 | 0.64 | 9.054 | 9.054 | Zaheer Khan was born on 7 October 1978. | Zaheer Khan was born on 7 October 1978. |
| qa_2_109_len31517.json | validation.jsonl (qa_2) | 31517 | CacheBlend | 0.00 | 0.00 | 0.40 | 8.262 | 8.262 | B-1 Lancer | B-1 Lancer |
| qa_2_110_len32601.json | validation.jsonl (qa_2) | 32601 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.549 | 8.549 | Blake Shelton | Blake Shelton |
| qa_2_111_len30586.json | validation.jsonl (qa_2) | 30586 | CacheBlend | 0.00 | 0.00 | 0.07 | 8.716 | 8.716 | between approximately 1100 and 1425 AD,  | between approximately 1100 and 1425 AD,  |
| qa_2_112_len31868.json | validation.jsonl (qa_2) | 31868 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.320 | 8.320 | 1936 | 1936 |
| qa_2_113_len32716.json | validation.jsonl (qa_2) | 32716 | CacheBlend | 0.00 | 0.00 | 0.07 | 9.218 | 9.218 | Ralph Murphy died first. He passed away  | Ralph Murphy died first. He passed away  |
| qa_2_114_len30597.json | validation.jsonl (qa_2) | 30597 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.844 | 7.844 | Picric acid | Picric acid |
| qa_2_115_len31380.json | validation.jsonl (qa_2) | 31380 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.064 | 8.064 | 16 | 16 |
| qa_2_116_len31774.json | validation.jsonl (qa_2) | 31774 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.240 | 8.240 | Ghost Rider | Ghost Rider |
| qa_2_117_len31538.json | validation.jsonl (qa_2) | 31538 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.128 | 8.128 | Senator | Senator |
| qa_2_118_len32050.json | validation.jsonl (qa_2) | 32050 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.279 | 8.279 | No | No |
| qa_2_119_len30712.json | validation.jsonl (qa_2) | 30712 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.806 | 7.806 | Yes | Yes |
| qa_2_120_len32159.json | validation.jsonl (qa_2) | 32159 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.552 | 8.552 | U.S. Route 287 | U.S. Route 287 |
| qa_2_121_len32108.json | validation.jsonl (qa_2) | 32108 | CacheBlend | 0.00 | 1.00 | 0.21 | 9.242 | 9.242 | Caroline Carver played the role of Princ | Caroline Carver played the role of Princ |
| qa_2_122_len32398.json | validation.jsonl (qa_2) | 32398 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.513 | 8.513 | The Spiderwick Chronicles | The Spiderwick Chronicles |
| qa_2_123_len32274.json | validation.jsonl (qa_2) | 32274 | CacheBlend | 0.00 | 0.00 | 0.20 | 8.498 | 8.498 | Kohlberg Kravis Roberts | Kohlberg Kravis Roberts |
| qa_2_124_len31640.json | validation.jsonl (qa_2) | 31640 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.337 | 8.337 | You Can't Kill Stephen King | You Can't Kill Stephen King |
| qa_2_125_len31802.json | validation.jsonl (qa_2) | 31802 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.272 | 8.272 | Richmond River | Richmond River |
| qa_2_126_len32646.json | validation.jsonl (qa_2) | 32646 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.585 | 8.585 | Owsley Stanley | Owsley Stanley |
| qa_2_127_len32694.json | validation.jsonl (qa_2) | 32694 | CacheBlend | 0.00 | 0.00 | 0.20 | 8.608 | 8.608 | 1386 | 1386 |
| qa_2_128_len32415.json | validation.jsonl (qa_2) | 32415 | CacheBlend | 0.00 | 0.00 | 0.67 | 8.641 | 8.641 | downtown San Luis Obispo | downtown San Luis Obispo |
| qa_2_129_len30184.json | validation.jsonl (qa_2) | 30184 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.812 | 7.812 | Papa John's Cardinal Stadium | Papa John's Cardinal Stadium |
| qa_2_130_len32056.json | validation.jsonl (qa_2) | 32056 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.547 | 8.547 | LaLee's Kin: The Legacy of Cotton | LaLee's Kin: The Legacy of Cotton |
| qa_2_131_len31463.json | validation.jsonl (qa_2) | 31463 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.169 | 8.169 | Pirate's Cove | Pirate's Cove |
| qa_2_132_len32387.json | validation.jsonl (qa_2) | 32387 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.477 | 8.477 | Square Enix | Square Enix |
| qa_2_133_len31896.json | validation.jsonl (qa_2) | 31896 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.447 | 8.447 | The Marshall Mathers LP 2 | The Marshall Mathers LP 2 |
| qa_2_134_len32449.json | validation.jsonl (qa_2) | 32449 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.505 | 8.505 | Harsh Times | Harsh Times |
| qa_2_135_len31438.json | validation.jsonl (qa_2) | 31438 | CacheBlend | 0.00 | 0.00 | 0.57 | 8.121 | 8.121 | Jillian Bell | Jillian Bell |
| qa_2_136_len32293.json | validation.jsonl (qa_2) | 32293 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.466 | 8.466 | Texas Tech University | Texas Tech University |
| qa_2_137_len31037.json | validation.jsonl (qa_2) | 31037 | CacheBlend | 0.00 | 0.00 | 0.21 | 8.876 | 8.876 | Stapleton Cotton, 1st Viscount Combermer | Stapleton Cotton, 1st Viscount Combermer |
| qa_2_138_len31463.json | validation.jsonl (qa_2) | 31463 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.136 | 8.136 | stop motion animation | stop motion animation |
| qa_2_139_len32591.json | validation.jsonl (qa_2) | 32591 | CacheBlend | 0.00 | 0.00 | 0.50 | 8.570 | 8.570 | 2008 | 2008 |
| qa_2_140_len32661.json | validation.jsonl (qa_2) | 32661 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.577 | 8.577 | Oxford University | Oxford University |
| qa_2_141_len32543.json | validation.jsonl (qa_2) | 32543 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.538 | 8.538 | Roberta Vinci | Roberta Vinci |
| qa_2_142_len31613.json | validation.jsonl (qa_2) | 31613 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.712 | 8.712 | Truth in Science and Discovery both embr | Truth in Science and Discovery both embr |
| qa_2_143_len32393.json | validation.jsonl (qa_2) | 32393 | CacheBlend | 0.00 | 0.00 | 0.13 | 8.445 | 8.445 | Ian Watkins | Ian Watkins |
| qa_2_144_len32167.json | validation.jsonl (qa_2) | 32167 | CacheBlend | 0.00 | 1.00 | 0.26 | 9.257 | 9.257 | The movies Queen of Blood and Battle Bey | The movies Queen of Blood and Battle Bey |
| qa_2_145_len30511.json | validation.jsonl (qa_2) | 30511 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.857 | 7.857 | Erika Jayne | Erika Jayne |
| qa_2_146_len32711.json | validation.jsonl (qa_2) | 32711 | CacheBlend | 0.00 | 1.00 | 0.42 | 9.045 | 9.045 | The Reading Post changed its name to Rea | The Reading Post changed its name to Rea |
| qa_2_147_len30681.json | validation.jsonl (qa_2) | 30681 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.942 | 7.942 | Kingdom of the Isles | Kingdom of the Isles |
| qa_2_148_len31691.json | validation.jsonl (qa_2) | 31691 | CacheBlend | 0.00 | 1.00 | 0.11 | 8.840 | 8.840 | Bruce Spizer is an expert on The Beatles | Bruce Spizer is an expert on The Beatles |
| qa_2_149_len31748.json | validation.jsonl (qa_2) | 31748 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.312 | 8.312 | Chattahoochee | Chattahoochee |
| qa_2_150_len32058.json | validation.jsonl (qa_2) | 32058 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.279 | 8.279 | director | director |
| qa_2_151_len32224.json | validation.jsonl (qa_2) | 32224 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.429 | 8.429 | 1901 | 1901 |
| qa_2_152_len32110.json | validation.jsonl (qa_2) | 32110 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.360 | 8.360 | Argand lamp | Argand lamp |
| qa_2_153_len30330.json | validation.jsonl (qa_2) | 30330 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.711 | 7.711 | 16 | 16 |
| qa_2_154_len31886.json | validation.jsonl (qa_2) | 31886 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.221 | 8.221 | Yes | Yes |
| qa_2_155_len32709.json | validation.jsonl (qa_2) | 32709 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.711 | 8.711 | 2014 FIFA World Cup | 2014 FIFA World Cup |
| qa_2_156_len32375.json | validation.jsonl (qa_2) | 32375 | CacheBlend | 0.00 | 1.00 | 0.46 | 8.814 | 8.814 | The Seven Days Battles were fought earli | The Seven Days Battles were fought earli |
| qa_2_157_len32196.json | validation.jsonl (qa_2) | 32196 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.418 | 8.418 | Lush Ltd. | Lush Ltd. |
| qa_2_158_len32037.json | validation.jsonl (qa_2) | 32037 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.260 | 9.260 | Trafford Town Hall, which is home to Tra | Trafford Town Hall, which is home to Tra |
| qa_2_159_len32334.json | validation.jsonl (qa_2) | 32334 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.499 | 8.499 | 1989 | 1989 |
| qa_2_160_len32507.json | validation.jsonl (qa_2) | 32507 | CacheBlend | 0.00 | 1.00 | 0.71 | 8.755 | 8.755 | Kathryn Jean Martin "Kathy" Sullivan | Kathryn Jean Martin "Kathy" Sullivan |
| qa_2_161_len31707.json | validation.jsonl (qa_2) | 31707 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.249 | 8.249 | Frank Lowy | Frank Lowy |
| qa_2_162_len32246.json | validation.jsonl (qa_2) | 32246 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.431 | 8.431 | Summerlin, Nevada | Summerlin, Nevada |
| qa_2_163_len30403.json | validation.jsonl (qa_2) | 30403 | CacheBlend | 0.00 | 0.00 | 0.89 | 7.787 | 7.787 | Kelly Osbourne | Kelly Osbourne |
| qa_2_164_len32441.json | validation.jsonl (qa_2) | 32441 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.537 | 8.537 | Sean Yseult | Sean Yseult |
| qa_2_165_len31303.json | validation.jsonl (qa_2) | 31303 | CacheBlend | 0.00 | 0.00 | 0.07 | 8.995 | 8.995 | 3/4th of the Food and Drug Administratio | 3/4th of the Food and Drug Administratio |
| qa_2_166_len32160.json | validation.jsonl (qa_2) | 32160 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.434 | 8.434 | McFarland, USA | McFarland, USA |
| qa_2_167_len32056.json | validation.jsonl (qa_2) | 32056 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.331 | 8.331 | Wichita | Wichita |
| qa_2_168_len30372.json | validation.jsonl (qa_2) | 30372 | CacheBlend | 0.00 | 0.00 | 0.37 | 8.687 | 8.687 | The 58th quadrennial American presidenti | The 58th quadrennial American presidenti |
| qa_2_169_len32303.json | validation.jsonl (qa_2) | 32303 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.393 | 8.393 | Roman | Roman |
| qa_2_170_len29727.json | validation.jsonl (qa_2) | 29727 | CacheBlend | 0.00 | 0.00 | 0.50 | 7.630 | 7.630 | Sue Grafton | Sue Grafton |
| qa_2_171_len31132.json | validation.jsonl (qa_2) | 31132 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.102 | 8.102 | Brooklyn, New York | Brooklyn, New York |
| qa_2_172_len31576.json | validation.jsonl (qa_2) | 31576 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.290 | 8.290 | 48,982 | 48,982 |
| qa_2_173_len32568.json | validation.jsonl (qa_2) | 32568 | CacheBlend | 0.00 | 0.00 | 0.89 | 8.588 | 8.588 | Pasek and Paul | Pasek and Paul |
| qa_2_174_len32411.json | validation.jsonl (qa_2) | 32411 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.490 | 8.490 | Niagara Park | Niagara Park |
| qa_2_175_len32470.json | validation.jsonl (qa_2) | 32470 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.432 | 8.432 | yes | yes |
| qa_2_176_len31431.json | validation.jsonl (qa_2) | 31431 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.040 | 8.040 | works | works |
| qa_2_177_len31630.json | validation.jsonl (qa_2) | 31630 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.103 | 9.103 | The small settlement situated on the riv | The small settlement situated on the riv |
| qa_2_178_len30925.json | validation.jsonl (qa_2) | 30925 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.959 | 7.959 | William Bonfield | William Bonfield |
| qa_2_179_len32423.json | validation.jsonl (qa_2) | 32423 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.421 | 8.421 | Scotland | Scotland |
| qa_2_180_len29197.json | validation.jsonl (qa_2) | 29197 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.356 | 7.356 | The Drop-In | The Drop-In |
| qa_2_181_len32221.json | validation.jsonl (qa_2) | 32221 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.342 | 8.342 | Welcome | Welcome |
| qa_2_182_len31321.json | validation.jsonl (qa_2) | 31321 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.103 | 8.103 | George Harrison | George Harrison |
| qa_2_183_len31596.json | validation.jsonl (qa_2) | 31596 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.178 | 8.178 | center | center |
| qa_2_184_len32380.json | validation.jsonl (qa_2) | 32380 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.554 | 8.554 | 1952 | 1952 |
| qa_2_185_len30717.json | validation.jsonl (qa_2) | 30717 | CacheBlend | 0.00 | 1.00 | 0.67 | 7.879 | 7.879 | third season | third season |
| qa_2_186_len32520.json | validation.jsonl (qa_2) | 32520 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.692 | 8.692 | Mark Neveldine and Brian Taylor | Mark Neveldine and Brian Taylor |
| qa_2_187_len30471.json | validation.jsonl (qa_2) | 30471 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.902 | 7.902 | Hidden America with Jonah Ray | Hidden America with Jonah Ray |
| qa_2_188_len31954.json | validation.jsonl (qa_2) | 31954 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.288 | 8.288 | Yes | Yes |
| qa_2_189_len31141.json | validation.jsonl (qa_2) | 31141 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.123 | 8.123 | 1943 | 1943 |
| qa_2_190_len31498.json | validation.jsonl (qa_2) | 31498 | CacheBlend | 0.00 | 0.00 | 0.44 | 8.210 | 8.210 | Oregon State Beavers | Oregon State Beavers |
| qa_2_191_len29900.json | validation.jsonl (qa_2) | 29900 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.575 | 7.575 | NBC | NBC |
| qa_2_192_len31226.json | validation.jsonl (qa_2) | 31226 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.100 | 8.100 | Algerian | Algerian |
| qa_2_193_len31968.json | validation.jsonl (qa_2) | 31968 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.366 | 8.366 | New York City | New York City |
| qa_2_194_len32169.json | validation.jsonl (qa_2) | 32169 | CacheBlend | 0.00 | 1.00 | 0.83 | 8.624 | 8.624 | Severe acute respiratory syndrome (SARS) | Severe acute respiratory syndrome (SARS) |
| qa_2_195_len32576.json | validation.jsonl (qa_2) | 32576 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.572 | 8.572 | Wendell Berry | Wendell Berry |
| qa_2_196_len31556.json | validation.jsonl (qa_2) | 31556 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.265 | 8.265 | Vikram Bhatt | Vikram Bhatt |
| qa_2_197_len29159.json | validation.jsonl (qa_2) | 29159 | CacheBlend | 0.00 | 1.00 | 0.33 | 7.773 | 7.773 | Robert Smith founded General Mills, whic | Robert Smith founded General Mills, whic |
| qa_2_198_len32190.json | validation.jsonl (qa_2) | 32190 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.289 | 9.289 | The character from Son of al Qaeda decla | The character from Son of al Qaeda decla |
| qa_2_199_len29142.json | validation.jsonl (qa_2) | 29142 | CacheBlend | 0.00 | 0.00 | 0.17 | 7.433 | 7.433 | Cyclone Gonu | Cyclone Gonu |
| qa_2_200_len31724.json | validation.jsonl (qa_2) | 31724 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.541 | 8.541 | Snatch premiered on the streaming servic | Snatch premiered on the streaming servic |
| qa_2_201_len30493.json | validation.jsonl (qa_2) | 30493 | CacheBlend | 0.00 | 1.00 | 0.80 | 7.873 | 7.873 | Salford City | Salford City |
| qa_2_202_len31148.json | validation.jsonl (qa_2) | 31148 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.005 | 8.005 | Michael Graves | Michael Graves |
| qa_2_203_len32602.json | validation.jsonl (qa_2) | 32602 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.493 | 9.493 | American Hustle received the award for B | American Hustle received the award for B |
| qa_2_204_len32407.json | validation.jsonl (qa_2) | 32407 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.503 | 8.503 | Kevin Spacey | Kevin Spacey |
| qa_2_205_len31610.json | validation.jsonl (qa_2) | 31610 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.973 | 8.973 | Onika Tanya Maraj is not mentioned as a  | Onika Tanya Maraj is not mentioned as a  |
| qa_2_206_len31049.json | validation.jsonl (qa_2) | 31049 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.937 | 7.937 | Yes | Yes |
| qa_2_207_len30987.json | validation.jsonl (qa_2) | 30987 | CacheBlend | 0.00 | 1.00 | 0.71 | 8.031 | 8.031 | Charles Hastings Judd | Charles Hastings Judd |
| qa_2_208_len30476.json | validation.jsonl (qa_2) | 30476 | CacheBlend | 0.00 | 0.00 | 0.71 | 7.836 | 7.836 | Pavel Alexandrov | Pavel Alexandrov |
| qa_2_209_len31760.json | validation.jsonl (qa_2) | 31760 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.246 | 8.246 | Gal Gadot | Gal Gadot |
| qa_2_210_len29911.json | validation.jsonl (qa_2) | 29911 | CacheBlend | 0.00 | 0.00 | 0.34 | 8.344 | 8.344 | Captain J. Edward Snyder was the captain | Captain J. Edward Snyder was the captain |
| qa_2_211_len31680.json | validation.jsonl (qa_2) | 31680 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.199 | 8.199 | Jackson Storm | Jackson Storm |
| qa_2_212_len32035.json | validation.jsonl (qa_2) | 32035 | CacheBlend | 0.00 | 0.00 | 0.67 | 8.418 | 8.418 | 712 Fifth Avenue | 712 Fifth Avenue |
| qa_2_213_len31506.json | validation.jsonl (qa_2) | 31506 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.104 | 8.104 | right-hand | right-hand |
| qa_2_214_len31684.json | validation.jsonl (qa_2) | 31684 | CacheBlend | 0.00 | 0.00 | 0.80 | 8.235 | 8.235 | Arthur Conan Doyle | Arthur Conan Doyle |
| qa_2_215_len32235.json | validation.jsonl (qa_2) | 32235 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.400 | 8.400 | Knoxville | Knoxville |
| qa_2_216_len32245.json | validation.jsonl (qa_2) | 32245 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.328 | 8.328 | five | five |
| qa_2_217_len31129.json | validation.jsonl (qa_2) | 31129 | CacheBlend | 0.00 | 0.00 | 0.50 | 8.091 | 8.091 | Sasanian Empire | Sasanian Empire |
| qa_2_218_len32672.json | validation.jsonl (qa_2) | 32672 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.575 | 8.575 | Erinsborough | Erinsborough |
| qa_2_219_len31578.json | validation.jsonl (qa_2) | 31578 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.170 | 8.170 | Samoa | Samoa |
| qa_2_220_len31878.json | validation.jsonl (qa_2) | 31878 | CacheBlend | 0.00 | 1.00 | 0.62 | 8.288 | 8.288 | Kimmy Schmidt | Kimmy Schmidt |
| qa_2_221_len32711.json | validation.jsonl (qa_2) | 32711 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.717 | 8.717 | Filthy Rich & Catflap | Filthy Rich & Catflap |
| qa_2_222_len31806.json | validation.jsonl (qa_2) | 31806 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.205 | 8.205 | Yes | Yes |
| qa_2_223_len32483.json | validation.jsonl (qa_2) | 32483 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.599 | 8.599 | 19th century | 19th century |
| qa_2_224_len32254.json | validation.jsonl (qa_2) | 32254 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.388 | 8.388 | Awake | Awake |
| qa_2_225_len32092.json | validation.jsonl (qa_2) | 32092 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.347 | 8.347 | French Canadians | French Canadians |
| qa_2_226_len30714.json | validation.jsonl (qa_2) | 30714 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.821 | 7.821 | American | American |
| qa_2_227_len30895.json | validation.jsonl (qa_2) | 30895 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.151 | 9.151 | The bridge in the Öresund Region that co | The bridge in the Öresund Region that co |
| qa_2_228_len32344.json | validation.jsonl (qa_2) | 32344 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.551 | 8.551 | Clint Eastwood | Clint Eastwood |
| qa_2_229_len31901.json | validation.jsonl (qa_2) | 31901 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.386 | 8.386 | Marsilea | Marsilea |
| qa_2_230_len31964.json | validation.jsonl (qa_2) | 31964 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.398 | 8.398 | St James's Palace | St James's Palace |
| qa_2_231_len30703.json | validation.jsonl (qa_2) | 30703 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.863 | 7.863 | Starlite | Starlite |
| qa_2_232_len31284.json | validation.jsonl (qa_2) | 31284 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.201 | 8.201 | CapitaGreen in Singapore | CapitaGreen in Singapore |
| qa_2_233_len30764.json | validation.jsonl (qa_2) | 30764 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.035 | 8.035 | Craig v. Boren | Craig v. Boren |
| qa_2_234_len31441.json | validation.jsonl (qa_2) | 31441 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.111 | 8.111 | Waterfront | Waterfront |
| qa_2_235_len30237.json | validation.jsonl (qa_2) | 30237 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.763 | 7.763 | Arrowhead Stadium | Arrowhead Stadium |
| qa_2_236_len32094.json | validation.jsonl (qa_2) | 32094 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.348 | 8.348 | cranberries | cranberries |
| qa_2_237_len32400.json | validation.jsonl (qa_2) | 32400 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.595 | 8.595 | Strip search phone call scam | Strip search phone call scam |
| qa_2_238_len30903.json | validation.jsonl (qa_2) | 30903 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.050 | 8.050 | Logan International Airport | Logan International Airport |
| qa_2_239_len30911.json | validation.jsonl (qa_2) | 30911 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.048 | 8.048 | Ehretia | Ehretia |
| qa_2_240_len30810.json | validation.jsonl (qa_2) | 30810 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.063 | 8.063 | Frank Thomas' Big Hurt | Frank Thomas' Big Hurt |
| qa_2_241_len31387.json | validation.jsonl (qa_2) | 31387 | CacheBlend | 0.00 | 0.00 | 0.18 | 8.238 | 8.238 | Paige O'Hara | Paige O'Hara |
| qa_2_242_len31624.json | validation.jsonl (qa_2) | 31624 | CacheBlend | 0.00 | 1.00 | 0.62 | 8.237 | 8.237 | Reggie Jackson | Reggie Jackson |
| qa_2_243_len31589.json | validation.jsonl (qa_2) | 31589 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.195 | 8.195 | Peter Perry | Peter Perry |
| qa_2_244_len31686.json | validation.jsonl (qa_2) | 31686 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.264 | 8.264 | Sarod | Sarod |
| qa_2_245_len30310.json | validation.jsonl (qa_2) | 30310 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.113 | 8.113 | Lemony Snicket's A Series of Unfortunate | Lemony Snicket's A Series of Unfortunate |
| qa_2_246_len32327.json | validation.jsonl (qa_2) | 32327 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.456 | 8.456 | 25 | 25 |
| qa_2_247_len31801.json | validation.jsonl (qa_2) | 31801 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.219 | 8.219 | No | No |
| qa_2_248_len31406.json | validation.jsonl (qa_2) | 31406 | CacheBlend | 0.00 | 0.00 | 0.80 | 8.111 | 8.111 | Thomas Friedman | Thomas Friedman |
| qa_2_249_len29731.json | validation.jsonl (qa_2) | 29731 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.564 | 7.564 | Taylor Swift | Taylor Swift |
| qa_2_250_len32062.json | validation.jsonl (qa_2) | 32062 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.337 | 8.337 | Istanbul | Istanbul |
| qa_2_251_len31295.json | validation.jsonl (qa_2) | 31295 | CacheBlend | 0.00 | 1.00 | 0.26 | 9.218 | 9.218 | Opry Mills is a super-regional shopping  | Opry Mills is a super-regional shopping  |
| qa_2_252_len32173.json | validation.jsonl (qa_2) | 32173 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.560 | 9.560 | It is not possible to determine this fro | It is not possible to determine this fro |
| qa_2_253_len30672.json | validation.jsonl (qa_2) | 30672 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.976 | 7.976 | Alain Resnais | Alain Resnais |
| qa_2_254_len32374.json | validation.jsonl (qa_2) | 32374 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.508 | 8.508 | 600 | 600 |
| qa_2_255_len30428.json | validation.jsonl (qa_2) | 30428 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.826 | 7.826 | Berea College | Berea College |
| qa_2_256_len32231.json | validation.jsonl (qa_2) | 32231 | CacheBlend | 0.00 | 0.00 | 0.75 | 8.470 | 8.470 | 1944 | 1944 |
| qa_2_257_len32091.json | validation.jsonl (qa_2) | 32091 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.391 | 8.391 | House of Windsor | House of Windsor |
| qa_2_258_len30288.json | validation.jsonl (qa_2) | 30288 | CacheBlend | 0.00 | 1.00 | 0.47 | 8.231 | 8.231 | James Burke produced the documentary tel | James Burke produced the documentary tel |
| qa_2_259_len32210.json | validation.jsonl (qa_2) | 32210 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.457 | 8.457 | Gulangyu | Gulangyu |
| qa_2_260_len31376.json | validation.jsonl (qa_2) | 31376 | CacheBlend | 0.00 | 1.00 | 0.27 | 8.721 | 8.721 | Kurt Vonnegut, best known for "Slaughter | Kurt Vonnegut, best known for "Slaughter |
| qa_2_261_len32162.json | validation.jsonl (qa_2) | 32162 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.414 | 8.414 | Leinster | Leinster |
| qa_2_262_len31692.json | validation.jsonl (qa_2) | 31692 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.772 | 8.772 | Sean Connor played for Ards in the NIFL  | Sean Connor played for Ards in the NIFL  |
| qa_2_263_len32274.json | validation.jsonl (qa_2) | 32274 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.429 | 8.429 | Romeo | Romeo |
| qa_2_264_len32213.json | validation.jsonl (qa_2) | 32213 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.385 | 8.385 | SKUM | SKUM |
| qa_2_265_len31537.json | validation.jsonl (qa_2) | 31537 | CacheBlend | 0.00 | 0.00 | 0.67 | 8.357 | 8.357 | Anabolic-androgenic steroids | Anabolic-androgenic steroids |
| qa_2_266_len30372.json | validation.jsonl (qa_2) | 30372 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.860 | 7.860 | Don S. Davis | Don S. Davis |
| qa_2_267_len32695.json | validation.jsonl (qa_2) | 32695 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.692 | 8.692 | Lady Antebellum | Lady Antebellum |
| qa_2_268_len32712.json | validation.jsonl (qa_2) | 32712 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.583 | 9.583 | The National Archives and Library of Eth | The National Archives and Library of Eth |
| qa_2_269_len32098.json | validation.jsonl (qa_2) | 32098 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.392 | 8.392 | Anna Clyne | Anna Clyne |
| qa_2_270_len32727.json | validation.jsonl (qa_2) | 32727 | CacheBlend | 0.00 | 1.00 | 0.75 | 8.838 | 8.838 | USS Essex (CV-9) | USS Essex (CV-9) |
| qa_2_271_len30866.json | validation.jsonl (qa_2) | 30866 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.894 | 7.894 | Yes | Yes |
| qa_2_272_len28671.json | validation.jsonl (qa_2) | 28671 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.192 | 7.192 | Catwoman | Catwoman |
| qa_2_273_len31791.json | validation.jsonl (qa_2) | 31791 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.306 | 8.306 | 20 mi | 20 mi |
| qa_2_274_len31221.json | validation.jsonl (qa_2) | 31221 | CacheBlend | 0.00 | 0.00 | 0.89 | 8.092 | 8.092 | Laura Osnes | Laura Osnes |
| qa_2_275_len30770.json | validation.jsonl (qa_2) | 30770 | CacheBlend | 0.00 | 1.00 | 0.57 | 8.037 | 8.037 | more than 20 | more than 20 |
| qa_2_276_len30627.json | validation.jsonl (qa_2) | 30627 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.935 | 7.935 | Eucryphia | Eucryphia |
| qa_2_277_len32523.json | validation.jsonl (qa_2) | 32523 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.506 | 8.506 | Venice | Venice |
| qa_2_278_len32511.json | validation.jsonl (qa_2) | 32511 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.591 | 8.591 | Waimea Bay | Waimea Bay |
| qa_2_279_len31696.json | validation.jsonl (qa_2) | 31696 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.266 | 8.266 | Christianity Today | Christianity Today |
| qa_2_280_len31845.json | validation.jsonl (qa_2) | 31845 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.279 | 8.279 | James Franco | James Franco |
| qa_2_281_len32508.json | validation.jsonl (qa_2) | 32508 | CacheBlend | 0.00 | 1.00 | 0.73 | 8.590 | 8.590 | Humphrey Goodman | Humphrey Goodman |
| qa_2_282_len30671.json | validation.jsonl (qa_2) | 30671 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.807 | 7.807 | December | December |
| qa_2_283_len29523.json | validation.jsonl (qa_2) | 29523 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.515 | 7.515 | Sacred Planet | Sacred Planet |
| qa_2_284_len30416.json | validation.jsonl (qa_2) | 30416 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.776 | 7.776 | Calgary | Calgary |
| qa_2_285_len32363.json | validation.jsonl (qa_2) | 32363 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.594 | 8.594 | Ferdi Taygan | Ferdi Taygan |
| qa_2_286_len31811.json | validation.jsonl (qa_2) | 31811 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.243 | 9.243 | There is no information provided about t | There is no information provided about t |
| qa_2_287_len31209.json | validation.jsonl (qa_2) | 31209 | CacheBlend | 0.00 | 1.00 | 0.27 | 9.158 | 9.158 | Ian Harland studied at Peterhouse, Cambr | Ian Harland studied at Peterhouse, Cambr |
| qa_2_288_len32482.json | validation.jsonl (qa_2) | 32482 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.550 | 8.550 | Suicide Kings | Suicide Kings |
| qa_2_289_len32306.json | validation.jsonl (qa_2) | 32306 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.794 | 8.794 | Hey Pa! There's a Goat on the Roof | Hey Pa! There's a Goat on the Roof |
| qa_2_290_len32438.json | validation.jsonl (qa_2) | 32438 | CacheBlend | 0.00 | 0.00 | 0.36 | 8.621 | 8.621 | 23-16 | 23-16 |
| qa_2_291_len30709.json | validation.jsonl (qa_2) | 30709 | CacheBlend | 0.00 | 1.00 | 0.53 | 7.989 | 7.989 | 8,211 | 8,211 |
| qa_2_292_len32718.json | validation.jsonl (qa_2) | 32718 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.563 | 8.563 | Salvie | Salvie |
| qa_2_293_len31209.json | validation.jsonl (qa_2) | 31209 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.049 | 8.049 | Architect | Architect |
| qa_2_294_len31094.json | validation.jsonl (qa_2) | 31094 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.050 | 8.050 | Will Arnett | Will Arnett |
| qa_2_295_len29867.json | validation.jsonl (qa_2) | 29867 | CacheBlend | 0.00 | 0.00 | 0.40 | 7.594 | 7.594 | German Army | German Army |
| qa_2_296_len30731.json | validation.jsonl (qa_2) | 30731 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.905 | 7.905 | the Canary Islands | the Canary Islands |
| qa_2_297_len32181.json | validation.jsonl (qa_2) | 32181 | CacheBlend | 0.00 | 1.00 | 0.59 | 9.135 | 9.135 | The Treaty of Gandamak officially ended  | The Treaty of Gandamak officially ended  |
| qa_2_298_len32651.json | validation.jsonl (qa_2) | 32651 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.497 | 8.497 | Oregon | Oregon |
| qa_2_299_len32200.json | validation.jsonl (qa_2) | 32200 | CacheBlend | 0.00 | 1.00 | 0.35 | 9.566 | 9.566 | The rock band that sang "All Join Hands" | The rock band that sang "All Join Hands" |
| qa_2_300_len32724.json | validation.jsonl (qa_2) | 32724 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.635 | 8.635 | Greene King | Greene King |
| qa_2_301_len32422.json | validation.jsonl (qa_2) | 32422 | CacheBlend | 0.00 | 1.00 | 0.75 | 8.506 | 8.506 | 20s | 20s |
| qa_2_302_len32008.json | validation.jsonl (qa_2) | 32008 | CacheBlend | 0.00 | 1.00 | 0.70 | 8.882 | 8.882 | Diana Quick's birthday is 23 November 19 | Diana Quick's birthday is 23 November 19 |
| qa_2_303_len29376.json | validation.jsonl (qa_2) | 29376 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.202 | 8.202 | The film is "Third and Indiana", which i | The film is "Third and Indiana", which i |
| qa_2_304_len32365.json | validation.jsonl (qa_2) | 32365 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.398 | 8.398 | No | No |
| qa_2_305_len30792.json | validation.jsonl (qa_2) | 30792 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.948 | 7.948 | 1901 | 1901 |
| qa_2_306_len31171.json | validation.jsonl (qa_2) | 31171 | CacheBlend | 0.00 | 0.00 | 0.14 | 8.443 | 8.443 | Rivington Hall Barn is near Rivington, L | Rivington Hall Barn is near Rivington, L |
| qa_2_307_len31709.json | validation.jsonl (qa_2) | 31709 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.237 | 8.237 | George Cayley | George Cayley |
| qa_2_308_len31374.json | validation.jsonl (qa_2) | 31374 | CacheBlend | 0.00 | 1.00 | 0.50 | 8.069 | 8.069 | Leonberger | Leonberger |
| qa_2_309_len32283.json | validation.jsonl (qa_2) | 32283 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.385 | 8.385 | No | No |
| qa_2_310_len32595.json | validation.jsonl (qa_2) | 32595 | CacheBlend | 0.00 | 0.00 | 0.75 | 8.561 | 8.561 | 2015 | 2015 |
| qa_2_311_len31753.json | validation.jsonl (qa_2) | 31753 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.235 | 8.235 | Shonen Jump | Shonen Jump |
| qa_2_312_len32192.json | validation.jsonl (qa_2) | 32192 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.372 | 8.372 | Kellogg's | Kellogg's |
| qa_2_313_len32236.json | validation.jsonl (qa_2) | 32236 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.384 | 8.384 | Am Rong | Am Rong |
| qa_2_314_len31720.json | validation.jsonl (qa_2) | 31720 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.198 | 8.198 | 29 | 29 |
| qa_2_315_len31422.json | validation.jsonl (qa_2) | 31422 | CacheBlend | 0.00 | 1.00 | 0.38 | 8.356 | 8.356 | Jimmy Fallon was born in Bay Ridge, Broo | Jimmy Fallon was born in Bay Ridge, Broo |
| qa_2_316_len31797.json | validation.jsonl (qa_2) | 31797 | CacheBlend | 0.00 | 1.00 | 0.86 | 8.296 | 8.296 | Northumbrian dialect | Northumbrian dialect |
| qa_2_317_len31695.json | validation.jsonl (qa_2) | 31695 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.221 | 8.221 | Oldham County | Oldham County |
| qa_2_318_len32593.json | validation.jsonl (qa_2) | 32593 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.455 | 8.455 | No | No |
| qa_2_319_len30442.json | validation.jsonl (qa_2) | 30442 | CacheBlend | 0.00 | 1.00 | 0.06 | 8.742 | 8.742 | The Voice of Ireland had The Saturdays a | The Voice of Ireland had The Saturdays a |
| qa_2_320_len30695.json | validation.jsonl (qa_2) | 30695 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.865 | 7.865 | John Fogerty | John Fogerty |
| qa_2_321_len32063.json | validation.jsonl (qa_2) | 32063 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.448 | 8.448 | FIFA Women's World Cup | FIFA Women's World Cup |
| qa_2_322_len32197.json | validation.jsonl (qa_2) | 32197 | CacheBlend | 0.00 | 0.00 | 0.50 | 8.406 | 8.406 | 1987 | 1987 |
| qa_2_323_len32077.json | validation.jsonl (qa_2) | 32077 | CacheBlend | 0.00 | 1.00 | 0.75 | 8.402 | 8.402 | Wilmette, Illinois | Wilmette, Illinois |
| qa_2_324_len30779.json | validation.jsonl (qa_2) | 30779 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.977 | 7.977 | Oedipus Rex | Oedipus Rex |
| qa_2_325_len32041.json | validation.jsonl (qa_2) | 32041 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.267 | 9.267 | Katherine Waterston and Chrisann Brennan | Katherine Waterston and Chrisann Brennan |
| qa_2_326_len32297.json | validation.jsonl (qa_2) | 32297 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.454 | 8.454 | Grampa Simpson | Grampa Simpson |
| qa_2_327_len32637.json | validation.jsonl (qa_2) | 32637 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.527 | 8.527 | Graduation | Graduation |
| qa_2_328_len30871.json | validation.jsonl (qa_2) | 30871 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.943 | 7.943 | Yes | Yes |
| qa_2_329_len32392.json | validation.jsonl (qa_2) | 32392 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.526 | 8.526 | Kunming | Kunming |
| qa_2_330_len31842.json | validation.jsonl (qa_2) | 31842 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.265 | 9.265 | Alex McKenna, the singer of "B Boy", was | Alex McKenna, the singer of "B Boy", was |
| qa_2_331_len32686.json | validation.jsonl (qa_2) | 32686 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.578 | 8.578 | minor basilica | minor basilica |
| qa_2_332_len30753.json | validation.jsonl (qa_2) | 30753 | CacheBlend | 0.00 | 1.00 | 0.89 | 7.957 | 7.957 | 150 million | 150 million |
| qa_2_333_len32494.json | validation.jsonl (qa_2) | 32494 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.486 | 8.486 | Aqua | Aqua |
| qa_2_334_len32334.json | validation.jsonl (qa_2) | 32334 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.575 | 8.575 | Reynolds v. Sims | Reynolds v. Sims |
| qa_2_335_len32363.json | validation.jsonl (qa_2) | 32363 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.483 | 8.483 | The Good Wife | The Good Wife |
| qa_2_336_len32139.json | validation.jsonl (qa_2) | 32139 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.297 | 8.297 | Pakistan | Pakistan |
| qa_2_337_len30906.json | validation.jsonl (qa_2) | 30906 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.980 | 7.980 | The Four Seasons | The Four Seasons |
| qa_2_338_len30951.json | validation.jsonl (qa_2) | 30951 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.923 | 7.923 | No | No |
| qa_2_339_len31407.json | validation.jsonl (qa_2) | 31407 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.127 | 8.127 | 50 mi | 50 mi |
| qa_2_340_len30014.json | validation.jsonl (qa_2) | 30014 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.801 | 7.801 | Victor Rådström | Victor Rådström |
| qa_2_341_len32343.json | validation.jsonl (qa_2) | 32343 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.484 | 8.484 | Happy Death Day | Happy Death Day |
| qa_2_342_len30790.json | validation.jsonl (qa_2) | 30790 | CacheBlend | 0.00 | 0.00 | 0.23 | 8.869 | 8.869 | John Hume was not the president of Sinn  | John Hume was not the president of Sinn  |
| qa_2_343_len30744.json | validation.jsonl (qa_2) | 30744 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.863 | 8.863 | The Battle of Belleau Wood occurred earl | The Battle of Belleau Wood occurred earl |
| qa_2_344_len31287.json | validation.jsonl (qa_2) | 31287 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.092 | 8.092 | Royal Australian Navy | Royal Australian Navy |
| qa_2_345_len31815.json | validation.jsonl (qa_2) | 31815 | CacheBlend | 0.00 | 0.00 | 0.71 | 8.417 | 8.417 | 240,000 | 240,000 |
| qa_2_346_len30274.json | validation.jsonl (qa_2) | 30274 | CacheBlend | 0.00 | 0.00 | 0.73 | 7.770 | 7.770 | West Point Foundry | West Point Foundry |
| qa_2_347_len31704.json | validation.jsonl (qa_2) | 31704 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.215 | 8.215 | Jack Ryan | Jack Ryan |
| qa_2_348_len32642.json | validation.jsonl (qa_2) | 32642 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.481 | 9.481 | Wheat salad is not typically composed of | Wheat salad is not typically composed of |
| qa_2_349_len29993.json | validation.jsonl (qa_2) | 29993 | CacheBlend | 0.00 | 0.00 | 0.25 | 7.698 | 7.698 | 1998 | 1998 |
| qa_2_350_len32279.json | validation.jsonl (qa_2) | 32279 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.435 | 8.435 | Current Mood | Current Mood |
| qa_2_351_len28661.json | validation.jsonl (qa_2) | 28661 | CacheBlend | 0.00 | 0.00 | 0.73 | 7.204 | 7.204 | Edward Albee | Edward Albee |
| qa_2_352_len32240.json | validation.jsonl (qa_2) | 32240 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.330 | 8.330 | Yes | Yes |
| qa_2_353_len31532.json | validation.jsonl (qa_2) | 31532 | CacheBlend | 0.00 | 1.00 | 0.09 | 8.905 | 8.905 | Vanessa Bley is not an impresario. The d | Vanessa Bley is not an impresario. The d |
| qa_2_354_len31734.json | validation.jsonl (qa_2) | 31734 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.177 | 8.177 | No | No |
| qa_2_355_len32062.json | validation.jsonl (qa_2) | 32062 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.280 | 8.280 | No | No |
| qa_2_356_len31977.json | validation.jsonl (qa_2) | 31977 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.292 | 8.292 | Oxford | Oxford |
| qa_2_357_len31733.json | validation.jsonl (qa_2) | 31733 | CacheBlend | 0.00 | 1.00 | 0.33 | 8.329 | 8.329 | J. G. Farrell | J. G. Farrell |
| qa_2_358_len31384.json | validation.jsonl (qa_2) | 31384 | CacheBlend | 0.00 | 1.00 | 0.40 | 8.125 | 8.125 | New York City | New York City |
| qa_2_359_len30549.json | validation.jsonl (qa_2) | 30549 | CacheBlend | 0.00 | 1.00 | 0.67 | 7.906 | 7.906 | Phil Lord and Christopher Miller | Phil Lord and Christopher Miller |
| qa_2_360_len30657.json | validation.jsonl (qa_2) | 30657 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.896 | 7.896 | Buckingham Palace | Buckingham Palace |
| qa_2_361_len32689.json | validation.jsonl (qa_2) | 32689 | CacheBlend | 0.00 | 1.00 | 0.18 | 9.081 | 9.081 | Scott Sandelin grew up in Hibbing, Minne | Scott Sandelin grew up in Hibbing, Minne |
| qa_2_362_len30847.json | validation.jsonl (qa_2) | 30847 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.914 | 7.914 | Phil Collins | Phil Collins |
| qa_2_363_len30487.json | validation.jsonl (qa_2) | 30487 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.792 | 7.792 | video game | video game |
| qa_2_364_len31579.json | validation.jsonl (qa_2) | 31579 | CacheBlend | 0.00 | 1.00 | 0.33 | 8.407 | 8.407 | Edmund Sharpe graduated from Cambridge U | Edmund Sharpe graduated from Cambridge U |
| qa_2_365_len31586.json | validation.jsonl (qa_2) | 31586 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.174 | 8.174 | Love Streams | Love Streams |
| qa_2_366_len32034.json | validation.jsonl (qa_2) | 32034 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.359 | 8.359 | Lowe's | Lowe's |
| qa_2_367_len30901.json | validation.jsonl (qa_2) | 30901 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.908 | 7.908 | second | second |
| qa_2_368_len30200.json | validation.jsonl (qa_2) | 30200 | CacheBlend | 0.00 | 0.00 | 0.75 | 7.743 | 7.743 | 1988 | 1988 |
| qa_2_369_len30400.json | validation.jsonl (qa_2) | 30400 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.789 | 7.789 | James Packer | James Packer |
| qa_2_370_len31597.json | validation.jsonl (qa_2) | 31597 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.185 | 8.185 | documentary | documentary |
| qa_2_371_len32331.json | validation.jsonl (qa_2) | 32331 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.399 | 8.399 | England | England |
| qa_2_372_len32423.json | validation.jsonl (qa_2) | 32423 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.503 | 8.503 | Yunho | Yunho |
| qa_2_373_len30769.json | validation.jsonl (qa_2) | 30769 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.922 | 7.922 | Hot toddy | Hot toddy |
| qa_2_374_len32178.json | validation.jsonl (qa_2) | 32178 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.417 | 8.417 | 2017 | 2017 |
| qa_2_375_len32192.json | validation.jsonl (qa_2) | 32192 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.361 | 8.361 | Gay Byrne | Gay Byrne |
| qa_2_376_len31632.json | validation.jsonl (qa_2) | 31632 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.415 | 8.415 | Hudson Bay Mining and Smelting Company | Hudson Bay Mining and Smelting Company |
| qa_2_377_len31459.json | validation.jsonl (qa_2) | 31459 | CacheBlend | 0.00 | 0.00 | 0.10 | 8.682 | 8.682 | Jens Risom introduced Mid-Century modern | Jens Risom introduced Mid-Century modern |
| qa_2_378_len30182.json | validation.jsonl (qa_2) | 30182 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.676 | 7.676 | Constantine | Constantine |
| qa_2_379_len32018.json | validation.jsonl (qa_2) | 32018 | CacheBlend | 0.00 | 0.00 | 0.17 | 9.249 | 9.249 | The Houston Astrodome is often referred  | The Houston Astrodome is often referred  |
| qa_2_380_len30800.json | validation.jsonl (qa_2) | 30800 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.256 | 8.256 | José Echegaray y Eizaguirre | José Echegaray y Eizaguirre |
| qa_2_381_len32400.json | validation.jsonl (qa_2) | 32400 | CacheBlend | 0.00 | 1.00 | 0.20 | 9.363 | 9.363 | Doncaster Belles played against Knowsley | Doncaster Belles played against Knowsley |
| qa_2_382_len31593.json | validation.jsonl (qa_2) | 31593 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.189 | 8.189 | Atomic Kitten | Atomic Kitten |
| qa_2_383_len32156.json | validation.jsonl (qa_2) | 32156 | CacheBlend | 0.00 | 0.00 | 0.31 | 8.511 | 8.511 | Elmwood Historic District–East | Elmwood Historic District–East |
| qa_2_384_len31426.json | validation.jsonl (qa_2) | 31426 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.143 | 8.143 | Philip K. Dick | Philip K. Dick |
| qa_2_385_len30290.json | validation.jsonl (qa_2) | 30290 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.719 | 7.719 | New York Giants | New York Giants |
| qa_2_386_len32496.json | validation.jsonl (qa_2) | 32496 | CacheBlend | 0.00 | 0.00 | 0.75 | 8.534 | 8.534 | 1960 | 1960 |
| qa_2_387_len30670.json | validation.jsonl (qa_2) | 30670 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.815 | 7.815 | Copenhagen | Copenhagen |
| qa_2_388_len29211.json | validation.jsonl (qa_2) | 29211 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.323 | 7.323 | No | No |
| qa_2_389_len32586.json | validation.jsonl (qa_2) | 32586 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.556 | 8.556 | Soha Ali Khan | Soha Ali Khan |
| qa_2_390_len31363.json | validation.jsonl (qa_2) | 31363 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.051 | 8.051 | horror | horror |
| qa_2_391_len32544.json | validation.jsonl (qa_2) | 32544 | CacheBlend | 0.00 | 0.00 | 0.83 | 8.591 | 8.591 | Lucas Grabeel | Lucas Grabeel |
| qa_2_392_len32501.json | validation.jsonl (qa_2) | 32501 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.468 | 8.468 | Rings | Rings |
| qa_2_393_len32624.json | validation.jsonl (qa_2) | 32624 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.468 | 8.468 | Kent | Kent |
| qa_2_394_len30477.json | validation.jsonl (qa_2) | 30477 | CacheBlend | 0.00 | 0.00 | 0.08 | 8.481 | 8.481 | Sir Peter Reginald Frederick Hall CBE wr | Sir Peter Reginald Frederick Hall CBE wr |
| qa_2_395_len32099.json | validation.jsonl (qa_2) | 32099 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.311 | 8.311 | French Open | French Open |
| qa_2_396_len32258.json | validation.jsonl (qa_2) | 32258 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.415 | 8.415 | Bismarck | Bismarck |
| qa_2_397_len32712.json | validation.jsonl (qa_2) | 32712 | CacheBlend | 0.00 | 1.00 | 0.25 | 8.526 | 8.526 | Manchester United | Manchester United |
| qa_2_398_len32012.json | validation.jsonl (qa_2) | 32012 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.953 | 8.953 | Snowdrop is a proprietary game engine fi | Snowdrop is a proprietary game engine fi |
| qa_2_399_len31813.json | validation.jsonl (qa_2) | 31813 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.270 | 8.270 | YouTube Rewind | YouTube Rewind |
| qa_2_400_len30106.json | validation.jsonl (qa_2) | 30106 | CacheBlend | 0.00 | 1.00 | 0.62 | 7.871 | 7.871 | King George IV and the Duke of Wellingto | King George IV and the Duke of Wellingto |
| qa_2_401_len30823.json | validation.jsonl (qa_2) | 30823 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.890 | 7.890 | Icehouse pieces | Icehouse pieces |
| qa_2_402_len30339.json | validation.jsonl (qa_2) | 30339 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.681 | 7.681 | ZZ Top | ZZ Top |
| qa_2_403_len31037.json | validation.jsonl (qa_2) | 31037 | CacheBlend | 0.00 | 0.00 | 0.75 | 8.047 | 8.047 | 1973 | 1973 |
| qa_2_404_len30171.json | validation.jsonl (qa_2) | 30171 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.245 | 8.245 | The Jeep Cherokee was originally sold as | The Jeep Cherokee was originally sold as |
| qa_2_405_len32716.json | validation.jsonl (qa_2) | 32716 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.640 | 8.640 | Leah Crocetto | Leah Crocetto |
| qa_2_406_len32458.json | validation.jsonl (qa_2) | 32458 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.433 | 8.433 | Gallery | Gallery |
| qa_2_407_len31436.json | validation.jsonl (qa_2) | 31436 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.107 | 8.107 | Langston Hughes | Langston Hughes |
| qa_2_408_len31901.json | validation.jsonl (qa_2) | 31901 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.123 | 9.123 | KGOT's studios are located in the Dimond | KGOT's studios are located in the Dimond |
| qa_2_409_len31811.json | validation.jsonl (qa_2) | 31811 | CacheBlend | 0.00 | 0.00 | 0.14 | 8.294 | 8.294 | Bad Meets Evil | Bad Meets Evil |
| qa_2_410_len29792.json | validation.jsonl (qa_2) | 29792 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.540 | 7.540 | Shanghai | Shanghai |
| qa_2_411_len31966.json | validation.jsonl (qa_2) | 31966 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.095 | 9.095 | Jacques Mathou starred in Delicatessen,  | Jacques Mathou starred in Delicatessen,  |
| qa_2_412_len32305.json | validation.jsonl (qa_2) | 32305 | CacheBlend | 0.00 | 0.00 | 0.67 | 8.554 | 8.554 | Lithuanian-French | Lithuanian-French |
| qa_2_413_len32251.json | validation.jsonl (qa_2) | 32251 | CacheBlend | 0.00 | 1.00 | 0.33 | 8.486 | 8.486 | the death of a family member | the death of a family member |
| qa_2_414_len31315.json | validation.jsonl (qa_2) | 31315 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.073 | 8.073 | Douglas Jackson | Douglas Jackson |
| qa_2_415_len32130.json | validation.jsonl (qa_2) | 32130 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.345 | 8.345 | Dirty Pretty Things | Dirty Pretty Things |
| qa_2_416_len31249.json | validation.jsonl (qa_2) | 31249 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.113 | 8.113 | Krusty the Clown | Krusty the Clown |
| qa_2_417_len30869.json | validation.jsonl (qa_2) | 30869 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.012 | 8.012 | Wilton Mall | Wilton Mall |
| qa_2_418_len31699.json | validation.jsonl (qa_2) | 31699 | CacheBlend | 0.00 | 1.00 | 0.57 | 8.305 | 8.305 | Chauncey Billups | Chauncey Billups |
| qa_2_419_len30680.json | validation.jsonl (qa_2) | 30680 | CacheBlend | 0.00 | 0.00 | 0.13 | 8.798 | 8.798 | The information provided does not includ | The information provided does not includ |
| qa_2_420_len31897.json | validation.jsonl (qa_2) | 31897 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.246 | 8.246 | No | No |
| qa_2_421_len32081.json | validation.jsonl (qa_2) | 32081 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.313 | 8.313 | documentary | documentary |
| qa_2_422_len30281.json | validation.jsonl (qa_2) | 30281 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.699 | 7.699 | 12 | 12 |
| qa_2_423_len31789.json | validation.jsonl (qa_2) | 31789 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.274 | 8.274 | defensive coordinator | defensive coordinator |
| qa_2_424_len31814.json | validation.jsonl (qa_2) | 31814 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.343 | 8.343 | 1.95 m | 1.95 m |
| qa_2_425_len32263.json | validation.jsonl (qa_2) | 32263 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.431 | 8.431 | MGM Resorts International | MGM Resorts International |
| qa_2_426_len32571.json | validation.jsonl (qa_2) | 32571 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.673 | 8.673 | Bedknobs and Broomsticks | Bedknobs and Broomsticks |
| qa_2_427_len30073.json | validation.jsonl (qa_2) | 30073 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.706 | 7.706 | The Holston River | The Holston River |
| qa_2_428_len31366.json | validation.jsonl (qa_2) | 31366 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.101 | 8.101 | Conservative Party | Conservative Party |
| qa_2_429_len30809.json | validation.jsonl (qa_2) | 30809 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.948 | 7.948 | Wes Archer | Wes Archer |
| qa_2_430_len30063.json | validation.jsonl (qa_2) | 30063 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.707 | 7.707 | Geelong Football Club | Geelong Football Club |
| qa_2_431_len31766.json | validation.jsonl (qa_2) | 31766 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.206 | 8.206 | Yes | Yes |
| qa_2_432_len31899.json | validation.jsonl (qa_2) | 31899 | CacheBlend | 0.00 | 0.00 | 0.80 | 8.348 | 8.348 | Delmer Daves | Delmer Daves |
| qa_2_433_len30921.json | validation.jsonl (qa_2) | 30921 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.010 | 8.010 | Test Icicles | Test Icicles |
| qa_2_434_len31739.json | validation.jsonl (qa_2) | 31739 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.279 | 8.279 | Walt Disney Productions | Walt Disney Productions |
| qa_2_435_len31935.json | validation.jsonl (qa_2) | 31935 | CacheBlend | 0.00 | 0.00 | 0.07 | 9.234 | 9.234 | The Jemez River, which is 7.51 km long,  | The Jemez River, which is 7.51 km long,  |
| qa_2_436_len29289.json | validation.jsonl (qa_2) | 29289 | CacheBlend | 0.00 | 0.00 | 0.31 | 7.609 | 7.609 | Olathe, Kansas, United States | Olathe, Kansas, United States |
| qa_2_437_len31634.json | validation.jsonl (qa_2) | 31634 | CacheBlend | 0.00 | 1.00 | 0.07 | 9.127 | 9.127 | The question asks for the boxer ranked s | The question asks for the boxer ranked s |
| qa_2_438_len31155.json | validation.jsonl (qa_2) | 31155 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.083 | 8.083 | 1909 | 1909 |
| qa_2_439_len31826.json | validation.jsonl (qa_2) | 31826 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.310 | 8.310 | Sun Valley, Idaho | Sun Valley, Idaho |
| qa_2_440_len30894.json | validation.jsonl (qa_2) | 30894 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.043 | 8.043 | President of the United States | President of the United States |
| qa_2_441_len32504.json | validation.jsonl (qa_2) | 32504 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.552 | 8.552 | The Bears and I | The Bears and I |
| qa_2_442_len32307.json | validation.jsonl (qa_2) | 32307 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.502 | 8.502 | Patterns of Sexual Behavior | Patterns of Sexual Behavior |
| qa_2_443_len30994.json | validation.jsonl (qa_2) | 30994 | CacheBlend | 0.00 | 0.00 | 0.80 | 7.955 | 7.955 | Larry Drake | Larry Drake |
| qa_2_444_len31171.json | validation.jsonl (qa_2) | 31171 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.187 | 8.187 | Bedknobs and Broomsticks | Bedknobs and Broomsticks |
| qa_2_445_len32294.json | validation.jsonl (qa_2) | 32294 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.807 | 8.807 | 46 Albert Square, Walford, London E20 | 46 Albert Square, Walford, London E20 |
| qa_2_446_len30741.json | validation.jsonl (qa_2) | 30741 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.920 | 7.920 | Betty Suarez | Betty Suarez |
| qa_2_447_len30964.json | validation.jsonl (qa_2) | 30964 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.952 | 7.952 | Phoenix Television | Phoenix Television |
| qa_2_448_len32129.json | validation.jsonl (qa_2) | 32129 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.326 | 8.326 | Planet Terror | Planet Terror |
| qa_2_449_len31097.json | validation.jsonl (qa_2) | 31097 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.929 | 8.929 | Chimaji Appa served as the younger broth | Chimaji Appa served as the younger broth |
| qa_2_450_len32073.json | validation.jsonl (qa_2) | 32073 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.381 | 8.381 | Donald Trump Jr. | Donald Trump Jr. |
| qa_2_451_len31098.json | validation.jsonl (qa_2) | 31098 | CacheBlend | 0.00 | 1.00 | 0.44 | 8.020 | 8.020 | Lord Dunmore | Lord Dunmore |
| qa_2_452_len31288.json | validation.jsonl (qa_2) | 31288 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.087 | 8.087 | Graduados | Graduados |
| qa_2_453_len31527.json | validation.jsonl (qa_2) | 31527 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.231 | 8.231 | Private Secretary and Treasurer | Private Secretary and Treasurer |
| qa_2_454_len32323.json | validation.jsonl (qa_2) | 32323 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.399 | 8.399 | No | No |
| qa_2_455_len32629.json | validation.jsonl (qa_2) | 32629 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.659 | 8.659 | Ogallala Aquifer | Ogallala Aquifer |
| qa_2_456_len30405.json | validation.jsonl (qa_2) | 30405 | CacheBlend | 0.00 | 0.00 | 0.40 | 7.823 | 7.823 | Ibn Arabi | Ibn Arabi |
| qa_2_457_len30636.json | validation.jsonl (qa_2) | 30636 | CacheBlend | 0.00 | 0.00 | 0.06 | 8.786 | 8.786 | The goddess associated with Frigg, possi | The goddess associated with Frigg, possi |
| qa_2_458_len30336.json | validation.jsonl (qa_2) | 30336 | CacheBlend | 0.00 | 1.00 | 0.40 | 7.744 | 7.744 | South Asian Games | South Asian Games |
| qa_2_459_len32590.json | validation.jsonl (qa_2) | 32590 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.573 | 8.573 | Viscount Cranborne | Viscount Cranborne |
| qa_2_460_len31300.json | validation.jsonl (qa_2) | 31300 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.122 | 8.122 | Kansas City crime family | Kansas City crime family |
| qa_2_461_len32396.json | validation.jsonl (qa_2) | 32396 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.444 | 8.444 | Albert Park | Albert Park |
| qa_2_462_len32626.json | validation.jsonl (qa_2) | 32626 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.585 | 8.585 | 2003 | 2003 |
| qa_2_463_len32603.json | validation.jsonl (qa_2) | 32603 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.553 | 8.553 | Unbreakable | Unbreakable |
| qa_2_464_len32563.json | validation.jsonl (qa_2) | 32563 | CacheBlend | 0.00 | 1.00 | 0.20 | 9.288 | 9.288 | Raymond Ochoa voices Arlo, the Apatosaur | Raymond Ochoa voices Arlo, the Apatosaur |
| qa_2_465_len32400.json | validation.jsonl (qa_2) | 32400 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.681 | 8.681 | COPS (Central Organization of Police Spe | COPS (Central Organization of Police Spe |
| qa_2_466_len32409.json | validation.jsonl (qa_2) | 32409 | CacheBlend | 0.00 | 1.00 | 0.00 | 9.417 | 9.417 | Gaja in Sanskrit means Elephant. 'Brisht | Gaja in Sanskrit means Elephant. 'Brisht |
| qa_2_467_len31961.json | validation.jsonl (qa_2) | 31961 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.324 | 8.324 | Battle of Guam | Battle of Guam |
| qa_2_468_len32109.json | validation.jsonl (qa_2) | 32109 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.393 | 8.393 | Catherine Cyran | Catherine Cyran |
| qa_2_469_len31172.json | validation.jsonl (qa_2) | 31172 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.016 | 8.016 | Spin Out | Spin Out |
| qa_2_470_len31027.json | validation.jsonl (qa_2) | 31027 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.938 | 7.938 | film | film |
| qa_2_471_len31395.json | validation.jsonl (qa_2) | 31395 | CacheBlend | 0.00 | 1.00 | 0.75 | 8.126 | 8.126 | executive search | executive search |
| qa_2_472_len31297.json | validation.jsonl (qa_2) | 31297 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.020 | 8.020 | author | author |
| qa_2_473_len32728.json | validation.jsonl (qa_2) | 32728 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.656 | 8.656 | Catholic Church in Ireland | Catholic Church in Ireland |
| qa_2_474_len30988.json | validation.jsonl (qa_2) | 30988 | CacheBlend | 0.00 | 1.00 | 0.32 | 8.568 | 8.568 | Bridge to Terabithia (1985 film) was sho | Bridge to Terabithia (1985 film) was sho |
| qa_2_475_len32191.json | validation.jsonl (qa_2) | 32191 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.350 | 8.350 | Joe Scarborough | Joe Scarborough |
| qa_2_476_len32476.json | validation.jsonl (qa_2) | 32476 | CacheBlend | 0.00 | 1.00 | 0.73 | 8.555 | 8.555 | NCAA Division I | NCAA Division I |
| qa_2_477_len31917.json | validation.jsonl (qa_2) | 31917 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.347 | 8.347 | Tea Moderna | Tea Moderna |
| qa_2_478_len31602.json | validation.jsonl (qa_2) | 31602 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.172 | 8.172 | United States | United States |
| qa_2_479_len31352.json | validation.jsonl (qa_2) | 31352 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.166 | 8.166 | Kathleen O'Brien | Kathleen O'Brien |
| qa_2_480_len30469.json | validation.jsonl (qa_2) | 30469 | CacheBlend | 0.00 | 1.00 | 0.89 | 7.864 | 7.864 | Warner Bros. Records | Warner Bros. Records |
| qa_2_481_len32538.json | validation.jsonl (qa_2) | 32538 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.538 | 8.538 | Darkthrone | Darkthrone |
| qa_2_482_len32153.json | validation.jsonl (qa_2) | 32153 | CacheBlend | 0.00 | 0.00 | 0.37 | 9.303 | 9.303 | Emilie du Châtelet translated and commen | Emilie du Châtelet translated and commen |
| qa_2_483_len32469.json | validation.jsonl (qa_2) | 32469 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.447 | 8.447 | No | No |
| qa_2_484_len30121.json | validation.jsonl (qa_2) | 30121 | CacheBlend | 0.00 | 1.00 | 0.19 | 8.611 | 8.611 | The Detroit mayoral elections of 2009 to | The Detroit mayoral elections of 2009 to |
| qa_2_485_len31805.json | validation.jsonl (qa_2) | 31805 | CacheBlend | 0.00 | 0.00 | 0.50 | 8.306 | 8.306 | 2013 | 2013 |
| qa_2_486_len29655.json | validation.jsonl (qa_2) | 29655 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.478 | 7.478 | rock | rock |
| qa_2_487_len28811.json | validation.jsonl (qa_2) | 28811 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.240 | 7.240 | Portal 2 | Portal 2 |
| qa_2_488_len32737.json | validation.jsonl (qa_2) | 32737 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.523 | 8.523 | Aldosterone | Aldosterone |
| qa_2_489_len30469.json | validation.jsonl (qa_2) | 30469 | CacheBlend | 0.00 | 1.00 | 0.50 | 7.896 | 7.896 | 33 or 34 | 33 or 34 |
| qa_2_490_len32711.json | validation.jsonl (qa_2) | 32711 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.744 | 8.744 | Berghuis v. Thompkins | Berghuis v. Thompkins |
| qa_2_491_len32494.json | validation.jsonl (qa_2) | 32494 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.518 | 8.518 | Stacey Kent | Stacey Kent |
| qa_2_492_len31992.json | validation.jsonl (qa_2) | 31992 | CacheBlend | 0.00 | 0.00 | 0.75 | 8.359 | 8.359 | 1969 | 1969 |
| qa_2_493_len32699.json | validation.jsonl (qa_2) | 32699 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.609 | 8.609 | IndyCar Series | IndyCar Series |
| qa_2_494_len31659.json | validation.jsonl (qa_2) | 31659 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.268 | 8.268 | 2002 | 2002 |
| qa_2_495_len32069.json | validation.jsonl (qa_2) | 32069 | CacheBlend | 0.00 | 0.00 | 0.33 | 8.313 | 8.313 | 24 | 24 |
| qa_2_496_len32405.json | validation.jsonl (qa_2) | 32405 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.471 | 8.471 | Newark | Newark |
| qa_2_497_len31175.json | validation.jsonl (qa_2) | 31175 | CacheBlend | 0.00 | 1.00 | 0.26 | 8.973 | 8.973 | The 2017 US Open was the 137th edition o | The 2017 US Open was the 137th edition o |
| qa_2_498_len30886.json | validation.jsonl (qa_2) | 30886 | CacheBlend | 0.00 | 0.00 | 0.80 | 7.974 | 7.974 | Ronald Reagan | Ronald Reagan |
| qa_2_499_len31501.json | validation.jsonl (qa_2) | 31501 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.103 | 8.103 | documentary | documentary |

## 二、按方法聚合（每轮一行，再给出跨轮中位数）

| 方法 | 轮次 | EM | SM | F1 | TTFT缓存均值(s) | TTFT全量均值(s) | 缓存加速比 |
|---|---|---|---|---|---|------|---|
| CacheBlend | r1 | 0.400 | 0.580 | 0.553 | 8.246 | 8.246 | 1.00× |
| CacheBlend | r2 | 0.350 | 0.520 | 0.494 | 8.382 | 8.382 | 1.00× |
| CacheBlend | r3 | 0.330 | 0.510 | 0.502 | 8.341 | 8.341 | 1.00× |
| CacheBlend | r4 | 0.360 | 0.560 | 0.509 | 8.325 | 8.325 | 1.00× |
| CacheBlend | r5 | 0.440 | 0.600 | 0.574 | 8.314 | 8.314 | 1.00× |
| **CacheBlend** | **中位** | **0.360** | **0.560** | **0.509** | **8.325** | **8.325** | **1.00×** |
| vLLM全量 | r1 | 0.400 | 0.580 | 0.553 | - | 8.246 | 1.00× |
| vLLM全量 | r2 | 0.350 | 0.520 | 0.494 | - | 8.382 | 1.00× |
| vLLM全量 | r3 | 0.330 | 0.510 | 0.502 | - | 8.341 | 1.00× |
| vLLM全量 | r4 | 0.360 | 0.560 | 0.509 | - | 8.325 | 1.00× |
| vLLM全量 | r5 | 0.440 | 0.600 | 0.574 | - | 8.314 | 1.00× |
| **vLLM全量** | **中位** | **0.360** | **0.560** | **0.509** | - | **8.325** | **1.00×** |

## 三、按数据集×方法聚合（跨轮中位数；QA 数据集看 SM/F1，samsum 摘要看 F1）

| 数据集 | 样本数 | 方法 | EM中位 | SM中位 | F1中位 | TTFT缓存中位(s) |
|---|---|---|---|---|---|---|
| validation.jsonl (qa_2) | 500 | CacheBlend | 0.000 | 1.000 | 0.571 | 8.327 |
