# 长上下文实验评分报告

- 样本目录：`experiments/inputs_qwen_ruler30k_500`（1000 个样本）
- 评分：EM（精确匹配，多答案取最大）+ SM（子串匹配）+ token-F1（多答案取最大）
- 归一化：小写/去冠词/去标点/Yes-No 规整（`example/utils.py`）
- 输入日志：`run_newvllm_mt128_qa2_c00.log`, `run_newvllm_mt128_qa2_c01.log`, `run_newvllm_mt128_qa2_c02.log`, `run_newvllm_mt128_qa2_c03.log`, `run_newvllm_mt128_qa2_c04.log`

## 一、逐样本明细（质量指标为多轮均值，TTFT 为多轮中位数）

| 样本 | 来源 | 长度 | 方法 | EM | SM | F1 | TTFT缓存(s) | TTFT全量(s) | 缓存答案 | 全量答案 |
|---|---|---|---|---|---|---|---|----|---|---|
| qa_2_000_len31234.json | validation.jsonl (qa_2) | 31234 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.025 | 8.025 | No | No |
| qa_2_001_len31871.json | validation.jsonl (qa_2) | 31871 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.307 | 8.307 | Shirley Temple | Shirley Temple |
| qa_2_002_len30167.json | validation.jsonl (qa_2) | 30167 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.830 | 7.830 | The Hork-Bajir Chronicles | The Hork-Bajir Chronicles |
| qa_2_003_len31814.json | validation.jsonl (qa_2) | 31814 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.189 | 8.189 | No | No |
| qa_2_004_len32050.json | validation.jsonl (qa_2) | 32050 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.334 | 8.334 | Greenwich Village | Greenwich Village |
| qa_2_005_len32340.json | validation.jsonl (qa_2) | 32340 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.446 | 8.446 | YG Entertainment | YG Entertainment |
| qa_2_006_len31806.json | validation.jsonl (qa_2) | 31806 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.347 | 8.347 | Eenasul Fateh | Eenasul Fateh |
| qa_2_007_len31346.json | validation.jsonl (qa_2) | 31346 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.171 | 8.171 | 1,400 | 1,400 |
| qa_2_008_len32138.json | validation.jsonl (qa_2) | 32138 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.368 | 8.368 | Annie Morton | Annie Morton |
| qa_2_009_len31131.json | validation.jsonl (qa_2) | 31131 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.993 | 7.993 | Yes | Yes |
| qa_2_010_len31807.json | validation.jsonl (qa_2) | 31807 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.255 | 8.255 | Kansas Song | Kansas Song |
| qa_2_011_len31916.json | validation.jsonl (qa_2) | 31916 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.321 | 8.321 | David Weissman | David Weissman |
| qa_2_012_len31839.json | validation.jsonl (qa_2) | 31839 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.327 | 8.327 | 1999 | 1999 |
| qa_2_013_len32311.json | validation.jsonl (qa_2) | 32311 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.410 | 8.410 | Yes | Yes |
| qa_2_014_len32118.json | validation.jsonl (qa_2) | 32118 | CacheBlend | 0.00 | 0.00 | 0.40 | 9.036 | 9.036 | 1903–1912 and then again from 1912–1924 | 1903–1912 and then again from 1912–1924 |
| qa_2_015_len32037.json | validation.jsonl (qa_2) | 32037 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.851 | 8.851 | The document does not provide the popula | The document does not provide the popula |
| qa_2_016_len32351.json | validation.jsonl (qa_2) | 32351 | CacheBlend | 0.00 | 0.00 | 0.25 | 8.488 | 8.488 | America East Conference | America East Conference |
| qa_2_017_len31167.json | validation.jsonl (qa_2) | 31167 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.992 | 7.992 | Yes | Yes |
| qa_2_018_len31835.json | validation.jsonl (qa_2) | 31835 | CacheBlend | 0.00 | 0.00 | 0.89 | 8.501 | 8.501 | 1969-1974 | 1969-1974 |
| qa_2_019_len30786.json | validation.jsonl (qa_2) | 30786 | CacheBlend | 0.00 | 1.00 | 0.88 | 8.030 | 8.030 | Robert Erskine Childers | Robert Erskine Childers |
| qa_2_020_len30327.json | validation.jsonl (qa_2) | 30327 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.819 | 7.819 | Sergio Pérez | Sergio Pérez |
| qa_2_021_len32159.json | validation.jsonl (qa_2) | 32159 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.375 | 8.375 | Pooh | Pooh |
| qa_2_022_len31724.json | validation.jsonl (qa_2) | 31724 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.223 | 8.223 | Apple TV | Apple TV |
| qa_2_023_len31158.json | validation.jsonl (qa_2) | 31158 | CacheBlend | 0.00 | 1.00 | 0.12 | 9.244 | 9.244 | Based on the information provided, Badly | Based on the information provided, Badly |
| qa_2_024_len30318.json | validation.jsonl (qa_2) | 30318 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.822 | 7.822 | World's Best Goalkeeper | World's Best Goalkeeper |
| qa_2_025_len30613.json | validation.jsonl (qa_2) | 30613 | CacheBlend | 0.00 | 0.00 | 0.23 | 8.548 | 8.548 | Lee Hazlewood died in 2007 and wrote "Th | Lee Hazlewood died in 2007 and wrote "Th |
| qa_2_026_len32716.json | validation.jsonl (qa_2) | 32716 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.619 | 8.619 | 1838 | 1838 |
| qa_2_027_len32343.json | validation.jsonl (qa_2) | 32343 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.433 | 8.433 | yes | yes |
| qa_2_028_len30034.json | validation.jsonl (qa_2) | 30034 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.703 | 7.703 | Henry J. Kaiser | Henry J. Kaiser |
| qa_2_029_len29006.json | validation.jsonl (qa_2) | 29006 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.392 | 7.392 | Arena of Khazan | Arena of Khazan |
| qa_2_030_len32408.json | validation.jsonl (qa_2) | 32408 | CacheBlend | 0.00 | 1.00 | 0.57 | 8.745 | 8.745 | March 14, 2000 | March 14, 2000 |
| qa_2_031_len31794.json | validation.jsonl (qa_2) | 31794 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.485 | 8.485 | Buck-Tick hails from Japan. | Buck-Tick hails from Japan. |
| qa_2_032_len31191.json | validation.jsonl (qa_2) | 31191 | CacheBlend | 0.00 | 0.00 | 0.29 | 8.232 | 8.232 | Charles Nungesser and François Coli | Charles Nungesser and François Coli |
| qa_2_033_len31857.json | validation.jsonl (qa_2) | 31857 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.232 | 8.232 | Yes | Yes |
| qa_2_034_len30981.json | validation.jsonl (qa_2) | 30981 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.032 | 8.032 | Letters to Cleo | Letters to Cleo |
| qa_2_035_len31102.json | validation.jsonl (qa_2) | 31102 | CacheBlend | 0.00 | 0.00 | 0.55 | 8.061 | 8.061 | 1924 | 1924 |
| qa_2_036_len32115.json | validation.jsonl (qa_2) | 32115 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.405 | 8.405 | 2000 | 2000 |
| qa_2_037_len31683.json | validation.jsonl (qa_2) | 31683 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.291 | 8.291 | World War II | World War II |
| qa_2_038_len32687.json | validation.jsonl (qa_2) | 32687 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.570 | 8.570 | No | No |
| qa_2_039_len32205.json | validation.jsonl (qa_2) | 32205 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.447 | 8.447 | Ithaca | Ithaca |
| qa_2_040_len29707.json | validation.jsonl (qa_2) | 29707 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.640 | 7.640 | Scotch Collie | Scotch Collie |
| qa_2_041_len30563.json | validation.jsonl (qa_2) | 30563 | CacheBlend | 0.00 | 1.00 | 0.05 | 9.101 | 9.101 | Tata Consultancy Services is headquarter | Tata Consultancy Services is headquarter |
| qa_2_042_len32525.json | validation.jsonl (qa_2) | 32525 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.591 | 8.591 | 1962 | 1962 |
| qa_2_043_len30954.json | validation.jsonl (qa_2) | 30954 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.699 | 8.699 | The battle in which Giuseppe Arimondi lo | The battle in which Giuseppe Arimondi lo |
| qa_2_044_len31499.json | validation.jsonl (qa_2) | 31499 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.180 | 8.180 | Nelson Rockefeller | Nelson Rockefeller |
| qa_2_045_len32664.json | validation.jsonl (qa_2) | 32664 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.064 | 9.064 | Dirleton Castle borders on the south sid | Dirleton Castle borders on the south sid |
| qa_2_046_len31525.json | validation.jsonl (qa_2) | 31525 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.240 | 8.240 | Phil Spector | Phil Spector |
| qa_2_047_len31157.json | validation.jsonl (qa_2) | 31157 | CacheBlend | 0.00 | 1.00 | 0.50 | 8.764 | 8.764 | a forum in which the Nobel Peace Laureat | a forum in which the Nobel Peace Laureat |
| qa_2_048_len30718.json | validation.jsonl (qa_2) | 30718 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.852 | 7.852 | Yes | Yes |
| qa_2_049_len32426.json | validation.jsonl (qa_2) | 32426 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.553 | 8.553 | English Electric Canberra | English Electric Canberra |
| qa_2_050_len30583.json | validation.jsonl (qa_2) | 30583 | CacheBlend | 0.00 | 1.00 | 0.90 | 8.147 | 8.147 | 2009 Big 12 Conference football season | 2009 Big 12 Conference football season |
| qa_2_051_len29134.json | validation.jsonl (qa_2) | 29134 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.394 | 7.394 | 539 | 539 |
| qa_2_052_len32148.json | validation.jsonl (qa_2) | 32148 | CacheBlend | 0.00 | 0.00 | 0.20 | 8.445 | 8.445 | Darlington Raceway | Darlington Raceway |
| qa_2_053_len30874.json | validation.jsonl (qa_2) | 30874 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.019 | 8.019 | New York City | New York City |
| qa_2_054_len31684.json | validation.jsonl (qa_2) | 31684 | CacheBlend | 0.00 | 0.00 | 0.97 | 8.674 | 8.674 | Max Martin, Savan Kotecha, Ilya Salmanza | Max Martin, Savan Kotecha, Ilya Salmanza |
| qa_2_055_len32186.json | validation.jsonl (qa_2) | 32186 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.397 | 8.397 | Princeton | Princeton |
| qa_2_056_len32706.json | validation.jsonl (qa_2) | 32706 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.591 | 8.591 | Drifting | Drifting |
| qa_2_057_len31681.json | validation.jsonl (qa_2) | 31681 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.314 | 8.314 | Keith Bostic | Keith Bostic |
| qa_2_058_len32641.json | validation.jsonl (qa_2) | 32641 | CacheBlend | 0.00 | 0.00 | 0.44 | 8.673 | 8.673 | 1,146 | 1,146 |
| qa_2_059_len31590.json | validation.jsonl (qa_2) | 31590 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.186 | 8.186 | No | No |
| qa_2_060_len31574.json | validation.jsonl (qa_2) | 31574 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.221 | 8.221 | Larry Johnson | Larry Johnson |
| qa_2_061_len30328.json | validation.jsonl (qa_2) | 30328 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.854 | 7.854 | Ronald Shusett | Ronald Shusett |
| qa_2_062_len32496.json | validation.jsonl (qa_2) | 32496 | CacheBlend | 0.00 | 1.00 | 0.55 | 8.562 | 8.562 | Virginia Woolf | Virginia Woolf |
| qa_2_063_len30735.json | validation.jsonl (qa_2) | 30735 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.925 | 7.925 | 821 | 821 |
| qa_2_064_len31090.json | validation.jsonl (qa_2) | 31090 | CacheBlend | 0.00 | 1.00 | 0.56 | 8.398 | 8.398 | ExxonMobil operates in more than 70 coun | ExxonMobil operates in more than 70 coun |
| qa_2_065_len30489.json | validation.jsonl (qa_2) | 30489 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.834 | 7.834 | Charmed | Charmed |
| qa_2_066_len31221.json | validation.jsonl (qa_2) | 31221 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.173 | 8.173 | International Boxing Hall of Fame | International Boxing Hall of Fame |
| qa_2_067_len31901.json | validation.jsonl (qa_2) | 31901 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.328 | 8.328 | Usher | Usher |
| qa_2_068_len29747.json | validation.jsonl (qa_2) | 29747 | CacheBlend | 0.00 | 0.00 | 0.22 | 7.652 | 7.652 | Brian Doyle-Murray | Brian Doyle-Murray |
| qa_2_069_len31334.json | validation.jsonl (qa_2) | 31334 | CacheBlend | 0.00 | 0.00 | 0.06 | 9.368 | 9.368 | The 2017–18 Wigan Athletic F.C. season w | The 2017–18 Wigan Athletic F.C. season w |
| qa_2_070_len30409.json | validation.jsonl (qa_2) | 30409 | CacheBlend | 0.00 | 1.00 | 0.86 | 7.762 | 7.762 | Teen Titans | Teen Titans |
| qa_2_071_len30940.json | validation.jsonl (qa_2) | 30940 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.961 | 7.961 | Strasbourg | Strasbourg |
| qa_2_072_len32001.json | validation.jsonl (qa_2) | 32001 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.268 | 8.268 | orange | orange |
| qa_2_073_len32695.json | validation.jsonl (qa_2) | 32695 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.653 | 8.653 | Tromeo and Juliet | Tromeo and Juliet |
| qa_2_074_len28897.json | validation.jsonl (qa_2) | 28897 | CacheBlend | 0.00 | 0.00 | 0.40 | 7.260 | 7.260 | Bill Clinton | Bill Clinton |
| qa_2_075_len30720.json | validation.jsonl (qa_2) | 30720 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.887 | 7.887 | John John Florence | John John Florence |
| qa_2_076_len31469.json | validation.jsonl (qa_2) | 31469 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.113 | 8.113 | Sela | Sela |
| qa_2_077_len31196.json | validation.jsonl (qa_2) | 31196 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.036 | 8.036 | Ais | Ais |
| qa_2_078_len30936.json | validation.jsonl (qa_2) | 30936 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.927 | 7.927 | British | British |
| qa_2_079_len32456.json | validation.jsonl (qa_2) | 32456 | CacheBlend | 0.00 | 0.00 | 0.75 | 8.546 | 8.546 | 1895 | 1895 |
| qa_2_080_len30443.json | validation.jsonl (qa_2) | 30443 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.805 | 7.805 | Lempster | Lempster |
| qa_2_081_len30467.json | validation.jsonl (qa_2) | 30467 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.806 | 7.806 | Bob Seger | Bob Seger |
| qa_2_082_len32651.json | validation.jsonl (qa_2) | 32651 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.830 | 8.830 | the practice of requiring only men to re | the practice of requiring only men to re |
| qa_2_083_len31347.json | validation.jsonl (qa_2) | 31347 | CacheBlend | 0.00 | 1.00 | 0.89 | 8.193 | 8.193 | Mondelez International | Mondelez International |
| qa_2_084_len31142.json | validation.jsonl (qa_2) | 31142 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.143 | 8.143 | Mimi Alford | Mimi Alford |
| qa_2_085_len31063.json | validation.jsonl (qa_2) | 31063 | CacheBlend | 0.00 | 1.00 | 0.58 | 8.432 | 8.432 | Paul Manafort was born on April 1, 1949. | Paul Manafort was born on April 1, 1949. |
| qa_2_086_len32396.json | validation.jsonl (qa_2) | 32396 | CacheBlend | 0.00 | 0.00 | 0.25 | 8.524 | 8.524 | 1970 | 1970 |
| qa_2_087_len30960.json | validation.jsonl (qa_2) | 30960 | CacheBlend | 0.00 | 1.00 | 0.18 | 9.435 | 9.435 | Teide National Park is located in Teneri | Teide National Park is located in Teneri |
| qa_2_088_len32564.json | validation.jsonl (qa_2) | 32564 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.150 | 9.150 | The given documents do not provide speci | The given documents do not provide speci |
| qa_2_089_len29446.json | validation.jsonl (qa_2) | 29446 | CacheBlend | 0.00 | 1.00 | 0.00 | 7.436 | 7.436 | film director | film director |
| qa_2_090_len31022.json | validation.jsonl (qa_2) | 31022 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.981 | 7.981 | The Conversation | The Conversation |
| qa_2_091_len29529.json | validation.jsonl (qa_2) | 29529 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.472 | 7.472 | John Waters | John Waters |
| qa_2_092_len32060.json | validation.jsonl (qa_2) | 32060 | CacheBlend | 0.00 | 1.00 | 0.57 | 8.323 | 8.323 | Las Vegas | Las Vegas |
| qa_2_093_len31803.json | validation.jsonl (qa_2) | 31803 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.217 | 8.217 | No | No |
| qa_2_094_len32728.json | validation.jsonl (qa_2) | 32728 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.530 | 8.530 | November | November |
| qa_2_095_len30382.json | validation.jsonl (qa_2) | 30382 | CacheBlend | 0.00 | 1.00 | 0.25 | 8.144 | 8.144 | Tysons Galleria is located in Fairfax Co | Tysons Galleria is located in Fairfax Co |
| qa_2_096_len30281.json | validation.jsonl (qa_2) | 30281 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.275 | 8.275 | Bordan Tkachuk was the CEO of Viglen, wh | Bordan Tkachuk was the CEO of Viglen, wh |
| qa_2_097_len31183.json | validation.jsonl (qa_2) | 31183 | CacheBlend | 0.00 | 0.00 | 0.89 | 8.129 | 8.129 | Lev Yilmaz | Lev Yilmaz |
| qa_2_098_len31177.json | validation.jsonl (qa_2) | 31177 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.062 | 8.062 | Rabat | Rabat |
| qa_2_099_len31084.json | validation.jsonl (qa_2) | 31084 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.967 | 7.967 | no | no |
| qa_2_100_len29933.json | validation.jsonl (qa_2) | 29933 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.600 | 9.600 | The Black Seminoles lived mostly in dist | The Black Seminoles lived mostly in dist |
| qa_2_101_len32041.json | validation.jsonl (qa_2) | 32041 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.363 | 8.363 | Kathy Troccoli | Kathy Troccoli |
| qa_2_102_len31969.json | validation.jsonl (qa_2) | 31969 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.372 | 8.372 | 1999 | 1999 |
| qa_2_103_len29605.json | validation.jsonl (qa_2) | 29605 | CacheBlend | 0.00 | 1.00 | 0.33 | 8.331 | 8.331 | The stadium is located at the corner of  | The stadium is located at the corner of  |
| qa_2_104_len31540.json | validation.jsonl (qa_2) | 31540 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.213 | 8.213 | Leo Varadkar | Leo Varadkar |
| qa_2_105_len31867.json | validation.jsonl (qa_2) | 31867 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.290 | 8.290 | George Raft | George Raft |
| qa_2_106_len31791.json | validation.jsonl (qa_2) | 31791 | CacheBlend | 0.00 | 0.00 | 0.25 | 8.321 | 8.321 | 1946 | 1946 |
| qa_2_107_len31973.json | validation.jsonl (qa_2) | 31973 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.373 | 8.373 | valuation and investment analysis | valuation and investment analysis |
| qa_2_108_len32698.json | validation.jsonl (qa_2) | 32698 | CacheBlend | 0.00 | 1.00 | 0.64 | 9.081 | 9.081 | Zaheer Khan was born on 7 October 1978. | Zaheer Khan was born on 7 October 1978. |
| qa_2_109_len31517.json | validation.jsonl (qa_2) | 31517 | CacheBlend | 0.00 | 0.00 | 0.40 | 8.285 | 8.285 | B-1 Lancer | B-1 Lancer |
| qa_2_110_len32601.json | validation.jsonl (qa_2) | 32601 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.568 | 8.568 | Blake Shelton | Blake Shelton |
| qa_2_111_len30586.json | validation.jsonl (qa_2) | 30586 | CacheBlend | 0.00 | 0.00 | 0.07 | 8.764 | 8.764 | between approximately 1100 and 1425 AD,  | between approximately 1100 and 1425 AD,  |
| qa_2_112_len31868.json | validation.jsonl (qa_2) | 31868 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.357 | 8.357 | 1936 | 1936 |
| qa_2_113_len32716.json | validation.jsonl (qa_2) | 32716 | CacheBlend | 0.00 | 0.00 | 0.07 | 9.280 | 9.280 | Ralph Murphy died first. He passed away  | Ralph Murphy died first. He passed away  |
| qa_2_114_len30597.json | validation.jsonl (qa_2) | 30597 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.852 | 7.852 | Picric acid | Picric acid |
| qa_2_115_len31380.json | validation.jsonl (qa_2) | 31380 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.080 | 8.080 | 16 | 16 |
| qa_2_116_len31774.json | validation.jsonl (qa_2) | 31774 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.255 | 8.255 | Ghost Rider | Ghost Rider |
| qa_2_117_len31538.json | validation.jsonl (qa_2) | 31538 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.142 | 8.142 | Senator | Senator |
| qa_2_118_len32050.json | validation.jsonl (qa_2) | 32050 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.295 | 8.295 | No | No |
| qa_2_119_len30712.json | validation.jsonl (qa_2) | 30712 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.823 | 7.823 | Yes | Yes |
| qa_2_120_len32159.json | validation.jsonl (qa_2) | 32159 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.577 | 8.577 | U.S. Route 287 | U.S. Route 287 |
| qa_2_121_len32108.json | validation.jsonl (qa_2) | 32108 | CacheBlend | 0.00 | 1.00 | 0.21 | 9.289 | 9.289 | Caroline Carver played the role of Princ | Caroline Carver played the role of Princ |
| qa_2_122_len32398.json | validation.jsonl (qa_2) | 32398 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.541 | 8.541 | The Spiderwick Chronicles | The Spiderwick Chronicles |
| qa_2_123_len32274.json | validation.jsonl (qa_2) | 32274 | CacheBlend | 0.00 | 0.00 | 0.20 | 8.517 | 8.517 | Kohlberg Kravis Roberts | Kohlberg Kravis Roberts |
| qa_2_124_len31640.json | validation.jsonl (qa_2) | 31640 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.367 | 8.367 | You Can't Kill Stephen King | You Can't Kill Stephen King |
| qa_2_125_len31802.json | validation.jsonl (qa_2) | 31802 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.289 | 8.289 | Richmond River | Richmond River |
| qa_2_126_len32646.json | validation.jsonl (qa_2) | 32646 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.617 | 8.617 | Owsley Stanley | Owsley Stanley |
| qa_2_127_len32694.json | validation.jsonl (qa_2) | 32694 | CacheBlend | 0.00 | 0.00 | 0.20 | 8.627 | 8.627 | 1386 | 1386 |
| qa_2_128_len32415.json | validation.jsonl (qa_2) | 32415 | CacheBlend | 0.00 | 0.00 | 0.67 | 8.675 | 8.675 | downtown San Luis Obispo | downtown San Luis Obispo |
| qa_2_129_len30184.json | validation.jsonl (qa_2) | 30184 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.831 | 7.831 | Papa John's Cardinal Stadium | Papa John's Cardinal Stadium |
| qa_2_130_len32056.json | validation.jsonl (qa_2) | 32056 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.567 | 8.567 | LaLee's Kin: The Legacy of Cotton | LaLee's Kin: The Legacy of Cotton |
| qa_2_131_len31463.json | validation.jsonl (qa_2) | 31463 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.185 | 8.185 | Pirate's Cove | Pirate's Cove |
| qa_2_132_len32387.json | validation.jsonl (qa_2) | 32387 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.497 | 8.497 | Square Enix | Square Enix |
| qa_2_133_len31896.json | validation.jsonl (qa_2) | 31896 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.464 | 8.464 | The Marshall Mathers LP 2 | The Marshall Mathers LP 2 |
| qa_2_134_len32449.json | validation.jsonl (qa_2) | 32449 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.518 | 8.518 | Harsh Times | Harsh Times |
| qa_2_135_len31438.json | validation.jsonl (qa_2) | 31438 | CacheBlend | 0.00 | 0.00 | 0.57 | 8.137 | 8.137 | Jillian Bell | Jillian Bell |
| qa_2_136_len32293.json | validation.jsonl (qa_2) | 32293 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.483 | 8.483 | Texas Tech University | Texas Tech University |
| qa_2_137_len31037.json | validation.jsonl (qa_2) | 31037 | CacheBlend | 0.00 | 0.00 | 0.21 | 8.932 | 8.932 | Stapleton Cotton, 1st Viscount Combermer | Stapleton Cotton, 1st Viscount Combermer |
| qa_2_138_len31463.json | validation.jsonl (qa_2) | 31463 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.157 | 8.157 | stop motion animation | stop motion animation |
| qa_2_139_len32591.json | validation.jsonl (qa_2) | 32591 | CacheBlend | 0.00 | 0.00 | 0.50 | 8.597 | 8.597 | 2008 | 2008 |
| qa_2_140_len32661.json | validation.jsonl (qa_2) | 32661 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.596 | 8.596 | Oxford University | Oxford University |
| qa_2_141_len32543.json | validation.jsonl (qa_2) | 32543 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.557 | 8.557 | Roberta Vinci | Roberta Vinci |
| qa_2_142_len31613.json | validation.jsonl (qa_2) | 31613 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.756 | 8.756 | Truth in Science and Discovery both embr | Truth in Science and Discovery both embr |
| qa_2_143_len32393.json | validation.jsonl (qa_2) | 32393 | CacheBlend | 0.00 | 0.00 | 0.13 | 8.461 | 8.461 | Ian Watkins | Ian Watkins |
| qa_2_144_len32167.json | validation.jsonl (qa_2) | 32167 | CacheBlend | 0.00 | 1.00 | 0.26 | 9.308 | 9.308 | The movies Queen of Blood and Battle Bey | The movies Queen of Blood and Battle Bey |
| qa_2_145_len30511.json | validation.jsonl (qa_2) | 30511 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.877 | 7.877 | Erika Jayne | Erika Jayne |
| qa_2_146_len32711.json | validation.jsonl (qa_2) | 32711 | CacheBlend | 0.00 | 1.00 | 0.42 | 9.068 | 9.068 | The Reading Post changed its name to Rea | The Reading Post changed its name to Rea |
| qa_2_147_len30681.json | validation.jsonl (qa_2) | 30681 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.954 | 7.954 | Kingdom of the Isles | Kingdom of the Isles |
| qa_2_148_len31691.json | validation.jsonl (qa_2) | 31691 | CacheBlend | 0.00 | 1.00 | 0.11 | 8.857 | 8.857 | Bruce Spizer is an expert on The Beatles | Bruce Spizer is an expert on The Beatles |
| qa_2_149_len31748.json | validation.jsonl (qa_2) | 31748 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.327 | 8.327 | Chattahoochee | Chattahoochee |
| qa_2_150_len32058.json | validation.jsonl (qa_2) | 32058 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.291 | 8.291 | director | director |
| qa_2_151_len32224.json | validation.jsonl (qa_2) | 32224 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.445 | 8.445 | 1901 | 1901 |
| qa_2_152_len32110.json | validation.jsonl (qa_2) | 32110 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.372 | 8.372 | Argand lamp | Argand lamp |
| qa_2_153_len30330.json | validation.jsonl (qa_2) | 30330 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.722 | 7.722 | 16 | 16 |
| qa_2_154_len31886.json | validation.jsonl (qa_2) | 31886 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.231 | 8.231 | Yes | Yes |
| qa_2_155_len32709.json | validation.jsonl (qa_2) | 32709 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.729 | 8.729 | 2014 FIFA World Cup | 2014 FIFA World Cup |
| qa_2_156_len32375.json | validation.jsonl (qa_2) | 32375 | CacheBlend | 0.00 | 1.00 | 0.46 | 8.839 | 8.839 | The Seven Days Battles were fought earli | The Seven Days Battles were fought earli |
| qa_2_157_len32196.json | validation.jsonl (qa_2) | 32196 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.434 | 8.434 | Lush Ltd. | Lush Ltd. |
| qa_2_158_len32037.json | validation.jsonl (qa_2) | 32037 | CacheBlend | 0.00 | 0.00 | 0.00 | 11.842 | 11.842 | Trafford Town Hall, which is home to Tra | Trafford Town Hall, which is home to Tra |
| qa_2_159_len32334.json | validation.jsonl (qa_2) | 32334 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.515 | 8.515 | 1989 | 1989 |
| qa_2_160_len32507.json | validation.jsonl (qa_2) | 32507 | CacheBlend | 0.00 | 1.00 | 0.71 | 8.769 | 8.769 | Kathryn Jean Martin "Kathy" Sullivan | Kathryn Jean Martin "Kathy" Sullivan |
| qa_2_161_len31707.json | validation.jsonl (qa_2) | 31707 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.258 | 8.258 | Frank Lowy | Frank Lowy |
| qa_2_162_len32246.json | validation.jsonl (qa_2) | 32246 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.447 | 8.447 | Summerlin, Nevada | Summerlin, Nevada |
| qa_2_163_len30403.json | validation.jsonl (qa_2) | 30403 | CacheBlend | 0.00 | 0.00 | 0.89 | 7.798 | 7.798 | Kelly Osbourne | Kelly Osbourne |
| qa_2_164_len32441.json | validation.jsonl (qa_2) | 32441 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.550 | 8.550 | Sean Yseult | Sean Yseult |
| qa_2_165_len31303.json | validation.jsonl (qa_2) | 31303 | CacheBlend | 0.00 | 1.00 | 0.19 | 9.613 | 9.613 | 3/4th of the Food and Drug Administratio | 3/4th of the Food and Drug Administratio |
| qa_2_166_len32160.json | validation.jsonl (qa_2) | 32160 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.465 | 8.465 | McFarland, USA | McFarland, USA |
| qa_2_167_len32056.json | validation.jsonl (qa_2) | 32056 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.360 | 8.360 | Wichita | Wichita |
| qa_2_168_len30372.json | validation.jsonl (qa_2) | 30372 | CacheBlend | 0.00 | 0.00 | 0.38 | 8.970 | 8.970 | The 58th quadrennial American presidenti | The 58th quadrennial American presidenti |
| qa_2_169_len32303.json | validation.jsonl (qa_2) | 32303 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.412 | 8.412 | Roman | Roman |
| qa_2_170_len29727.json | validation.jsonl (qa_2) | 29727 | CacheBlend | 0.00 | 0.00 | 0.50 | 7.657 | 7.657 | Sue Grafton | Sue Grafton |
| qa_2_171_len31132.json | validation.jsonl (qa_2) | 31132 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.132 | 8.132 | Brooklyn, New York | Brooklyn, New York |
| qa_2_172_len31576.json | validation.jsonl (qa_2) | 31576 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.320 | 8.320 | 48,982 | 48,982 |
| qa_2_173_len32568.json | validation.jsonl (qa_2) | 32568 | CacheBlend | 0.00 | 0.00 | 0.89 | 8.619 | 8.619 | Pasek and Paul | Pasek and Paul |
| qa_2_174_len32411.json | validation.jsonl (qa_2) | 32411 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.515 | 8.515 | Niagara Park | Niagara Park |
| qa_2_175_len32470.json | validation.jsonl (qa_2) | 32470 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.458 | 8.458 | yes | yes |
| qa_2_176_len31431.json | validation.jsonl (qa_2) | 31431 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.067 | 8.067 | works | works |
| qa_2_177_len31630.json | validation.jsonl (qa_2) | 31630 | CacheBlend | 0.00 | 0.00 | 0.02 | 12.035 | 12.035 | The small settlement situated on the riv | The small settlement situated on the riv |
| qa_2_178_len30925.json | validation.jsonl (qa_2) | 30925 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.992 | 7.992 | William Bonfield | William Bonfield |
| qa_2_179_len32423.json | validation.jsonl (qa_2) | 32423 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.485 | 8.485 | Scotland | Scotland |
| qa_2_180_len29197.json | validation.jsonl (qa_2) | 29197 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.417 | 7.417 | The Drop-In | The Drop-In |
| qa_2_181_len32221.json | validation.jsonl (qa_2) | 32221 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.379 | 8.379 | Welcome | Welcome |
| qa_2_182_len31321.json | validation.jsonl (qa_2) | 31321 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.112 | 8.112 | George Harrison | George Harrison |
| qa_2_183_len31596.json | validation.jsonl (qa_2) | 31596 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.187 | 8.187 | center | center |
| qa_2_184_len32380.json | validation.jsonl (qa_2) | 32380 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.569 | 8.569 | 1952 | 1952 |
| qa_2_185_len30717.json | validation.jsonl (qa_2) | 30717 | CacheBlend | 0.00 | 1.00 | 0.67 | 7.888 | 7.888 | third season | third season |
| qa_2_186_len32520.json | validation.jsonl (qa_2) | 32520 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.706 | 8.706 | Mark Neveldine and Brian Taylor | Mark Neveldine and Brian Taylor |
| qa_2_187_len30471.json | validation.jsonl (qa_2) | 30471 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.914 | 7.914 | Hidden America with Jonah Ray | Hidden America with Jonah Ray |
| qa_2_188_len31954.json | validation.jsonl (qa_2) | 31954 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.296 | 8.296 | Yes | Yes |
| qa_2_189_len31141.json | validation.jsonl (qa_2) | 31141 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.137 | 8.137 | 1943 | 1943 |
| qa_2_190_len31498.json | validation.jsonl (qa_2) | 31498 | CacheBlend | 0.00 | 0.00 | 0.44 | 8.248 | 8.248 | Oregon State Beavers | Oregon State Beavers |
| qa_2_191_len29900.json | validation.jsonl (qa_2) | 29900 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.584 | 7.584 | NBC | NBC |
| qa_2_192_len31226.json | validation.jsonl (qa_2) | 31226 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.122 | 8.122 | Algerian | Algerian |
| qa_2_193_len31968.json | validation.jsonl (qa_2) | 31968 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.386 | 8.386 | New York City | New York City |
| qa_2_194_len32169.json | validation.jsonl (qa_2) | 32169 | CacheBlend | 0.00 | 1.00 | 0.83 | 8.639 | 8.639 | Severe acute respiratory syndrome (SARS) | Severe acute respiratory syndrome (SARS) |
| qa_2_195_len32576.json | validation.jsonl (qa_2) | 32576 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.666 | 8.666 | Wendell Berry | Wendell Berry |
| qa_2_196_len31556.json | validation.jsonl (qa_2) | 31556 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.341 | 8.341 | Vikram Bhatt | Vikram Bhatt |
| qa_2_197_len29159.json | validation.jsonl (qa_2) | 29159 | CacheBlend | 0.00 | 1.00 | 0.33 | 7.847 | 7.847 | Robert Smith founded General Mills, whic | Robert Smith founded General Mills, whic |
| qa_2_198_len32190.json | validation.jsonl (qa_2) | 32190 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.986 | 9.986 | The character from Son of al Qaeda decla | The character from Son of al Qaeda decla |
| qa_2_199_len29142.json | validation.jsonl (qa_2) | 29142 | CacheBlend | 0.00 | 0.00 | 0.17 | 7.483 | 7.483 | Cyclone Gonu | Cyclone Gonu |
| qa_2_200_len31724.json | validation.jsonl (qa_2) | 31724 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.521 | 8.521 | Snatch premiered on the streaming servic | Snatch premiered on the streaming servic |
| qa_2_201_len30493.json | validation.jsonl (qa_2) | 30493 | CacheBlend | 0.00 | 1.00 | 0.80 | 7.856 | 7.856 | Salford City | Salford City |
| qa_2_202_len31148.json | validation.jsonl (qa_2) | 31148 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.021 | 8.021 | Michael Graves | Michael Graves |
| qa_2_203_len32602.json | validation.jsonl (qa_2) | 32602 | CacheBlend | 0.00 | 0.00 | 0.05 | 9.980 | 9.980 | American Hustle received the award for B | American Hustle received the award for B |
| qa_2_204_len32407.json | validation.jsonl (qa_2) | 32407 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.509 | 8.509 | Kevin Spacey | Kevin Spacey |
| qa_2_205_len31610.json | validation.jsonl (qa_2) | 31610 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.936 | 8.936 | Onika Tanya Maraj is not mentioned as a  | Onika Tanya Maraj is not mentioned as a  |
| qa_2_206_len31049.json | validation.jsonl (qa_2) | 31049 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.958 | 7.958 | Yes | Yes |
| qa_2_207_len30987.json | validation.jsonl (qa_2) | 30987 | CacheBlend | 0.00 | 1.00 | 0.71 | 8.058 | 8.058 | Charles Hastings Judd | Charles Hastings Judd |
| qa_2_208_len30476.json | validation.jsonl (qa_2) | 30476 | CacheBlend | 0.00 | 0.00 | 0.71 | 7.885 | 7.885 | Pavel Alexandrov | Pavel Alexandrov |
| qa_2_209_len31760.json | validation.jsonl (qa_2) | 31760 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.293 | 8.293 | Gal Gadot | Gal Gadot |
| qa_2_210_len29911.json | validation.jsonl (qa_2) | 29911 | CacheBlend | 0.00 | 0.00 | 0.34 | 8.381 | 8.381 | Captain J. Edward Snyder was the captain | Captain J. Edward Snyder was the captain |
| qa_2_211_len31680.json | validation.jsonl (qa_2) | 31680 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.257 | 8.257 | Jackson Storm | Jackson Storm |
| qa_2_212_len32035.json | validation.jsonl (qa_2) | 32035 | CacheBlend | 0.00 | 0.00 | 0.67 | 8.466 | 8.466 | 712 Fifth Avenue | 712 Fifth Avenue |
| qa_2_213_len31506.json | validation.jsonl (qa_2) | 31506 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.152 | 8.152 | right-hand | right-hand |
| qa_2_214_len31684.json | validation.jsonl (qa_2) | 31684 | CacheBlend | 0.00 | 0.00 | 0.80 | 8.299 | 8.299 | Arthur Conan Doyle | Arthur Conan Doyle |
| qa_2_215_len32235.json | validation.jsonl (qa_2) | 32235 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.446 | 8.446 | Knoxville | Knoxville |
| qa_2_216_len32245.json | validation.jsonl (qa_2) | 32245 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.390 | 8.390 | five | five |
| qa_2_217_len31129.json | validation.jsonl (qa_2) | 31129 | CacheBlend | 0.00 | 0.00 | 0.50 | 8.131 | 8.131 | Sasanian Empire | Sasanian Empire |
| qa_2_218_len32672.json | validation.jsonl (qa_2) | 32672 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.628 | 8.628 | Erinsborough | Erinsborough |
| qa_2_219_len31578.json | validation.jsonl (qa_2) | 31578 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.239 | 8.239 | Samoa | Samoa |
| qa_2_220_len31878.json | validation.jsonl (qa_2) | 31878 | CacheBlend | 0.00 | 1.00 | 0.62 | 8.337 | 8.337 | Kimmy Schmidt | Kimmy Schmidt |
| qa_2_221_len32711.json | validation.jsonl (qa_2) | 32711 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.761 | 8.761 | Filthy Rich & Catflap | Filthy Rich & Catflap |
| qa_2_222_len31806.json | validation.jsonl (qa_2) | 31806 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.256 | 8.256 | Yes | Yes |
| qa_2_223_len32483.json | validation.jsonl (qa_2) | 32483 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.600 | 8.600 | 19th century | 19th century |
| qa_2_224_len32254.json | validation.jsonl (qa_2) | 32254 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.412 | 8.412 | Awake | Awake |
| qa_2_225_len32092.json | validation.jsonl (qa_2) | 32092 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.374 | 8.374 | French Canadians | French Canadians |
| qa_2_226_len30714.json | validation.jsonl (qa_2) | 30714 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.853 | 7.853 | American | American |
| qa_2_227_len30895.json | validation.jsonl (qa_2) | 30895 | CacheBlend | 0.00 | 1.00 | 0.07 | 10.183 | 10.183 | The bridge in the Öresund Region that co | The bridge in the Öresund Region that co |
| qa_2_228_len32344.json | validation.jsonl (qa_2) | 32344 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.520 | 8.520 | Clint Eastwood | Clint Eastwood |
| qa_2_229_len31901.json | validation.jsonl (qa_2) | 31901 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.356 | 8.356 | Marsilea | Marsilea |
| qa_2_230_len31964.json | validation.jsonl (qa_2) | 31964 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.368 | 8.368 | St James's Palace | St James's Palace |
| qa_2_231_len30703.json | validation.jsonl (qa_2) | 30703 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.857 | 7.857 | Starlite | Starlite |
| qa_2_232_len31284.json | validation.jsonl (qa_2) | 31284 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.168 | 8.168 | CapitaGreen in Singapore | CapitaGreen in Singapore |
| qa_2_233_len30764.json | validation.jsonl (qa_2) | 30764 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.999 | 7.999 | Craig v. Boren | Craig v. Boren |
| qa_2_234_len31441.json | validation.jsonl (qa_2) | 31441 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.098 | 8.098 | Waterfront | Waterfront |
| qa_2_235_len30237.json | validation.jsonl (qa_2) | 30237 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.742 | 7.742 | Arrowhead Stadium | Arrowhead Stadium |
| qa_2_236_len32094.json | validation.jsonl (qa_2) | 32094 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.337 | 8.337 | cranberries | cranberries |
| qa_2_237_len32400.json | validation.jsonl (qa_2) | 32400 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.556 | 8.556 | Strip search phone call scam | Strip search phone call scam |
| qa_2_238_len30903.json | validation.jsonl (qa_2) | 30903 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.017 | 8.017 | Logan International Airport | Logan International Airport |
| qa_2_239_len30911.json | validation.jsonl (qa_2) | 30911 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.017 | 8.017 | Ehretia | Ehretia |
| qa_2_240_len30810.json | validation.jsonl (qa_2) | 30810 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.023 | 8.023 | Frank Thomas' Big Hurt | Frank Thomas' Big Hurt |
| qa_2_241_len31387.json | validation.jsonl (qa_2) | 31387 | CacheBlend | 0.00 | 0.00 | 0.18 | 8.198 | 8.198 | Paige O'Hara | Paige O'Hara |
| qa_2_242_len31624.json | validation.jsonl (qa_2) | 31624 | CacheBlend | 0.00 | 1.00 | 0.62 | 8.218 | 8.218 | Reggie Jackson | Reggie Jackson |
| qa_2_243_len31589.json | validation.jsonl (qa_2) | 31589 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.185 | 8.185 | Peter Perry | Peter Perry |
| qa_2_244_len31686.json | validation.jsonl (qa_2) | 31686 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.243 | 8.243 | Sarod | Sarod |
| qa_2_245_len30310.json | validation.jsonl (qa_2) | 30310 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.023 | 8.023 | Lemony Snicket's A Series of Unfortunate | Lemony Snicket's A Series of Unfortunate |
| qa_2_246_len32327.json | validation.jsonl (qa_2) | 32327 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.443 | 8.443 | 25 | 25 |
| qa_2_247_len31801.json | validation.jsonl (qa_2) | 31801 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.217 | 8.217 | No | No |
| qa_2_248_len31406.json | validation.jsonl (qa_2) | 31406 | CacheBlend | 0.00 | 0.00 | 0.80 | 8.097 | 8.097 | Thomas Friedman | Thomas Friedman |
| qa_2_249_len29731.json | validation.jsonl (qa_2) | 29731 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.553 | 7.553 | Taylor Swift | Taylor Swift |
| qa_2_250_len32062.json | validation.jsonl (qa_2) | 32062 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.323 | 8.323 | Istanbul | Istanbul |
| qa_2_251_len31295.json | validation.jsonl (qa_2) | 31295 | CacheBlend | 0.00 | 1.00 | 0.26 | 9.001 | 9.001 | Opry Mills is a super-regional shopping  | Opry Mills is a super-regional shopping  |
| qa_2_252_len32173.json | validation.jsonl (qa_2) | 32173 | CacheBlend | 0.00 | 1.00 | 0.07 | 10.341 | 10.341 | It is not possible to determine this fro | It is not possible to determine this fro |
| qa_2_253_len30672.json | validation.jsonl (qa_2) | 30672 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.939 | 7.939 | Alain Resnais | Alain Resnais |
| qa_2_254_len32374.json | validation.jsonl (qa_2) | 32374 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.489 | 8.489 | 600 | 600 |
| qa_2_255_len30428.json | validation.jsonl (qa_2) | 30428 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.807 | 7.807 | Berea College | Berea College |
| qa_2_256_len32231.json | validation.jsonl (qa_2) | 32231 | CacheBlend | 0.00 | 0.00 | 0.75 | 8.443 | 8.443 | 1944 | 1944 |
| qa_2_257_len32091.json | validation.jsonl (qa_2) | 32091 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.369 | 8.369 | House of Windsor | House of Windsor |
| qa_2_258_len30288.json | validation.jsonl (qa_2) | 30288 | CacheBlend | 0.00 | 1.00 | 0.47 | 8.118 | 8.118 | James Burke produced the documentary tel | James Burke produced the documentary tel |
| qa_2_259_len32210.json | validation.jsonl (qa_2) | 32210 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.430 | 8.430 | Gulangyu | Gulangyu |
| qa_2_260_len31376.json | validation.jsonl (qa_2) | 31376 | CacheBlend | 0.00 | 1.00 | 0.27 | 8.581 | 8.581 | Kurt Vonnegut, best known for "Slaughter | Kurt Vonnegut, best known for "Slaughter |
| qa_2_261_len32162.json | validation.jsonl (qa_2) | 32162 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.395 | 8.395 | Leinster | Leinster |
| qa_2_262_len31692.json | validation.jsonl (qa_2) | 31692 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.646 | 8.646 | Sean Connor played for Ards in the NIFL  | Sean Connor played for Ards in the NIFL  |
| qa_2_263_len32274.json | validation.jsonl (qa_2) | 32274 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.408 | 8.408 | Romeo | Romeo |
| qa_2_264_len32213.json | validation.jsonl (qa_2) | 32213 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.372 | 8.372 | SKUM | SKUM |
| qa_2_265_len31537.json | validation.jsonl (qa_2) | 31537 | CacheBlend | 0.00 | 0.00 | 0.67 | 8.314 | 8.314 | Anabolic-androgenic steroids | Anabolic-androgenic steroids |
| qa_2_266_len30372.json | validation.jsonl (qa_2) | 30372 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.829 | 7.829 | Don S. Davis | Don S. Davis |
| qa_2_267_len32695.json | validation.jsonl (qa_2) | 32695 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.655 | 8.655 | Lady Antebellum | Lady Antebellum |
| qa_2_268_len32712.json | validation.jsonl (qa_2) | 32712 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.370 | 9.370 | The National Archives and Library of Eth | The National Archives and Library of Eth |
| qa_2_269_len32098.json | validation.jsonl (qa_2) | 32098 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.373 | 8.373 | Anna Clyne | Anna Clyne |
| qa_2_270_len32727.json | validation.jsonl (qa_2) | 32727 | CacheBlend | 0.00 | 1.00 | 0.75 | 8.776 | 8.776 | USS Essex (CV-9) | USS Essex (CV-9) |
| qa_2_271_len30866.json | validation.jsonl (qa_2) | 30866 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.891 | 7.891 | Yes | Yes |
| qa_2_272_len28671.json | validation.jsonl (qa_2) | 28671 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.179 | 7.179 | Catwoman | Catwoman |
| qa_2_273_len31791.json | validation.jsonl (qa_2) | 31791 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.283 | 8.283 | 20 mi | 20 mi |
| qa_2_274_len31221.json | validation.jsonl (qa_2) | 31221 | CacheBlend | 0.00 | 0.00 | 0.89 | 8.071 | 8.071 | Laura Osnes | Laura Osnes |
| qa_2_275_len30770.json | validation.jsonl (qa_2) | 30770 | CacheBlend | 0.00 | 1.00 | 0.57 | 7.999 | 7.999 | more than 20 | more than 20 |
| qa_2_276_len30627.json | validation.jsonl (qa_2) | 30627 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.904 | 7.904 | Eucryphia | Eucryphia |
| qa_2_277_len32523.json | validation.jsonl (qa_2) | 32523 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.492 | 8.492 | Venice | Venice |
| qa_2_278_len32511.json | validation.jsonl (qa_2) | 32511 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.559 | 8.559 | Waimea Bay | Waimea Bay |
| qa_2_279_len31696.json | validation.jsonl (qa_2) | 31696 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.243 | 8.243 | Christianity Today | Christianity Today |
| qa_2_280_len31845.json | validation.jsonl (qa_2) | 31845 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.264 | 8.264 | James Franco | James Franco |
| qa_2_281_len32508.json | validation.jsonl (qa_2) | 32508 | CacheBlend | 0.00 | 1.00 | 0.73 | 8.565 | 8.565 | Humphrey Goodman | Humphrey Goodman |
| qa_2_282_len30671.json | validation.jsonl (qa_2) | 30671 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.804 | 7.804 | December | December |
| qa_2_283_len29523.json | validation.jsonl (qa_2) | 29523 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.493 | 7.493 | Sacred Planet | Sacred Planet |
| qa_2_284_len30416.json | validation.jsonl (qa_2) | 30416 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.762 | 7.762 | Calgary | Calgary |
| qa_2_285_len32363.json | validation.jsonl (qa_2) | 32363 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.555 | 8.555 | Ferdi Taygan | Ferdi Taygan |
| qa_2_286_len31811.json | validation.jsonl (qa_2) | 31811 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.029 | 9.029 | There is no information provided about t | There is no information provided about t |
| qa_2_287_len31209.json | validation.jsonl (qa_2) | 31209 | CacheBlend | 0.00 | 1.00 | 0.27 | 8.911 | 8.911 | Ian Harland studied at Peterhouse, Cambr | Ian Harland studied at Peterhouse, Cambr |
| qa_2_288_len32482.json | validation.jsonl (qa_2) | 32482 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.525 | 8.525 | Suicide Kings | Suicide Kings |
| qa_2_289_len32306.json | validation.jsonl (qa_2) | 32306 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.723 | 8.723 | Hey Pa! There's a Goat on the Roof | Hey Pa! There's a Goat on the Roof |
| qa_2_290_len32438.json | validation.jsonl (qa_2) | 32438 | CacheBlend | 0.00 | 0.00 | 0.36 | 8.578 | 8.578 | 23-16 | 23-16 |
| qa_2_291_len30709.json | validation.jsonl (qa_2) | 30709 | CacheBlend | 0.00 | 1.00 | 0.53 | 7.952 | 7.952 | 8,211 | 8,211 |
| qa_2_292_len32718.json | validation.jsonl (qa_2) | 32718 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.552 | 8.552 | Salvie | Salvie |
| qa_2_293_len31209.json | validation.jsonl (qa_2) | 31209 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.036 | 8.036 | Architect | Architect |
| qa_2_294_len31094.json | validation.jsonl (qa_2) | 31094 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.030 | 8.030 | Will Arnett | Will Arnett |
| qa_2_295_len29867.json | validation.jsonl (qa_2) | 29867 | CacheBlend | 0.00 | 0.00 | 0.40 | 7.585 | 7.585 | German Army | German Army |
| qa_2_296_len30731.json | validation.jsonl (qa_2) | 30731 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.907 | 7.907 | the Canary Islands | the Canary Islands |
| qa_2_297_len32181.json | validation.jsonl (qa_2) | 32181 | CacheBlend | 0.00 | 1.00 | 0.59 | 9.029 | 9.029 | The Treaty of Gandamak officially ended  | The Treaty of Gandamak officially ended  |
| qa_2_298_len32651.json | validation.jsonl (qa_2) | 32651 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.519 | 8.519 | Oregon | Oregon |
| qa_2_299_len32200.json | validation.jsonl (qa_2) | 32200 | CacheBlend | 0.00 | 1.00 | 0.25 | 10.035 | 10.035 | The rock band that sang "All Join Hands" | The rock band that sang "All Join Hands" |
| qa_2_300_len32724.json | validation.jsonl (qa_2) | 32724 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.621 | 8.621 | Greene King | Greene King |
| qa_2_301_len32422.json | validation.jsonl (qa_2) | 32422 | CacheBlend | 0.00 | 1.00 | 0.75 | 8.496 | 8.496 | 20s | 20s |
| qa_2_302_len32008.json | validation.jsonl (qa_2) | 32008 | CacheBlend | 0.00 | 1.00 | 0.70 | 8.768 | 8.768 | Diana Quick's birthday is 23 November 19 | Diana Quick's birthday is 23 November 19 |
| qa_2_303_len29376.json | validation.jsonl (qa_2) | 29376 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.150 | 8.150 | The film is "Third and Indiana", which i | The film is "Third and Indiana", which i |
| qa_2_304_len32365.json | validation.jsonl (qa_2) | 32365 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.408 | 8.408 | No | No |
| qa_2_305_len30792.json | validation.jsonl (qa_2) | 30792 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.968 | 7.968 | 1901 | 1901 |
| qa_2_306_len31171.json | validation.jsonl (qa_2) | 31171 | CacheBlend | 0.00 | 0.00 | 0.14 | 8.446 | 8.446 | Rivington Hall Barn is near Rivington, L | Rivington Hall Barn is near Rivington, L |
| qa_2_307_len31709.json | validation.jsonl (qa_2) | 31709 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.255 | 8.255 | George Cayley | George Cayley |
| qa_2_308_len31374.json | validation.jsonl (qa_2) | 31374 | CacheBlend | 0.00 | 1.00 | 0.50 | 8.075 | 8.075 | Leonberger | Leonberger |
| qa_2_309_len32283.json | validation.jsonl (qa_2) | 32283 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.410 | 8.410 | No | No |
| qa_2_310_len32595.json | validation.jsonl (qa_2) | 32595 | CacheBlend | 0.00 | 0.00 | 0.75 | 8.584 | 8.584 | 2015 | 2015 |
| qa_2_311_len31753.json | validation.jsonl (qa_2) | 31753 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.257 | 8.257 | Shonen Jump | Shonen Jump |
| qa_2_312_len32192.json | validation.jsonl (qa_2) | 32192 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.395 | 8.395 | Kellogg's | Kellogg's |
| qa_2_313_len32236.json | validation.jsonl (qa_2) | 32236 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.412 | 8.412 | Am Rong | Am Rong |
| qa_2_314_len31720.json | validation.jsonl (qa_2) | 31720 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.222 | 8.222 | 29 | 29 |
| qa_2_315_len31422.json | validation.jsonl (qa_2) | 31422 | CacheBlend | 0.00 | 1.00 | 0.38 | 8.378 | 8.378 | Jimmy Fallon was born in Bay Ridge, Broo | Jimmy Fallon was born in Bay Ridge, Broo |
| qa_2_316_len31797.json | validation.jsonl (qa_2) | 31797 | CacheBlend | 0.00 | 1.00 | 0.86 | 8.319 | 8.319 | Northumbrian dialect | Northumbrian dialect |
| qa_2_317_len31695.json | validation.jsonl (qa_2) | 31695 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.246 | 8.246 | Oldham County | Oldham County |
| qa_2_318_len32593.json | validation.jsonl (qa_2) | 32593 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.480 | 8.480 | No | No |
| qa_2_319_len30442.json | validation.jsonl (qa_2) | 30442 | CacheBlend | 0.00 | 1.00 | 0.06 | 8.854 | 8.854 | The Voice of Ireland had The Saturdays a | The Voice of Ireland had The Saturdays a |
| qa_2_320_len30695.json | validation.jsonl (qa_2) | 30695 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.889 | 7.889 | John Fogerty | John Fogerty |
| qa_2_321_len32063.json | validation.jsonl (qa_2) | 32063 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.463 | 8.463 | FIFA Women's World Cup | FIFA Women's World Cup |
| qa_2_322_len32197.json | validation.jsonl (qa_2) | 32197 | CacheBlend | 0.00 | 0.00 | 0.50 | 8.458 | 8.458 | 1987 | 1987 |
| qa_2_323_len32077.json | validation.jsonl (qa_2) | 32077 | CacheBlend | 0.00 | 1.00 | 0.75 | 8.458 | 8.458 | Wilmette, Illinois | Wilmette, Illinois |
| qa_2_324_len30779.json | validation.jsonl (qa_2) | 30779 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.049 | 8.049 | Oedipus Rex | Oedipus Rex |
| qa_2_325_len32041.json | validation.jsonl (qa_2) | 32041 | CacheBlend | 0.00 | 1.00 | 0.07 | 9.979 | 9.979 | Katherine Waterston and Chrisann Brennan | Katherine Waterston and Chrisann Brennan |
| qa_2_326_len32297.json | validation.jsonl (qa_2) | 32297 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.529 | 8.529 | Grampa Simpson | Grampa Simpson |
| qa_2_327_len32637.json | validation.jsonl (qa_2) | 32637 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.567 | 8.567 | Graduation | Graduation |
| qa_2_328_len30871.json | validation.jsonl (qa_2) | 30871 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.954 | 7.954 | Yes | Yes |
| qa_2_329_len32392.json | validation.jsonl (qa_2) | 32392 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.530 | 8.530 | Kunming | Kunming |
| qa_2_330_len31842.json | validation.jsonl (qa_2) | 31842 | CacheBlend | 0.00 | 0.00 | 0.00 | 10.456 | 10.456 | Alex McKenna, the singer of "B Boy", was | Alex McKenna, the singer of "B Boy", was |
| qa_2_331_len32686.json | validation.jsonl (qa_2) | 32686 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.623 | 8.623 | minor basilica | minor basilica |
| qa_2_332_len30753.json | validation.jsonl (qa_2) | 30753 | CacheBlend | 0.00 | 1.00 | 0.89 | 8.003 | 8.003 | 150 million | 150 million |
| qa_2_333_len32494.json | validation.jsonl (qa_2) | 32494 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.534 | 8.534 | Aqua | Aqua |
| qa_2_334_len32334.json | validation.jsonl (qa_2) | 32334 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.622 | 8.622 | Reynolds v. Sims | Reynolds v. Sims |
| qa_2_335_len32363.json | validation.jsonl (qa_2) | 32363 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.527 | 8.527 | The Good Wife | The Good Wife |
| qa_2_336_len32139.json | validation.jsonl (qa_2) | 32139 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.340 | 8.340 | Pakistan | Pakistan |
| qa_2_337_len30906.json | validation.jsonl (qa_2) | 30906 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.023 | 8.023 | The Four Seasons | The Four Seasons |
| qa_2_338_len30951.json | validation.jsonl (qa_2) | 30951 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.965 | 7.965 | No | No |
| qa_2_339_len31407.json | validation.jsonl (qa_2) | 31407 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.170 | 8.170 | 50 mi | 50 mi |
| qa_2_340_len30014.json | validation.jsonl (qa_2) | 30014 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.849 | 7.849 | Victor Rådström | Victor Rådström |
| qa_2_341_len32343.json | validation.jsonl (qa_2) | 32343 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.529 | 8.529 | Happy Death Day | Happy Death Day |
| qa_2_342_len30790.json | validation.jsonl (qa_2) | 30790 | CacheBlend | 0.00 | 0.00 | 0.10 | 10.759 | 10.759 | John Hume was not the president of Sinn  | John Hume was not the president of Sinn  |
| qa_2_343_len30744.json | validation.jsonl (qa_2) | 30744 | CacheBlend | 0.00 | 1.00 | 0.14 | 10.474 | 10.474 | The Battle of Belleau Wood occurred earl | The Battle of Belleau Wood occurred earl |
| qa_2_344_len31287.json | validation.jsonl (qa_2) | 31287 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.126 | 8.126 | Royal Australian Navy | Royal Australian Navy |
| qa_2_345_len31815.json | validation.jsonl (qa_2) | 31815 | CacheBlend | 0.00 | 0.00 | 0.71 | 8.426 | 8.426 | 240,000 | 240,000 |
| qa_2_346_len30274.json | validation.jsonl (qa_2) | 30274 | CacheBlend | 0.00 | 0.00 | 0.73 | 7.778 | 7.778 | West Point Foundry | West Point Foundry |
| qa_2_347_len31704.json | validation.jsonl (qa_2) | 31704 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.222 | 8.222 | Jack Ryan | Jack Ryan |
| qa_2_348_len32642.json | validation.jsonl (qa_2) | 32642 | CacheBlend | 0.00 | 1.00 | 0.00 | 10.728 | 10.728 | Wheat salad is not typically composed of | Wheat salad is not typically composed of |
| qa_2_349_len29993.json | validation.jsonl (qa_2) | 29993 | CacheBlend | 0.00 | 0.00 | 0.25 | 7.706 | 7.706 | 1998 | 1998 |
| qa_2_350_len32279.json | validation.jsonl (qa_2) | 32279 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.444 | 8.444 | Current Mood | Current Mood |
| qa_2_351_len28661.json | validation.jsonl (qa_2) | 28661 | CacheBlend | 0.00 | 0.00 | 0.73 | 7.215 | 7.215 | Edward Albee | Edward Albee |
| qa_2_352_len32240.json | validation.jsonl (qa_2) | 32240 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.341 | 8.341 | Yes | Yes |
| qa_2_353_len31532.json | validation.jsonl (qa_2) | 31532 | CacheBlend | 0.00 | 1.00 | 0.09 | 8.945 | 8.945 | Vanessa Bley is not an impresario. The d | Vanessa Bley is not an impresario. The d |
| qa_2_354_len31734.json | validation.jsonl (qa_2) | 31734 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.189 | 8.189 | No | No |
| qa_2_355_len32062.json | validation.jsonl (qa_2) | 32062 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.291 | 8.291 | No | No |
| qa_2_356_len31977.json | validation.jsonl (qa_2) | 31977 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.302 | 8.302 | Oxford | Oxford |
| qa_2_357_len31733.json | validation.jsonl (qa_2) | 31733 | CacheBlend | 0.00 | 1.00 | 0.33 | 8.329 | 8.329 | J. G. Farrell | J. G. Farrell |
| qa_2_358_len31384.json | validation.jsonl (qa_2) | 31384 | CacheBlend | 0.00 | 1.00 | 0.40 | 8.135 | 8.135 | New York City | New York City |
| qa_2_359_len30549.json | validation.jsonl (qa_2) | 30549 | CacheBlend | 0.00 | 1.00 | 0.67 | 7.918 | 7.918 | Phil Lord and Christopher Miller | Phil Lord and Christopher Miller |
| qa_2_360_len30657.json | validation.jsonl (qa_2) | 30657 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.910 | 7.910 | Buckingham Palace | Buckingham Palace |
| qa_2_361_len32689.json | validation.jsonl (qa_2) | 32689 | CacheBlend | 0.00 | 1.00 | 0.18 | 9.108 | 9.108 | Scott Sandelin grew up in Hibbing, Minne | Scott Sandelin grew up in Hibbing, Minne |
| qa_2_362_len30847.json | validation.jsonl (qa_2) | 30847 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.924 | 7.924 | Phil Collins | Phil Collins |
| qa_2_363_len30487.json | validation.jsonl (qa_2) | 30487 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.803 | 7.803 | video game | video game |
| qa_2_364_len31579.json | validation.jsonl (qa_2) | 31579 | CacheBlend | 0.00 | 1.00 | 0.33 | 8.430 | 8.430 | Edmund Sharpe graduated from Cambridge U | Edmund Sharpe graduated from Cambridge U |
| qa_2_365_len31586.json | validation.jsonl (qa_2) | 31586 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.183 | 8.183 | Love Streams | Love Streams |
| qa_2_366_len32034.json | validation.jsonl (qa_2) | 32034 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.359 | 8.359 | Lowe's | Lowe's |
| qa_2_367_len30901.json | validation.jsonl (qa_2) | 30901 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.916 | 7.916 | second | second |
| qa_2_368_len30200.json | validation.jsonl (qa_2) | 30200 | CacheBlend | 0.00 | 0.00 | 0.75 | 7.757 | 7.757 | 1988 | 1988 |
| qa_2_369_len30400.json | validation.jsonl (qa_2) | 30400 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.797 | 7.797 | James Packer | James Packer |
| qa_2_370_len31597.json | validation.jsonl (qa_2) | 31597 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.183 | 8.183 | documentary | documentary |
| qa_2_371_len32331.json | validation.jsonl (qa_2) | 32331 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.412 | 8.412 | England | England |
| qa_2_372_len32423.json | validation.jsonl (qa_2) | 32423 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.516 | 8.516 | Yunho | Yunho |
| qa_2_373_len30769.json | validation.jsonl (qa_2) | 30769 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.933 | 7.933 | Hot toddy | Hot toddy |
| qa_2_374_len32178.json | validation.jsonl (qa_2) | 32178 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.433 | 8.433 | 2017 | 2017 |
| qa_2_375_len32192.json | validation.jsonl (qa_2) | 32192 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.363 | 8.363 | Gay Byrne | Gay Byrne |
| qa_2_376_len31632.json | validation.jsonl (qa_2) | 31632 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.427 | 8.427 | Hudson Bay Mining and Smelting Company | Hudson Bay Mining and Smelting Company |
| qa_2_377_len31459.json | validation.jsonl (qa_2) | 31459 | CacheBlend | 0.00 | 0.00 | 0.10 | 8.705 | 8.705 | Jens Risom introduced Mid-Century modern | Jens Risom introduced Mid-Century modern |
| qa_2_378_len30182.json | validation.jsonl (qa_2) | 30182 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.687 | 7.687 | Constantine | Constantine |
| qa_2_379_len32018.json | validation.jsonl (qa_2) | 32018 | CacheBlend | 0.00 | 0.00 | 0.09 | 10.560 | 10.560 | The Houston Astrodome is often referred  | The Houston Astrodome is often referred  |
| qa_2_380_len30800.json | validation.jsonl (qa_2) | 30800 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.283 | 8.283 | José Echegaray y Eizaguirre | José Echegaray y Eizaguirre |
| qa_2_381_len32400.json | validation.jsonl (qa_2) | 32400 | CacheBlend | 0.00 | 1.00 | 0.20 | 9.400 | 9.400 | Doncaster Belles played against Knowsley | Doncaster Belles played against Knowsley |
| qa_2_382_len31593.json | validation.jsonl (qa_2) | 31593 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.217 | 8.217 | Atomic Kitten | Atomic Kitten |
| qa_2_383_len32156.json | validation.jsonl (qa_2) | 32156 | CacheBlend | 0.00 | 0.00 | 0.31 | 8.536 | 8.536 | Elmwood Historic District–East | Elmwood Historic District–East |
| qa_2_384_len31426.json | validation.jsonl (qa_2) | 31426 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.168 | 8.168 | Philip K. Dick | Philip K. Dick |
| qa_2_385_len30290.json | validation.jsonl (qa_2) | 30290 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.744 | 7.744 | New York Giants | New York Giants |
| qa_2_386_len32496.json | validation.jsonl (qa_2) | 32496 | CacheBlend | 0.00 | 0.00 | 0.75 | 8.562 | 8.562 | 1960 | 1960 |
| qa_2_387_len30670.json | validation.jsonl (qa_2) | 30670 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.837 | 7.837 | Copenhagen | Copenhagen |
| qa_2_388_len29211.json | validation.jsonl (qa_2) | 29211 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.344 | 7.344 | No | No |
| qa_2_389_len32586.json | validation.jsonl (qa_2) | 32586 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.585 | 8.585 | Soha Ali Khan | Soha Ali Khan |
| qa_2_390_len31363.json | validation.jsonl (qa_2) | 31363 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.078 | 8.078 | horror | horror |
| qa_2_391_len32544.json | validation.jsonl (qa_2) | 32544 | CacheBlend | 0.00 | 0.00 | 0.83 | 8.624 | 8.624 | Lucas Grabeel | Lucas Grabeel |
| qa_2_392_len32501.json | validation.jsonl (qa_2) | 32501 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.492 | 8.492 | Rings | Rings |
| qa_2_393_len32624.json | validation.jsonl (qa_2) | 32624 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.494 | 8.494 | Kent | Kent |
| qa_2_394_len30477.json | validation.jsonl (qa_2) | 30477 | CacheBlend | 0.00 | 0.00 | 0.08 | 8.567 | 8.567 | Sir Peter Reginald Frederick Hall CBE wr | Sir Peter Reginald Frederick Hall CBE wr |
| qa_2_395_len32099.json | validation.jsonl (qa_2) | 32099 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.338 | 8.338 | French Open | French Open |
| qa_2_396_len32258.json | validation.jsonl (qa_2) | 32258 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.461 | 8.461 | Bismarck | Bismarck |
| qa_2_397_len32712.json | validation.jsonl (qa_2) | 32712 | CacheBlend | 0.00 | 1.00 | 0.25 | 8.567 | 8.567 | Manchester United | Manchester United |
| qa_2_398_len32012.json | validation.jsonl (qa_2) | 32012 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.981 | 8.981 | Snowdrop is a proprietary game engine fi | Snowdrop is a proprietary game engine fi |
| qa_2_399_len31813.json | validation.jsonl (qa_2) | 31813 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.285 | 8.285 | YouTube Rewind | YouTube Rewind |
| qa_2_400_len30106.json | validation.jsonl (qa_2) | 30106 | CacheBlend | 0.00 | 1.00 | 0.62 | 7.917 | 7.917 | King George IV and the Duke of Wellingto | King George IV and the Duke of Wellingto |
| qa_2_401_len30823.json | validation.jsonl (qa_2) | 30823 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.951 | 7.951 | Icehouse pieces | Icehouse pieces |
| qa_2_402_len30339.json | validation.jsonl (qa_2) | 30339 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.708 | 7.708 | ZZ Top | ZZ Top |
| qa_2_403_len31037.json | validation.jsonl (qa_2) | 31037 | CacheBlend | 0.00 | 0.00 | 0.75 | 8.038 | 8.038 | 1973 | 1973 |
| qa_2_404_len30171.json | validation.jsonl (qa_2) | 30171 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.174 | 8.174 | The Jeep Cherokee was originally sold as | The Jeep Cherokee was originally sold as |
| qa_2_405_len32716.json | validation.jsonl (qa_2) | 32716 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.632 | 8.632 | Leah Crocetto | Leah Crocetto |
| qa_2_406_len32458.json | validation.jsonl (qa_2) | 32458 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.428 | 8.428 | Gallery | Gallery |
| qa_2_407_len31436.json | validation.jsonl (qa_2) | 31436 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.114 | 8.114 | Langston Hughes | Langston Hughes |
| qa_2_408_len31901.json | validation.jsonl (qa_2) | 31901 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.119 | 9.119 | KGOT's studios are located in the Dimond | KGOT's studios are located in the Dimond |
| qa_2_409_len31811.json | validation.jsonl (qa_2) | 31811 | CacheBlend | 0.00 | 0.00 | 0.14 | 8.299 | 8.299 | Bad Meets Evil | Bad Meets Evil |
| qa_2_410_len29792.json | validation.jsonl (qa_2) | 29792 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.548 | 7.548 | Shanghai | Shanghai |
| qa_2_411_len31966.json | validation.jsonl (qa_2) | 31966 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.160 | 9.160 | Jacques Mathou starred in Delicatessen,  | Jacques Mathou starred in Delicatessen,  |
| qa_2_412_len32305.json | validation.jsonl (qa_2) | 32305 | CacheBlend | 0.00 | 0.00 | 0.67 | 8.625 | 8.625 | Lithuanian-French | Lithuanian-French |
| qa_2_413_len32251.json | validation.jsonl (qa_2) | 32251 | CacheBlend | 0.00 | 1.00 | 0.33 | 8.541 | 8.541 | the death of a family member | the death of a family member |
| qa_2_414_len31315.json | validation.jsonl (qa_2) | 31315 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.132 | 8.132 | Douglas Jackson | Douglas Jackson |
| qa_2_415_len32130.json | validation.jsonl (qa_2) | 32130 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.406 | 8.406 | Dirty Pretty Things | Dirty Pretty Things |
| qa_2_416_len31249.json | validation.jsonl (qa_2) | 31249 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.173 | 8.173 | Krusty the Clown | Krusty the Clown |
| qa_2_417_len30869.json | validation.jsonl (qa_2) | 30869 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.019 | 8.019 | Wilton Mall | Wilton Mall |
| qa_2_418_len31699.json | validation.jsonl (qa_2) | 31699 | CacheBlend | 0.00 | 1.00 | 0.57 | 8.348 | 8.348 | Chauncey Billups | Chauncey Billups |
| qa_2_419_len30680.json | validation.jsonl (qa_2) | 30680 | CacheBlend | 0.00 | 0.00 | 0.08 | 10.558 | 10.558 | The information provided does not includ | The information provided does not includ |
| qa_2_420_len31897.json | validation.jsonl (qa_2) | 31897 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.288 | 8.288 | No | No |
| qa_2_421_len32081.json | validation.jsonl (qa_2) | 32081 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.360 | 8.360 | documentary | documentary |
| qa_2_422_len30281.json | validation.jsonl (qa_2) | 30281 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.742 | 7.742 | 12 | 12 |
| qa_2_423_len31789.json | validation.jsonl (qa_2) | 31789 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.321 | 8.321 | defensive coordinator | defensive coordinator |
| qa_2_424_len31814.json | validation.jsonl (qa_2) | 31814 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.387 | 8.387 | 1.95 m | 1.95 m |
| qa_2_425_len32263.json | validation.jsonl (qa_2) | 32263 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.442 | 8.442 | MGM Resorts International | MGM Resorts International |
| qa_2_426_len32571.json | validation.jsonl (qa_2) | 32571 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.679 | 8.679 | Bedknobs and Broomsticks | Bedknobs and Broomsticks |
| qa_2_427_len30073.json | validation.jsonl (qa_2) | 30073 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.715 | 7.715 | The Holston River | The Holston River |
| qa_2_428_len31366.json | validation.jsonl (qa_2) | 31366 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.110 | 8.110 | Conservative Party | Conservative Party |
| qa_2_429_len30809.json | validation.jsonl (qa_2) | 30809 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.962 | 7.962 | Wes Archer | Wes Archer |
| qa_2_430_len30063.json | validation.jsonl (qa_2) | 30063 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.716 | 7.716 | Geelong Football Club | Geelong Football Club |
| qa_2_431_len31766.json | validation.jsonl (qa_2) | 31766 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.214 | 8.214 | Yes | Yes |
| qa_2_432_len31899.json | validation.jsonl (qa_2) | 31899 | CacheBlend | 0.00 | 0.00 | 0.80 | 8.353 | 8.353 | Delmer Daves | Delmer Daves |
| qa_2_433_len30921.json | validation.jsonl (qa_2) | 30921 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.014 | 8.014 | Test Icicles | Test Icicles |
| qa_2_434_len31739.json | validation.jsonl (qa_2) | 31739 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.287 | 8.287 | Walt Disney Productions | Walt Disney Productions |
| qa_2_435_len31935.json | validation.jsonl (qa_2) | 31935 | CacheBlend | 0.00 | 0.00 | 0.07 | 10.340 | 10.340 | The Jemez River, which is 7.51 km long,  | The Jemez River, which is 7.51 km long,  |
| qa_2_436_len29289.json | validation.jsonl (qa_2) | 29289 | CacheBlend | 0.00 | 0.00 | 0.31 | 7.618 | 7.618 | Olathe, Kansas, United States | Olathe, Kansas, United States |
| qa_2_437_len31634.json | validation.jsonl (qa_2) | 31634 | CacheBlend | 0.00 | 1.00 | 0.03 | 10.707 | 10.707 | The question asks for the boxer ranked s | The question asks for the boxer ranked s |
| qa_2_438_len31155.json | validation.jsonl (qa_2) | 31155 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.093 | 8.093 | 1909 | 1909 |
| qa_2_439_len31826.json | validation.jsonl (qa_2) | 31826 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.316 | 8.316 | Sun Valley, Idaho | Sun Valley, Idaho |
| qa_2_440_len30894.json | validation.jsonl (qa_2) | 30894 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.048 | 8.048 | President of the United States | President of the United States |
| qa_2_441_len32504.json | validation.jsonl (qa_2) | 32504 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.556 | 8.556 | The Bears and I | The Bears and I |
| qa_2_442_len32307.json | validation.jsonl (qa_2) | 32307 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.507 | 8.507 | Patterns of Sexual Behavior | Patterns of Sexual Behavior |
| qa_2_443_len30994.json | validation.jsonl (qa_2) | 30994 | CacheBlend | 0.00 | 0.00 | 0.80 | 7.962 | 7.962 | Larry Drake | Larry Drake |
| qa_2_444_len31171.json | validation.jsonl (qa_2) | 31171 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.191 | 8.191 | Bedknobs and Broomsticks | Bedknobs and Broomsticks |
| qa_2_445_len32294.json | validation.jsonl (qa_2) | 32294 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.808 | 8.808 | 46 Albert Square, Walford, London E20 | 46 Albert Square, Walford, London E20 |
| qa_2_446_len30741.json | validation.jsonl (qa_2) | 30741 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.927 | 7.927 | Betty Suarez | Betty Suarez |
| qa_2_447_len30964.json | validation.jsonl (qa_2) | 30964 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.960 | 7.960 | Phoenix Television | Phoenix Television |
| qa_2_448_len32129.json | validation.jsonl (qa_2) | 32129 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.334 | 8.334 | Planet Terror | Planet Terror |
| qa_2_449_len31097.json | validation.jsonl (qa_2) | 31097 | CacheBlend | 0.00 | 0.00 | 0.04 | 12.126 | 12.126 | Chimaji Appa served as the younger broth | Chimaji Appa served as the younger broth |
| qa_2_450_len32073.json | validation.jsonl (qa_2) | 32073 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.389 | 8.389 | Donald Trump Jr. | Donald Trump Jr. |
| qa_2_451_len31098.json | validation.jsonl (qa_2) | 31098 | CacheBlend | 0.00 | 1.00 | 0.44 | 8.027 | 8.027 | Lord Dunmore | Lord Dunmore |
| qa_2_452_len31288.json | validation.jsonl (qa_2) | 31288 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.095 | 8.095 | Graduados | Graduados |
| qa_2_453_len31527.json | validation.jsonl (qa_2) | 31527 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.236 | 8.236 | Private Secretary and Treasurer | Private Secretary and Treasurer |
| qa_2_454_len32323.json | validation.jsonl (qa_2) | 32323 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.408 | 8.408 | No | No |
| qa_2_455_len32629.json | validation.jsonl (qa_2) | 32629 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.662 | 8.662 | Ogallala Aquifer | Ogallala Aquifer |
| qa_2_456_len30405.json | validation.jsonl (qa_2) | 30405 | CacheBlend | 0.00 | 0.00 | 0.40 | 7.830 | 7.830 | Ibn Arabi | Ibn Arabi |
| qa_2_457_len30636.json | validation.jsonl (qa_2) | 30636 | CacheBlend | 0.00 | 0.00 | 0.03 | 10.378 | 10.378 | The goddess associated with Frigg, possi | The goddess associated with Frigg, possi |
| qa_2_458_len30336.json | validation.jsonl (qa_2) | 30336 | CacheBlend | 0.00 | 1.00 | 0.40 | 7.751 | 7.751 | South Asian Games | South Asian Games |
| qa_2_459_len32590.json | validation.jsonl (qa_2) | 32590 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.579 | 8.579 | Viscount Cranborne | Viscount Cranborne |
| qa_2_460_len31300.json | validation.jsonl (qa_2) | 31300 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.129 | 8.129 | Kansas City crime family | Kansas City crime family |
| qa_2_461_len32396.json | validation.jsonl (qa_2) | 32396 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.453 | 8.453 | Albert Park | Albert Park |
| qa_2_462_len32626.json | validation.jsonl (qa_2) | 32626 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.591 | 8.591 | 2003 | 2003 |
| qa_2_463_len32603.json | validation.jsonl (qa_2) | 32603 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.559 | 8.559 | Unbreakable | Unbreakable |
| qa_2_464_len32563.json | validation.jsonl (qa_2) | 32563 | CacheBlend | 0.00 | 1.00 | 0.20 | 9.278 | 9.278 | Raymond Ochoa voices Arlo, the Apatosaur | Raymond Ochoa voices Arlo, the Apatosaur |
| qa_2_465_len32400.json | validation.jsonl (qa_2) | 32400 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.689 | 8.689 | COPS (Central Organization of Police Spe | COPS (Central Organization of Police Spe |
| qa_2_466_len32409.json | validation.jsonl (qa_2) | 32409 | CacheBlend | 0.00 | 1.00 | 0.00 | 10.413 | 10.413 | Gaja in Sanskrit means Elephant. 'Brisht | Gaja in Sanskrit means Elephant. 'Brisht |
| qa_2_467_len31961.json | validation.jsonl (qa_2) | 31961 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.332 | 8.332 | Battle of Guam | Battle of Guam |
| qa_2_468_len32109.json | validation.jsonl (qa_2) | 32109 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.403 | 8.403 | Catherine Cyran | Catherine Cyran |
| qa_2_469_len31172.json | validation.jsonl (qa_2) | 31172 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.074 | 8.074 | Spin Out | Spin Out |
| qa_2_470_len31027.json | validation.jsonl (qa_2) | 31027 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.983 | 7.983 | film | film |
| qa_2_471_len31395.json | validation.jsonl (qa_2) | 31395 | CacheBlend | 0.00 | 1.00 | 0.75 | 8.167 | 8.167 | executive search | executive search |
| qa_2_472_len31297.json | validation.jsonl (qa_2) | 31297 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.066 | 8.066 | author | author |
| qa_2_473_len32728.json | validation.jsonl (qa_2) | 32728 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.700 | 8.700 | Catholic Church in Ireland | Catholic Church in Ireland |
| qa_2_474_len30988.json | validation.jsonl (qa_2) | 30988 | CacheBlend | 0.00 | 1.00 | 0.32 | 8.602 | 8.602 | Bridge to Terabithia (1985 film) was sho | Bridge to Terabithia (1985 film) was sho |
| qa_2_475_len32191.json | validation.jsonl (qa_2) | 32191 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.395 | 8.395 | Joe Scarborough | Joe Scarborough |
| qa_2_476_len32476.json | validation.jsonl (qa_2) | 32476 | CacheBlend | 0.00 | 1.00 | 0.73 | 8.596 | 8.596 | NCAA Division I | NCAA Division I |
| qa_2_477_len31917.json | validation.jsonl (qa_2) | 31917 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.394 | 8.394 | Tea Moderna | Tea Moderna |
| qa_2_478_len31602.json | validation.jsonl (qa_2) | 31602 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.218 | 8.218 | United States | United States |
| qa_2_479_len31352.json | validation.jsonl (qa_2) | 31352 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.209 | 8.209 | Kathleen O'Brien | Kathleen O'Brien |
| qa_2_480_len30469.json | validation.jsonl (qa_2) | 30469 | CacheBlend | 0.00 | 1.00 | 0.89 | 7.906 | 7.906 | Warner Bros. Records | Warner Bros. Records |
| qa_2_481_len32538.json | validation.jsonl (qa_2) | 32538 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.585 | 8.585 | Darkthrone | Darkthrone |
| qa_2_482_len32153.json | validation.jsonl (qa_2) | 32153 | CacheBlend | 0.00 | 0.00 | 0.37 | 9.435 | 9.435 | Emilie du Châtelet translated and commen | Emilie du Châtelet translated and commen |
| qa_2_483_len32469.json | validation.jsonl (qa_2) | 32469 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.455 | 8.455 | No | No |
| qa_2_484_len30121.json | validation.jsonl (qa_2) | 30121 | CacheBlend | 0.00 | 1.00 | 0.19 | 8.614 | 8.614 | The Detroit mayoral elections of 2009 to | The Detroit mayoral elections of 2009 to |
| qa_2_485_len31805.json | validation.jsonl (qa_2) | 31805 | CacheBlend | 0.00 | 0.00 | 0.50 | 8.318 | 8.318 | 2013 | 2013 |
| qa_2_486_len29655.json | validation.jsonl (qa_2) | 29655 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.488 | 7.488 | rock | rock |
| qa_2_487_len28811.json | validation.jsonl (qa_2) | 28811 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.250 | 7.250 | Portal 2 | Portal 2 |
| qa_2_488_len32737.json | validation.jsonl (qa_2) | 32737 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.534 | 8.534 | Aldosterone | Aldosterone |
| qa_2_489_len30469.json | validation.jsonl (qa_2) | 30469 | CacheBlend | 0.00 | 1.00 | 0.50 | 7.905 | 7.905 | 33 or 34 | 33 or 34 |
| qa_2_490_len32711.json | validation.jsonl (qa_2) | 32711 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.753 | 8.753 | Berghuis v. Thompkins | Berghuis v. Thompkins |
| qa_2_491_len32494.json | validation.jsonl (qa_2) | 32494 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.526 | 8.526 | Stacey Kent | Stacey Kent |
| qa_2_492_len31992.json | validation.jsonl (qa_2) | 31992 | CacheBlend | 0.00 | 0.00 | 0.75 | 8.365 | 8.365 | 1969 | 1969 |
| qa_2_493_len32699.json | validation.jsonl (qa_2) | 32699 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.615 | 8.615 | IndyCar Series | IndyCar Series |
| qa_2_494_len31659.json | validation.jsonl (qa_2) | 31659 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.272 | 8.272 | 2002 | 2002 |
| qa_2_495_len32069.json | validation.jsonl (qa_2) | 32069 | CacheBlend | 0.00 | 0.00 | 0.33 | 8.319 | 8.319 | 24 | 24 |
| qa_2_496_len32405.json | validation.jsonl (qa_2) | 32405 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.476 | 8.476 | Newark | Newark |
| qa_2_497_len31175.json | validation.jsonl (qa_2) | 31175 | CacheBlend | 0.00 | 1.00 | 0.26 | 9.021 | 9.021 | The 2017 US Open was the 137th edition o | The 2017 US Open was the 137th edition o |
| qa_2_498_len30886.json | validation.jsonl (qa_2) | 30886 | CacheBlend | 0.00 | 0.00 | 0.80 | 7.982 | 7.982 | Ronald Reagan | Ronald Reagan |
| qa_2_499_len31501.json | validation.jsonl (qa_2) | 31501 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.111 | 8.111 | documentary | documentary |

## 二、按方法聚合（每轮一行，再给出跨轮中位数）

| 方法 | 轮次 | EM | SM | F1 | TTFT缓存均值(s) | TTFT全量均值(s) | 缓存加速比 |
|---|---|---|---|---|---|------|---|
| CacheBlend | r1 | 0.400 | 0.580 | 0.552 | 8.270 | 8.270 | 1.00× |
| CacheBlend | r2 | 0.350 | 0.530 | 0.496 | 8.485 | 8.485 | 1.00× |
| CacheBlend | r3 | 0.330 | 0.530 | 0.503 | 8.349 | 8.349 | 1.00× |
| CacheBlend | r4 | 0.360 | 0.580 | 0.507 | 8.426 | 8.426 | 1.00× |
| CacheBlend | r5 | 0.440 | 0.600 | 0.574 | 8.435 | 8.435 | 1.00× |
| **CacheBlend** | **中位** | **0.360** | **0.580** | **0.507** | **8.426** | **8.426** | **1.00×** |
| vLLM全量 | r1 | 0.400 | 0.580 | 0.552 | - | 8.270 | 1.00× |
| vLLM全量 | r2 | 0.350 | 0.530 | 0.496 | - | 8.485 | 1.00× |
| vLLM全量 | r3 | 0.330 | 0.530 | 0.503 | - | 8.349 | 1.00× |
| vLLM全量 | r4 | 0.360 | 0.580 | 0.507 | - | 8.426 | 1.00× |
| vLLM全量 | r5 | 0.440 | 0.600 | 0.574 | - | 8.435 | 1.00× |
| **vLLM全量** | **中位** | **0.360** | **0.580** | **0.507** | - | **8.426** | **1.00×** |

## 三、按数据集×方法聚合（跨轮中位数；QA 数据集看 SM/F1，samsum 摘要看 F1）

| 数据集 | 样本数 | 方法 | EM中位 | SM中位 | F1中位 | TTFT缓存中位(s) |
|---|---|---|---|---|---|---|
| validation.jsonl (qa_2) | 500 | CacheBlend | 0.000 | 1.000 | 0.571 | 8.337 |
