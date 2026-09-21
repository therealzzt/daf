# RULER qa_1/qa_2（Qwen2.5，32768 档）样本转换报告

- 分词器：`Qwen/Qwen2.5-7B-Instruct`
- 数据来源：`experiments/ruler_src/data_gen_500/{qa_1,qa_2}/validation.jsonl`（RULER 官方 qa.py 生成，--max_seq_length 32768，seed 42，各 500 条）
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
| qa_1_024_len31374.json | qa_1 | 24 | 183 | 31374 | Norman mercenary |
| qa_1_025_len30462.json | qa_1 | 25 | 183 | 30462 | Robert Guiscard |
| qa_1_026_len32340.json | qa_1 | 26 | 193 | 32340 | 1082 |
| qa_1_027_len30910.json | qa_1 | 27 | 183 | 30910 | 30,000 |
| qa_1_028_len32017.json | qa_1 | 28 | 183 | 32017 | Deabolis |
| qa_1_029_len31291.json | qa_1 | 29 | 183 | 31291 | Bohemond |
| qa_1_030_len31310.json | qa_1 | 30 | 193 | 31310 | Deabolis |
| qa_1_031_len32496.json | qa_1 | 31 | 193 | 32496 | 1185 |
| qa_1_032_len32245.json | qa_1 | 32 | 193 | 32245 | Dyrrachium |
| qa_1_033_len30549.json | qa_1 | 33 | 183 | 30549 | the Adriatic |
| qa_1_034_len31855.json | qa_1 | 34 | 183 | 31855 | King Ethelred II |
| qa_1_035_len32730.json | qa_1 | 35 | 193 | 32730 | Duke Richard II |
| qa_1_036_len30672.json | qa_1 | 36 | 183 | 30672 | Normandy |
| qa_1_037_len32088.json | qa_1 | 37 | 183 | 32088 | Sweyn Forkbeard |
| qa_1_038_len31185.json | qa_1 | 38 | 183 | 31185 | Harthacnut |
| qa_1_039_len30640.json | qa_1 | 39 | 193 | 30640 | 1041 |
| qa_1_040_len31793.json | qa_1 | 40 | 183 | 31793 | Robert of Jumièges |
| qa_1_041_len32112.json | qa_1 | 41 | 183 | 32112 | Battle of Hastings |
| qa_1_042_len28768.json | qa_1 | 42 | 183 | 28768 | William II |
| qa_1_043_len29775.json | qa_1 | 43 | 183 | 29775 | 1066 |
| qa_1_044_len32613.json | qa_1 | 44 | 193 | 32613 | Anglo-Saxons |
| qa_1_045_len32597.json | qa_1 | 45 | 193 | 32597 | Modern English |
| qa_1_046_len31139.json | qa_1 | 46 | 183 | 31139 | 1169 |
| qa_1_047_len32728.json | qa_1 | 47 | 193 | 32728 | Ireland |
| qa_1_048_len32119.json | qa_1 | 48 | 193 | 32119 | Irish |
| qa_1_049_len31169.json | qa_1 | 49 | 183 | 31169 | Edgar |
| qa_1_050_len32145.json | qa_1 | 50 | 193 | 32145 | King Malcolm III of Scotland |
| qa_1_051_len32137.json | qa_1 | 51 | 193 | 32137 | 1072 |
| qa_1_052_len29728.json | qa_1 | 52 | 173 | 29728 | Duncan |
| qa_1_053_len30282.json | qa_1 | 53 | 183 | 30282 | Sybilla of Normandy |
| qa_1_054_len30042.json | qa_1 | 54 | 173 | 30042 | Norman |
| qa_1_055_len30991.json | qa_1 | 55 | 193 | 30991 | Hereford |
| qa_1_056_len31411.json | qa_1 | 56 | 183 | 31411 | the Welsh |
| qa_1_057_len29615.json | qa_1 | 57 | 173 | 29615 | Edward the Confessor |
| qa_1_058_len32630.json | qa_1 | 58 | 183 | 32630 | Wales |
| qa_1_059_len32398.json | qa_1 | 59 | 183 | 32398 | 1018 |
| qa_1_060_len32336.json | qa_1 | 60 | 183 | 32336 | William of Montreuil |
| qa_1_061_len30969.json | qa_1 | 61 | 183 | 30969 | 1097 |
| qa_1_062_len29765.json | qa_1 | 62 | 183 | 29765 | Tancred |
| qa_1_063_len31774.json | qa_1 | 63 | 193 | 31774 | Jerusalem |
| qa_1_064_len32621.json | qa_1 | 64 | 183 | 32621 | 380 years |
| qa_1_065_len31954.json | qa_1 | 65 | 183 | 31954 | a storm |
| qa_1_066_len32699.json | qa_1 | 66 | 193 | 32699 | Berengaria |
| qa_1_067_len32231.json | qa_1 | 67 | 193 | 32231 | 1191 |
| qa_1_068_len31186.json | qa_1 | 68 | 193 | 31186 | Isaac Komnenos |
| qa_1_069_len31696.json | qa_1 | 69 | 193 | 31696 | Conrad of Montferrat |
| qa_1_070_len31505.json | qa_1 | 70 | 183 | 31505 | silver |
| qa_1_071_len31228.json | qa_1 | 71 | 193 | 31228 | Guy de Lusignan |
| qa_1_072_len32173.json | qa_1 | 72 | 183 | 32173 | Africa |
| qa_1_073_len31989.json | qa_1 | 73 | 183 | 31989 | Bethencourt |
| qa_1_074_len30148.json | qa_1 | 74 | 183 | 30148 | Enrique Pérez de Guzmán |
| qa_1_075_len32155.json | qa_1 | 75 | 193 | 32155 | Maciot de Bethencourt |
| qa_1_076_len31246.json | qa_1 | 76 | 183 | 31246 | Channel Islands |
| qa_1_077_len32170.json | qa_1 | 77 | 193 | 32170 | two |
| qa_1_078_len31327.json | qa_1 | 78 | 183 | 31327 | Romanesque |
| qa_1_079_len30646.json | qa_1 | 79 | 183 | 30646 | rounded |
| qa_1_080_len30668.json | qa_1 | 80 | 183 | 30668 | Early Gothic |
| qa_1_081_len32397.json | qa_1 | 81 | 193 | 32397 | Anglo-Saxon |
| qa_1_082_len29758.json | qa_1 | 82 | 183 | 29758 | Sicily |
| qa_1_083_len30969.json | qa_1 | 83 | 183 | 30969 | early 11th century |
| qa_1_084_len30551.json | qa_1 | 84 | 183 | 30551 | dukes |
| qa_1_085_len32000.json | qa_1 | 85 | 193 | 32000 | 16th century |
| qa_1_086_len31039.json | qa_1 | 86 | 183 | 31039 | embroidery |
| qa_1_087_len28340.json | qa_1 | 87 | 173 | 28340 | Bayeux Tapestry |
| qa_1_088_len31102.json | qa_1 | 88 | 183 | 31102 | Odo |
| qa_1_089_len31470.json | qa_1 | 89 | 183 | 31470 | mosaics |
| qa_1_090_len30235.json | qa_1 | 90 | 183 | 30235 | 11th |
| qa_1_091_len31074.json | qa_1 | 91 | 183 | 31074 | William of Volpiano and John of Ravenna |
| qa_1_092_len31265.json | qa_1 | 92 | 193 | 31265 | southern Italy |
| qa_1_093_len32404.json | qa_1 | 93 | 193 | 32404 | Latin monastery at Sant'Eufemia. |
| qa_1_094_len30691.json | qa_1 | 94 | 183 | 30691 | Robert Guiscard |
| qa_1_095_len30985.json | qa_1 | 95 | 183 | 30985 | singing |
| qa_1_096_len31883.json | qa_1 | 96 | 193 | 31883 | Computational complexity theory |
| qa_1_097_len31198.json | qa_1 | 97 | 193 | 31198 | inherent difficulty |
| qa_1_098_len32639.json | qa_1 | 98 | 193 | 32639 | computational problems |
| qa_1_099_len30722.json | qa_1 | 99 | 193 | 30722 | if its solution requires significant resources |
| qa_1_100_len30867.json | qa_1 | 100 | 193 | 30867 | mathematical models of computation |
| qa_1_101_len32721.json | qa_1 | 101 | 193 | 32721 | time and storage |
| qa_1_102_len32234.json | qa_1 | 102 | 193 | 32234 | number of gates in a circuit |
| qa_1_103_len31221.json | qa_1 | 103 | 193 | 31221 | determine the practical limits on what computers can and can |
| qa_1_104_len29976.json | qa_1 | 104 | 193 | 29976 | analysis of algorithms and computability theory |
| qa_1_105_len32495.json | qa_1 | 105 | 193 | 32495 | analysis of algorithms |
| qa_1_106_len31241.json | qa_1 | 106 | 193 | 31241 | computational complexity theory |
| qa_1_107_len30414.json | qa_1 | 107 | 193 | 30414 | computability theory |
| qa_1_108_len29609.json | qa_1 | 108 | 193 | 29609 | problem instance |
| qa_1_109_len30615.json | qa_1 | 109 | 193 | 30615 | the problem |
| qa_1_110_len32263.json | qa_1 | 110 | 193 | 32263 | concrete |
| qa_1_111_len32402.json | qa_1 | 111 | 193 | 32402 | instances |
| qa_1_112_len30238.json | qa_1 | 112 | 193 | 30238 | solution |
| qa_1_113_len32383.json | qa_1 | 113 | 193 | 32383 | 2000 |
| qa_1_114_len31725.json | qa_1 | 114 | 193 | 31725 | round trip through all sites in Milan |
| qa_1_115_len31106.json | qa_1 | 115 | 193 | 31106 | computational problems |
| qa_1_116_len31489.json | qa_1 | 116 | 193 | 31489 | problem instance |
| qa_1_117_len32334.json | qa_1 | 117 | 193 | 32334 | binary alphabet |
| qa_1_118_len30653.json | qa_1 | 118 | 193 | 30653 | bitstrings |
| qa_1_119_len31885.json | qa_1 | 119 | 193 | 31885 | binary notation |
| qa_1_120_len30497.json | qa_1 | 120 | 193 | 30497 | adjacency matrices |
| qa_1_121_len30432.json | qa_1 | 121 | 193 | 30432 | Decision problems |
| qa_1_122_len29351.json | qa_1 | 122 | 183 | 29351 | yes or no |
| qa_1_123_len31399.json | qa_1 | 123 | 193 | 31399 | 1 or 0 |
| qa_1_124_len31034.json | qa_1 | 124 | 193 | 31034 | yes |
| qa_1_125_len32527.json | qa_1 | 125 | 193 | 32527 | yes |
| qa_1_126_len31350.json | qa_1 | 126 | 193 | 31350 | arbitrary graph |
| qa_1_127_len31881.json | qa_1 | 127 | 193 | 31881 | formal language |
| qa_1_128_len31494.json | qa_1 | 128 | 193 | 31494 | how graphs are encoded as binary strings |
| qa_1_129_len31707.json | qa_1 | 129 | 193 | 31707 | a computational problem |
| qa_1_130_len30722.json | qa_1 | 130 | 193 | 30722 | a single output |
| qa_1_131_len30888.json | qa_1 | 131 | 193 | 30888 | A function problem |
| qa_1_132_len31804.json | qa_1 | 132 | 193 | 31804 | the integer factorization problem |
| qa_1_133_len30956.json | qa_1 | 133 | 193 | 30956 | complex |
| qa_1_134_len29660.json | qa_1 | 134 | 183 | 29660 | decision problems |
| qa_1_135_len31906.json | qa_1 | 135 | 193 | 31906 | set of triples |
| qa_1_136_len30284.json | qa_1 | 136 | 193 | 30284 | how much time the best algorithm requires to solve the probl |
| qa_1_137_len30021.json | qa_1 | 137 | 183 | 30021 | the instance |
| qa_1_138_len31003.json | qa_1 | 138 | 193 | 31003 | as a function of the size of the instance |
| qa_1_139_len29185.json | qa_1 | 139 | 183 | 29185 | bits |
| qa_1_140_len32724.json | qa_1 | 140 | 193 | 32724 | an increase in the input size |
| qa_1_141_len30850.json | qa_1 | 141 | 193 | 30850 | Cobham's thesis |
| qa_1_142_len30924.json | qa_1 | 142 | 193 | 30924 | the time taken |
| qa_1_143_len30440.json | qa_1 | 143 | 193 | 30440 | worst-case time complexity |
| qa_1_144_len31849.json | qa_1 | 144 | 193 | 31849 | T(n) |
| qa_1_145_len31762.json | qa_1 | 145 | 193 | 31762 | polynomial time algorithm |
| qa_1_146_len31153.json | qa_1 | 146 | 183 | 31153 | A Turing machine |
| qa_1_147_len30808.json | qa_1 | 147 | 183 | 30808 | an algorithm |
| qa_1_148_len30919.json | qa_1 | 148 | 193 | 30919 | the Turing machine |
| qa_1_149_len29541.json | qa_1 | 149 | 193 | 29541 | symbols |
| qa_1_150_len31416.json | qa_1 | 150 | 193 | 31416 | A deterministic Turing machine |
| qa_1_151_len30560.json | qa_1 | 151 | 193 | 30560 | rules |
| qa_1_152_len31940.json | qa_1 | 152 | 193 | 31940 | A probabilistic Turing machine |
| qa_1_153_len31478.json | qa_1 | 153 | 193 | 31478 | A non-deterministic Turing machine |
| qa_1_154_len31973.json | qa_1 | 154 | 193 | 31973 | randomized algorithms |
| qa_1_155_len32311.json | qa_1 | 155 | 193 | 32311 | complexity classes |
| qa_1_156_len30908.json | qa_1 | 156 | 193 | 30908 | time or space |
| qa_1_157_len30931.json | qa_1 | 157 | 193 | 30931 | probabilistic Turing machines, non-deterministic Turing mach |
| qa_1_158_len30549.json | qa_1 | 158 | 193 | 30549 | random access machines |
| qa_1_159_len31675.json | qa_1 | 159 | 193 | 31675 | computational power |
| qa_1_160_len30636.json | qa_1 | 160 | 193 | 30636 | time and memory |
| qa_1_161_len31439.json | qa_1 | 161 | 193 | 31439 | the machines operate deterministically |
| qa_1_162_len31743.json | qa_1 | 162 | 193 | 31743 | non-deterministic |
| qa_1_163_len30939.json | qa_1 | 163 | 193 | 30939 | unusual resources |
| qa_1_164_len31356.json | qa_1 | 164 | 193 | 31356 | mathematical models |
| qa_1_165_len28460.json | qa_1 | 165 | 183 | 28460 | time |
| qa_1_166_len31388.json | qa_1 | 166 | 193 | 31388 | state transitions |
| qa_1_167_len30875.json | qa_1 | 167 | 193 | 30875 | difficulty |
| qa_1_168_len30998.json | qa_1 | 168 | 193 | 30998 | DTIME(f(n)) |
| qa_1_169_len31532.json | qa_1 | 169 | 193 | 31532 | time |
| qa_1_170_len30390.json | qa_1 | 170 | 183 | 30390 | complexity resources |
| qa_1_171_len31008.json | qa_1 | 171 | 193 | 31008 | computational resource |
| qa_1_172_len30865.json | qa_1 | 172 | 193 | 30865 | Blum complexity axioms |
| qa_1_173_len31870.json | qa_1 | 173 | 193 | 31870 | Complexity measures |
| qa_1_174_len31211.json | qa_1 | 174 | 193 | 31211 | Complexity measures |
| qa_1_175_len30506.json | qa_1 | 175 | 193 | 30506 | best, worst and average |
| qa_1_176_len31439.json | qa_1 | 176 | 193 | 31439 | complexity measure |
| qa_1_177_len31057.json | qa_1 | 177 | 183 | 31057 | time |
| qa_1_178_len32448.json | qa_1 | 178 | 193 | 32448 | inputs |
| qa_1_179_len29277.json | qa_1 | 179 | 193 | 29277 | deterministic sorting algorithm quicksort |
| qa_1_180_len30144.json | qa_1 | 180 | 193 | 30144 | worst-case |
| qa_1_181_len31234.json | qa_1 | 181 | 193 | 31234 | O(n2) |
| qa_1_182_len30516.json | qa_1 | 182 | 193 | 30516 | the most efficient algorithm |
| qa_1_183_len30110.json | qa_1 | 183 | 193 | 30110 | analysis of algorithms |
| qa_1_184_len31384.json | qa_1 | 184 | 193 | 31384 | lower bounds |
| qa_1_185_len31464.json | qa_1 | 185 | 193 | 31464 | upper bound |
| qa_1_186_len32249.json | qa_1 | 186 | 193 | 32249 | all possible algorithms |
| qa_1_187_len29855.json | qa_1 | 187 | 193 | 29855 | big O notation |
| qa_1_188_len31697.json | qa_1 | 188 | 193 | 31697 | constant factors and smaller terms |
| qa_1_189_len30490.json | qa_1 | 189 | 193 | 30490 | T(n) = O(n2) |
| qa_1_190_len31632.json | qa_1 | 190 | 193 | 31632 | the computational model |
| qa_1_191_len30933.json | qa_1 | 191 | 193 | 30933 | complexity classes |
| qa_1_192_len31924.json | qa_1 | 192 | 193 | 31924 | framework |
| qa_1_193_len30679.json | qa_1 | 193 | 193 | 30679 | complicated definitions |
| qa_1_194_len30507.json | qa_1 | 194 | 193 | 30507 | chosen machine model |
| qa_1_195_len32452.json | qa_1 | 195 | 193 | 32452 | linear time |
| qa_1_196_len31923.json | qa_1 | 196 | 193 | 31923 | single-tape Turing machines |
| qa_1_197_len31358.json | qa_1 | 197 | 193 | 31358 | Cobham-Edmonds thesis |
| qa_1_198_len30981.json | qa_1 | 198 | 193 | 30981 | complexity class P |
| qa_1_199_len31501.json | qa_1 | 199 | 193 | 31501 | time or space |
| qa_1_200_len30958.json | qa_1 | 200 | 193 | 30958 | bounding |
| qa_1_201_len29409.json | qa_1 | 201 | 183 | 29409 | complexity classes |
| qa_1_202_len29532.json | qa_1 | 202 | 193 | 29532 | BPP, ZPP and RP |
| qa_1_203_len28703.json | qa_1 | 203 | 183 | 28703 | Boolean |
| qa_1_204_len31871.json | qa_1 | 204 | 193 | 31871 | quantum |
| qa_1_205_len32732.json | qa_1 | 205 | 193 | 32732 | #P |
| qa_1_206_len31259.json | qa_1 | 206 | 193 | 31259 | Interactive |
| qa_1_207_len31512.json | qa_1 | 207 | 193 | 31512 | computation time |
| qa_1_208_len31177.json | qa_1 | 208 | 193 | 31177 | DTIME(n2) |
| qa_1_209_len30901.json | qa_1 | 209 | 193 | 30901 | time and space hierarchy theorems |
| qa_1_210_len31797.json | qa_1 | 210 | 193 | 31797 | a proper hierarchy on the classes defined |
| qa_1_211_len32719.json | qa_1 | 211 | 193 | 32719 | quantitative statements |
| qa_1_212_len32063.json | qa_1 | 212 | 193 | 32063 | time and space hierarchy theorems |
| qa_1_213_len30545.json | qa_1 | 213 | 193 | 30545 | EXPTIME |
| qa_1_214_len31652.json | qa_1 | 214 | 193 | 31652 | PSPACE |
| qa_1_215_len30827.json | qa_1 | 215 | 193 | 30827 | reduction |
| qa_1_216_len30674.json | qa_1 | 216 | 193 | 30674 | another problem |
| qa_1_217_len30986.json | qa_1 | 217 | 193 | 30986 | reduces |
| qa_1_218_len31722.json | qa_1 | 218 | 193 | 31722 | Karp reductions and Levin reductions |
| qa_1_219_len31630.json | qa_1 | 219 | 193 | 31630 | the bound on the complexity of reductions |
| qa_1_220_len31566.json | qa_1 | 220 | 193 | 31566 | polynomial-time reduction |
| qa_1_221_len32243.json | qa_1 | 221 | 193 | 32243 | multiplying two integers |
| qa_1_222_len30846.json | qa_1 | 222 | 193 | 30846 | polynomial time |
| qa_1_223_len31405.json | qa_1 | 223 | 193 | 31405 | input |
| qa_1_224_len31646.json | qa_1 | 224 | 193 | 31646 | multiplication |
| qa_1_225_len31833.json | qa_1 | 225 | 193 | 31833 | the type of reduction being used |
| qa_1_226_len31015.json | qa_1 | 226 | 193 | 31015 | if every problem in C can be reduced to X |
| qa_1_227_len31324.json | qa_1 | 227 | 193 | 31324 | solve any problem in C |
| qa_1_228_len30643.json | qa_1 | 228 | 193 | 30643 | NP-hard |
| qa_1_229_len31179.json | qa_1 | 229 | 193 | 31179 | NP-complete |
| qa_1_230_len30565.json | qa_1 | 230 | 193 | 30565 | NP |
| qa_1_231_len31371.json | qa_1 | 231 | 193 | 31371 | there is no known polynomial-time solution |
| qa_1_232_len32512.json | qa_1 | 232 | 193 | 32512 | NP |
| qa_1_233_len29898.json | qa_1 | 233 | 193 | 29898 | P |
| qa_1_234_len30965.json | qa_1 | 234 | 193 | 30965 | Cobham–Edmonds thesis |
| qa_1_235_len31067.json | qa_1 | 235 | 193 | 31067 | NP |
| qa_1_236_len32345.json | qa_1 | 236 | 193 | 32345 | Boolean satisfiability problem |
| qa_1_237_len31474.json | qa_1 | 237 | 193 | 31474 | Turing machines |
| qa_1_238_len29505.json | qa_1 | 238 | 193 | 29505 | more efficient solutions |
| qa_1_239_len31551.json | qa_1 | 239 | 193 | 31551 | protein structure prediction |
| qa_1_240_len30933.json | qa_1 | 240 | 193 | 30933 | $1,000,000 |
| qa_1_241_len31312.json | qa_1 | 241 | 193 | 31312 | Ladner |
| qa_1_242_len32560.json | qa_1 | 242 | 193 | 32560 | NP-intermediate problems |
| qa_1_243_len31183.json | qa_1 | 243 | 193 | 31183 | graph isomorphism problem |
| qa_1_244_len31226.json | qa_1 | 244 | 193 | 31226 | The graph isomorphism problem |
| qa_1_245_len31295.json | qa_1 | 245 | 193 | 31295 | NP-complete |
| qa_1_246_len32112.json | qa_1 | 246 | 193 | 32112 | polynomial time hierarchy |
| qa_1_247_len32297.json | qa_1 | 247 | 193 | 32297 | second level |
| qa_1_248_len31640.json | qa_1 | 248 | 193 | 31640 | Laszlo Babai and Eugene Luks |
| qa_1_249_len31866.json | qa_1 | 249 | 193 | 31866 | The integer factorization problem |
| qa_1_250_len31586.json | qa_1 | 250 | 193 | 31586 | k |
| qa_1_251_len28664.json | qa_1 | 251 | 183 | 28664 | modern cryptographic systems |
| qa_1_252_len31535.json | qa_1 | 252 | 193 | 31535 | the general number field sieve |
| qa_1_253_len31320.json | qa_1 | 253 | 193 | 31320 | suspected to be unequal |
| qa_1_254_len31413.json | qa_1 | 254 | 193 | 31413 | P ⊆ NP ⊆ PP ⊆ PSPACE |
| qa_1_255_len30881.json | qa_1 | 255 | 193 | 30881 | between P and PSPACE |
| qa_1_256_len31282.json | qa_1 | 256 | 193 | 31282 | Proving that any of these classes are unequal |
| qa_1_257_len32623.json | qa_1 | 257 | 193 | 32623 | co-NP |
| qa_1_258_len30674.json | qa_1 | 258 | 193 | 30674 | reversed |
| qa_1_259_len29117.json | qa_1 | 259 | 183 | 29117 | not equal |
| qa_1_260_len31335.json | qa_1 | 260 | 193 | 31335 | P is not equal to NP |
| qa_1_261_len31149.json | qa_1 | 261 | 193 | 31149 | L |
| qa_1_262_len31104.json | qa_1 | 262 | 193 | 31104 | strictly contained in P or equal to P |
| qa_1_263_len30795.json | qa_1 | 263 | 193 | 30795 | complexity classes |
| qa_1_264_len31684.json | qa_1 | 264 | 193 | 31684 | NL and NC |
| qa_1_265_len32121.json | qa_1 | 265 | 193 | 32121 | if they are distinct or equal classes |
| qa_1_266_len31937.json | qa_1 | 266 | 193 | 31937 | intractable problems |
| qa_1_267_len31146.json | qa_1 | 267 | 193 | 31146 | exponential-time algorithms |
| qa_1_268_len31853.json | qa_1 | 268 | 193 | 31853 | NP-complete problems |
| qa_1_269_len31835.json | qa_1 | 269 | 193 | 31835 | Presburger arithmetic |
| qa_1_270_len30737.json | qa_1 | 270 | 193 | 30737 | algorithms have been written |
| qa_1_271_len30881.json | qa_1 | 271 | 193 | 30881 | NP-complete knapsack problem |
| qa_1_272_len32130.json | qa_1 | 272 | 193 | 32130 | in less than quadratic time |
| qa_1_273_len30842.json | qa_1 | 273 | 193 | 30842 | NP-complete Boolean satisfiability problem |
| qa_1_274_len30868.json | qa_1 | 274 | 193 | 30868 | foundations were laid out |
| qa_1_275_len32013.json | qa_1 | 275 | 193 | 32013 | Alan Turing |
| qa_1_276_len31867.json | qa_1 | 276 | 193 | 31867 | Turing machines |
| qa_1_277_len31901.json | qa_1 | 277 | 193 | 31901 | 1936 |
| qa_1_278_len29925.json | qa_1 | 278 | 183 | 29925 | a computer |
| qa_1_279_len31001.json | qa_1 | 279 | 193 | 31001 | On the Computational Complexity of Algorithms |
| qa_1_280_len30550.json | qa_1 | 280 | 193 | 30550 | Juris Hartmanis and Richard Stearns |
| qa_1_281_len29974.json | qa_1 | 281 | 183 | 29974 | 1965 |
| qa_1_282_len31053.json | qa_1 | 282 | 193 | 31053 | time and space |
| qa_1_283_len30808.json | qa_1 | 283 | 193 | 30808 | 1965 |
| qa_1_284_len31855.json | qa_1 | 284 | 193 | 31855 | John Myhill |
| qa_1_285_len32677.json | qa_1 | 285 | 193 | 32677 | 1961 |
| qa_1_286_len30262.json | qa_1 | 286 | 193 | 30262 | Hisao Yamada |
| qa_1_287_len30107.json | qa_1 | 287 | 193 | 30107 | input encoding |
| qa_1_288_len31437.json | qa_1 | 288 | 193 | 31437 | encoding |
| qa_1_289_len31295.json | qa_1 | 289 | 193 | 31295 | Manuel Blum |
| qa_1_290_len31128.json | qa_1 | 290 | 193 | 31128 | speed-up theorem |
| qa_1_291_len30987.json | qa_1 | 291 | 193 | 30987 | "Reducibility Among Combinatorial Problems" |
| qa_1_292_len32118.json | qa_1 | 292 | 193 | 32118 | 21 |
| qa_1_293_len32519.json | qa_1 | 293 | 193 | 32519 | SoCal |
| qa_1_294_len29827.json | qa_1 | 294 | 193 | 29827 | 10 counties |
| qa_1_295_len30052.json | qa_1 | 295 | 183 | 30052 | economic center |
| qa_1_296_len30599.json | qa_1 | 296 | 193 | 30599 | demographics and economic ties |
| qa_1_297_len32212.json | qa_1 | 297 | 193 | 32212 | historical political divisions |
| qa_1_298_len32464.json | qa_1 | 298 | 193 | 32464 | Southern California Megaregion |
| qa_1_299_len31144.json | qa_1 | 299 | 193 | 31144 | 11 |
| qa_1_300_len32131.json | qa_1 | 300 | 193 | 32131 | Nevada |
| qa_1_301_len31150.json | qa_1 | 301 | 193 | 31150 | Mexican |
| qa_1_302_len32292.json | qa_1 | 302 | 193 | 32292 | Tijuana |
| qa_1_303_len31781.json | qa_1 | 303 | 193 | 31781 | Pacific |
| qa_1_304_len32194.json | qa_1 | 304 | 193 | 32194 | seven |
| qa_1_305_len32172.json | qa_1 | 305 | 193 | 32172 | 12 million |
| qa_1_306_len32614.json | qa_1 | 306 | 193 | 32614 | San Diego |
| qa_1_307_len30463.json | qa_1 | 307 | 193 | 30463 | 17.5 million |
| qa_1_308_len32638.json | qa_1 | 308 | 193 | 32638 | Colorado River |
| qa_1_309_len29928.json | qa_1 | 309 | 193 | 29928 | Colorado Desert |
| qa_1_310_len28522.json | qa_1 | 310 | 183 | 28522 | Mojave Desert |
| qa_1_311_len31561.json | qa_1 | 311 | 193 | 31561 | Mexico–United States border |
| qa_1_312_len31698.json | qa_1 | 312 | 193 | 31698 | California |
| qa_1_313_len32213.json | qa_1 | 313 | 193 | 32213 | 3,792,621 |
| qa_1_314_len29980.json | qa_1 | 314 | 183 | 29980 | Los Angeles |
| qa_1_315_len31592.json | qa_1 | 315 | 193 | 31592 | San Diego |
| qa_1_316_len31813.json | qa_1 | 316 | 193 | 31813 | south |
| qa_1_317_len31451.json | qa_1 | 317 | 193 | 31451 | Los Angeles |
| qa_1_318_len29239.json | qa_1 | 318 | 183 | 29239 | United States |
| qa_1_319_len31507.json | qa_1 | 319 | 193 | 31507 | counties |
| qa_1_320_len31578.json | qa_1 | 320 | 193 | 31578 | 15 |
| qa_1_321_len31108.json | qa_1 | 321 | 193 | 31108 | counties |
| qa_1_322_len31775.json | qa_1 | 322 | 193 | 31775 | Hollywood |
| qa_1_323_len31335.json | qa_1 | 323 | 193 | 31335 | Los Angeles |
| qa_1_324_len30926.json | qa_1 | 324 | 193 | 30926 | The Walt Disney Company |
| qa_1_325_len31401.json | qa_1 | 325 | 193 | 31401 | music |
| qa_1_326_len31852.json | qa_1 | 326 | 193 | 31852 | Sony |
| qa_1_327_len31029.json | qa_1 | 327 | 193 | 31029 | skateboard |
| qa_1_328_len30992.json | qa_1 | 328 | 193 | 30992 | Tony Hawk |
| qa_1_329_len30654.json | qa_1 | 329 | 193 | 30654 | Shaun White |
| qa_1_330_len30417.json | qa_1 | 330 | 193 | 30417 | Oahu |
| qa_1_331_len31615.json | qa_1 | 331 | 193 | 31615 | Transpac |
| qa_1_332_len29893.json | qa_1 | 332 | 193 | 29893 | Palm Springs |
| qa_1_333_len32011.json | qa_1 | 333 | 193 | 32011 | beaches |
| qa_1_334_len31569.json | qa_1 | 334 | 193 | 31569 | southern |
| qa_1_335_len30554.json | qa_1 | 335 | 193 | 30554 | open spaces |
| qa_1_336_len32077.json | qa_1 | 336 | 193 | 32077 | 37° 9' 58.23" |
| qa_1_337_len31075.json | qa_1 | 337 | 193 | 31075 | 11 |
| qa_1_338_len29543.json | qa_1 | 338 | 183 | 29543 | ten |
| qa_1_339_len31626.json | qa_1 | 339 | 193 | 31626 | Tehachapi Mountains |
| qa_1_340_len31882.json | qa_1 | 340 | 193 | 31882 | northern |
| qa_1_341_len31066.json | qa_1 | 341 | 193 | 31066 | Mexico |
| qa_1_342_len31418.json | qa_1 | 342 | 193 | 31418 | Alta California |
| qa_1_343_len29351.json | qa_1 | 343 | 183 | 29351 | Monterey |
| qa_1_344_len30649.json | qa_1 | 344 | 193 | 30649 | the Missouri Compromise |
| qa_1_345_len31504.json | qa_1 | 345 | 193 | 31504 | free |
| qa_1_346_len31588.json | qa_1 | 346 | 193 | 31588 | inequitable taxes |
| qa_1_347_len32298.json | qa_1 | 347 | 193 | 32298 | Cow Counties |
| qa_1_348_len30576.json | qa_1 | 348 | 193 | 30576 | three |
| qa_1_349_len30863.json | qa_1 | 349 | 193 | 30863 | 75 |
| qa_1_350_len31952.json | qa_1 | 350 | 193 | 31952 | Milton Latham |
| qa_1_351_len31055.json | qa_1 | 351 | 193 | 31055 | Los Angeles Times |
| qa_1_352_len31369.json | qa_1 | 352 | 193 | 31369 | 1900 |
| qa_1_353_len30108.json | qa_1 | 353 | 193 | 30108 | 1999 |
| qa_1_354_len31497.json | qa_1 | 354 | 193 | 31497 | Imperial |
| qa_1_355_len31182.json | qa_1 | 355 | 193 | 31182 | seven |
| qa_1_356_len30802.json | qa_1 | 356 | 193 | 30802 | regional tourism groups |
| qa_1_357_len28766.json | qa_1 | 357 | 183 | 28766 | California State Automobile Association |
| qa_1_358_len31443.json | qa_1 | 358 | 193 | 31443 | three-region |
| qa_1_359_len31784.json | qa_1 | 359 | 193 | 31784 | Tehachapis |
| qa_1_360_len32287.json | qa_1 | 360 | 193 | 32287 | southern |
| qa_1_361_len31826.json | qa_1 | 361 | 193 | 31826 | third |
| qa_1_362_len31764.json | qa_1 | 362 | 193 | 31764 | vast areas |
| qa_1_363_len30889.json | qa_1 | 363 | 193 | 30889 | suburban |
| qa_1_364_len31108.json | qa_1 | 364 | 193 | 31108 | highways |
| qa_1_365_len32328.json | qa_1 | 365 | 193 | 32328 | international metropolitan |
| qa_1_366_len31038.json | qa_1 | 366 | 193 | 31038 | Camp Pendleton |
| qa_1_367_len30408.json | qa_1 | 367 | 193 | 30408 | Inland Empire |
| qa_1_368_len30559.json | qa_1 | 368 | 193 | 30559 | United States Census Bureau |
| qa_1_369_len32256.json | qa_1 | 369 | 193 | 32256 | Orange |
| qa_1_370_len31498.json | qa_1 | 370 | 193 | 31498 | 1990s |
| qa_1_371_len30083.json | qa_1 | 371 | 193 | 30083 | Mediterranean |
| qa_1_372_len30547.json | qa_1 | 372 | 193 | 30547 | infrequent rain |
| qa_1_373_len32203.json | qa_1 | 373 | 193 | 32203 | 60's |
| qa_1_374_len32014.json | qa_1 | 374 | 193 | 32014 | very rare |
| qa_1_375_len32550.json | qa_1 | 375 | 193 | 32550 | 70 |
| qa_1_376_len32319.json | qa_1 | 376 | 193 | 32319 | varied |
| qa_1_377_len31609.json | qa_1 | 377 | 193 | 31609 | Pacific Ocean |
| qa_1_378_len31522.json | qa_1 | 378 | 193 | 31522 | topographic |
| qa_1_379_len31519.json | qa_1 | 379 | 193 | 31519 | Peninsular |
| qa_1_380_len31072.json | qa_1 | 380 | 193 | 31072 | valleys |
| qa_1_381_len31168.json | qa_1 | 381 | 193 | 31168 | 10,000 |
| qa_1_382_len30045.json | qa_1 | 382 | 183 | 30045 | small |
| qa_1_383_len31908.json | qa_1 | 383 | 193 | 31908 | 6.7 |
| qa_1_384_len32179.json | qa_1 | 384 | 193 | 32179 | property damage |
| qa_1_385_len31185.json | qa_1 | 385 | 193 | 31185 | $20 billion |
| qa_1_386_len32048.json | qa_1 | 386 | 193 | 32048 | San Andreas |
| qa_1_387_len32039.json | qa_1 | 387 | 193 | 32039 | 6.7 |
| qa_1_388_len29885.json | qa_1 | 388 | 193 | 29885 | Puente Hills |
| qa_1_389_len32356.json | qa_1 | 389 | 193 | 32356 | USGS |
| qa_1_390_len30911.json | qa_1 | 390 | 193 | 30911 | occurrence |
| qa_1_391_len31443.json | qa_1 | 391 | 193 | 31443 | economically |
| qa_1_392_len30635.json | qa_1 | 392 | 193 | 30635 | global |
| qa_1_393_len31652.json | qa_1 | 393 | 193 | 31652 | economic |
| qa_1_394_len29458.json | qa_1 | 394 | 193 | 29458 | 2010 |
| qa_1_395_len30255.json | qa_1 | 395 | 183 | 30255 | high growth rates |
| qa_1_396_len28472.json | qa_1 | 396 | 183 | 28472 | 10.0% |
| qa_1_397_len30756.json | qa_1 | 397 | 193 | 30756 | tech-oriented |
| qa_1_398_len31715.json | qa_1 | 398 | 193 | 31715 | Greater Sacramento |
| qa_1_399_len32553.json | qa_1 | 399 | 193 | 32553 | Metropolitan Statistical Areas |
| qa_1_400_len30519.json | qa_1 | 400 | 193 | 30519 | two |
| qa_1_401_len30995.json | qa_1 | 401 | 193 | 30995 | five million |
| qa_1_402_len31777.json | qa_1 | 402 | 193 | 31777 | Southern Border Region |
| qa_1_403_len30928.json | qa_1 | 403 | 193 | 30928 | 17,786,419 |
| qa_1_404_len30596.json | qa_1 | 404 | 193 | 30596 | Los Angeles |
| qa_1_405_len31678.json | qa_1 | 405 | 193 | 31678 | 1.3 million |
| qa_1_406_len31009.json | qa_1 | 406 | 193 | 31009 | twelve |
| qa_1_407_len30930.json | qa_1 | 407 | 193 | 30930 | 100,000 |
| qa_1_408_len31508.json | qa_1 | 408 | 193 | 31508 | Riverside |
| qa_1_409_len32603.json | qa_1 | 409 | 193 | 32603 | petroleum |
| qa_1_410_len30976.json | qa_1 | 410 | 193 | 30976 | Hollywood |
| qa_1_411_len30867.json | qa_1 | 411 | 193 | 30867 | the housing bubble |
| qa_1_412_len31735.json | qa_1 | 412 | 193 | 31735 | diverse |
| qa_1_413_len31568.json | qa_1 | 413 | 193 | 31568 | heavily impacted |
| qa_1_414_len32711.json | qa_1 | 414 | 193 | 32711 | 1920s |
| qa_1_415_len32163.json | qa_1 | 415 | 193 | 32163 | richest |
| qa_1_416_len30849.json | qa_1 | 416 | 193 | 30849 | cattle |
| qa_1_417_len30527.json | qa_1 | 417 | 193 | 30527 | citrus |
| qa_1_418_len30567.json | qa_1 | 418 | 193 | 30567 | aerospace |
| qa_1_419_len30925.json | qa_1 | 419 | 193 | 30925 | business |
| qa_1_420_len31772.json | qa_1 | 420 | 193 | 31772 | Central business districts |
| qa_1_421_len31550.json | qa_1 | 421 | 193 | 31550 | South Coast Metro |
| qa_1_422_len30644.json | qa_1 | 422 | 193 | 30644 | business |
| qa_1_423_len31812.json | qa_1 | 423 | 193 | 31812 | Los Angeles Area |
| qa_1_424_len31379.json | qa_1 | 424 | 193 | 31379 | San Fernando Valley |
| qa_1_425_len32255.json | qa_1 | 425 | 193 | 32255 | Los Angeles |
| qa_1_426_len30733.json | qa_1 | 426 | 193 | 30733 | business |
| qa_1_427_len30853.json | qa_1 | 427 | 193 | 30853 | Riverside |
| qa_1_428_len31749.json | qa_1 | 428 | 193 | 31749 | Hospitality Business/Financial Centre |
| qa_1_429_len31042.json | qa_1 | 429 | 193 | 31042 | Orange |
| qa_1_430_len30297.json | qa_1 | 430 | 193 | 30297 | University of California, Irvine |
| qa_1_431_len30087.json | qa_1 | 431 | 193 | 30087 | West Irvine |
| qa_1_432_len31340.json | qa_1 | 432 | 193 | 31340 | South Coast Metro |
| qa_1_433_len30556.json | qa_1 | 433 | 193 | 30556 | rapidly |
| qa_1_434_len30846.json | qa_1 | 434 | 193 | 30846 | Downtown San Diego |
| qa_1_435_len30516.json | qa_1 | 435 | 193 | 30516 | Northern San Diego |
| qa_1_436_len30213.json | qa_1 | 436 | 193 | 30213 | North County |
| qa_1_437_len30183.json | qa_1 | 437 | 193 | 30183 | San Diego |
| qa_1_438_len31529.json | qa_1 | 438 | 193 | 31529 | Los Angeles International Airport |
| qa_1_439_len29042.json | qa_1 | 439 | 183 | 29042 | passenger volume |
| qa_1_440_len32641.json | qa_1 | 440 | 193 | 32641 | third |
| qa_1_441_len31400.json | qa_1 | 441 | 193 | 31400 | San Diego International Airport |
| qa_1_442_len29771.json | qa_1 | 442 | 183 | 29771 | Van Nuys Airport |
| qa_1_443_len32302.json | qa_1 | 443 | 193 | 32302 | Metrolink |
| qa_1_444_len30420.json | qa_1 | 444 | 183 | 30420 | seven |
| qa_1_445_len30842.json | qa_1 | 445 | 193 | 30842 | Six |
| qa_1_446_len30929.json | qa_1 | 446 | 193 | 30929 | Orange |
| qa_1_447_len31793.json | qa_1 | 447 | 193 | 31793 | Port of Los Angeles |
| qa_1_448_len30896.json | qa_1 | 448 | 193 | 30896 | Port of San Diego |
| qa_1_449_len30936.json | qa_1 | 449 | 193 | 30936 | Southern |
| qa_1_450_len32193.json | qa_1 | 450 | 193 | 32193 | The Tech Coast |
| qa_1_451_len31782.json | qa_1 | 451 | 193 | 31782 | research |
| qa_1_452_len31468.json | qa_1 | 452 | 193 | 31468 | private |
| qa_1_453_len31085.json | qa_1 | 453 | 193 | 31085 | 5 |
| qa_1_454_len31497.json | qa_1 | 454 | 193 | 31497 | 12 |
| qa_1_455_len30461.json | qa_1 | 455 | 193 | 30461 | NFL |
| qa_1_456_len30509.json | qa_1 | 456 | 183 | 30509 | NBA |
| qa_1_457_len31031.json | qa_1 | 457 | 193 | 31031 | MLB |
| qa_1_458_len30982.json | qa_1 | 458 | 193 | 30982 | Los Angeles Kings |
| qa_1_459_len30810.json | qa_1 | 459 | 193 | 30810 | LA Galaxy |
| qa_1_460_len31244.json | qa_1 | 460 | 193 | 31244 | Chivas USA |
| qa_1_461_len31020.json | qa_1 | 461 | 193 | 31020 | two |
| qa_1_462_len30989.json | qa_1 | 462 | 193 | 30989 | 2014 |
| qa_1_463_len30766.json | qa_1 | 463 | 193 | 30766 | StubHub Center |
| qa_1_464_len30944.json | qa_1 | 464 | 193 | 30944 | 2018 |
| qa_1_465_len31899.json | qa_1 | 465 | 193 | 31899 | College |
| qa_1_466_len31771.json | qa_1 | 466 | 193 | 31771 | UCLA |
| qa_1_467_len30828.json | qa_1 | 467 | 193 | 30828 | Trojans |
| qa_1_468_len30653.json | qa_1 | 468 | 193 | 30653 | Pac-12 |
| qa_1_469_len29430.json | qa_1 | 469 | 183 | 29430 | Division I |
| qa_1_470_len30347.json | qa_1 | 470 | 193 | 30347 | Rugby |
| qa_1_471_len31303.json | qa_1 | 471 | 193 | 31303 | high school |
| qa_1_472_len32030.json | qa_1 | 472 | 193 | 32030 | an official school sport |
| qa_1_473_len32001.json | qa_1 | 473 | 193 | 32001 | BSkyB |
| qa_1_474_len32455.json | qa_1 | 474 | 193 | 32455 | BSkyB |
| qa_1_475_len31865.json | qa_1 | 475 | 183 | 31865 | 2014 |
| qa_1_476_len31147.json | qa_1 | 476 | 183 | 31147 | Sky plc |
| qa_1_477_len31000.json | qa_1 | 477 | 183 | 31000 | Sky UK Limited |
| qa_1_478_len32650.json | qa_1 | 478 | 193 | 32650 | 2006 |
| qa_1_479_len29309.json | qa_1 | 479 | 173 | 29309 | two |
| qa_1_480_len31175.json | qa_1 | 480 | 183 | 31175 | Sky |
| qa_1_481_len31842.json | qa_1 | 481 | 183 | 31842 | £1.3bn |
| qa_1_482_len31313.json | qa_1 | 482 | 183 | 31313 | ONdigital |
| qa_1_483_len31429.json | qa_1 | 483 | 183 | 31429 | Freeview |
| qa_1_484_len31369.json | qa_1 | 484 | 173 | 31369 | three |
| qa_1_485_len32004.json | qa_1 | 485 | 193 | 32004 | Sky Three |
| qa_1_486_len30152.json | qa_1 | 486 | 183 | 30152 | Pick TV |
| qa_1_487_len31993.json | qa_1 | 487 | 183 | 31993 | Sky+ PVR |
| qa_1_488_len31846.json | qa_1 | 488 | 183 | 31846 | September 2007 |
| qa_1_489_len31273.json | qa_1 | 489 | 183 | 31273 | monthly fee |
| qa_1_490_len32200.json | qa_1 | 490 | 183 | 32200 | January 2010 |
| qa_1_491_len31410.json | qa_1 | 491 | 183 | 31410 | Sky+HD Box |
| qa_1_492_len32464.json | qa_1 | 492 | 183 | 32464 | VideoGuard |
| qa_1_493_len32303.json | qa_1 | 493 | 183 | 32303 | NDS |
| qa_1_494_len30821.json | qa_1 | 494 | 183 | 30821 | Cisco Systems |
| qa_1_495_len31257.json | qa_1 | 495 | 183 | 31257 | BSkyB |
| qa_1_496_len31906.json | qa_1 | 496 | 193 | 31906 | Sky+ |
| qa_1_497_len31565.json | qa_1 | 497 | 183 | 31565 | 2007 |
| qa_1_498_len31165.json | qa_1 | 498 | 183 | 31165 | basic channels |
| qa_1_499_len30934.json | qa_1 | 499 | 183 | 30934 | substantially increased the asking price |

qa_1 长度分布：min=28340, 中位=31295, max=32732, 均值=31250

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
| qa_2_024_len30318.json | qa_2 | 24 | 228 | 30318 | World's Best Goalkeeper |
| qa_2_025_len30613.json | qa_2 | 25 | 238 | 30613 | Barton Lee Hazlewood |
| qa_2_026_len32716.json | qa_2 | 26 | 228 | 32716 | 1838 |
| qa_2_027_len32343.json | qa_2 | 27 | 228 | 32343 | yes |
| qa_2_028_len30034.json | qa_2 | 28 | 228 | 30034 | Henry J. Kaiser |
| qa_2_029_len29006.json | qa_2 | 29 | 228 | 29006 | Arena of Khazan |
| qa_2_030_len32408.json | qa_2 | 30 | 228 | 32408 | 2000 |
| qa_2_031_len31794.json | qa_2 | 31 | 228 | 31794 | Fujioka, Gunma |
| qa_2_032_len31191.json | qa_2 | 32 | 228 | 31191 | Charles Eugène |
| qa_2_033_len31857.json | qa_2 | 33 | 218 | 31857 | no |
| qa_2_034_len30981.json | qa_2 | 34 | 238 | 30981 | Letters to Cleo |
| qa_2_035_len31102.json | qa_2 | 35 | 238 | 31102 | October 1922 |
| qa_2_036_len32115.json | qa_2 | 36 | 228 | 32115 | 2000 |
| qa_2_037_len31683.json | qa_2 | 37 | 228 | 31683 | World War II |
| qa_2_038_len32687.json | qa_2 | 38 | 228 | 32687 | no |
| qa_2_039_len32205.json | qa_2 | 39 | 238 | 32205 | New York City |
| qa_2_040_len29707.json | qa_2 | 40 | 218 | 29707 | Scotch Collie |
| qa_2_041_len30563.json | qa_2 | 41 | 228 | 30563 | Mumbai |
| qa_2_042_len32525.json | qa_2 | 42 | 228 | 32525 | 1962 |
| qa_2_043_len30954.json | qa_2 | 43 | 218 | 30954 | sovereignty |
| qa_2_044_len31499.json | qa_2 | 44 | 218 | 31499 | Nelson Rockefeller |
| qa_2_045_len32664.json | qa_2 | 45 | 228 | 32664 | Yellowcraig |
| qa_2_046_len31525.json | qa_2 | 46 | 228 | 31525 | Phil Spector |
| qa_2_047_len31157.json | qa_2 | 47 | 218 | 31157 | Organizations could come together to address global issues |
| qa_2_048_len30718.json | qa_2 | 48 | 218 | 30718 | yes |
| qa_2_049_len32426.json | qa_2 | 49 | 238 | 32426 | English Electric Canberra |
| qa_2_050_len30583.json | qa_2 | 50 | 228 | 30583 | 2009 Big 12 Conference |
| qa_2_051_len29134.json | qa_2 | 51 | 218 | 29134 | 1,462 |
| qa_2_052_len32148.json | qa_2 | 52 | 228 | 32148 | Indianapolis Motor Speedway |
| qa_2_053_len30874.json | qa_2 | 53 | 228 | 30874 | Rome |
| qa_2_054_len31684.json | qa_2 | 54 | 228 | 31684 | Max Martin, Savan Kotecha and Ilya Salmanzadeh |
| qa_2_055_len32186.json | qa_2 | 55 | 228 | 32186 | Marion, South Australia |
| qa_2_056_len32706.json | qa_2 | 56 | 238 | 32706 | Drifting |
| qa_2_057_len31681.json | qa_2 | 57 | 228 | 31681 | Keith Bostic |
| qa_2_058_len32641.json | qa_2 | 58 | 228 | 32641 | 35,124 |
| qa_2_059_len31590.json | qa_2 | 59 | 238 | 31590 | no |
| qa_2_060_len31574.json | qa_2 | 60 | 238 | 31574 | shortest player ever to play in the National Basketball Asso |
| qa_2_061_len30328.json | qa_2 | 61 | 228 | 30328 | Ronald Shusett |
| qa_2_062_len32496.json | qa_2 | 62 | 238 | 32496 | Adeline Virginia Woolf |
| qa_2_063_len30735.json | qa_2 | 63 | 228 | 30735 | 821 |
| qa_2_064_len31090.json | qa_2 | 64 | 228 | 31090 | more than 70 countries |
| qa_2_065_len30489.json | qa_2 | 65 | 218 | 30489 | Charmed |
| qa_2_066_len31221.json | qa_2 | 66 | 228 | 31221 | International Boxing Hall of Fame |
| qa_2_067_len31901.json | qa_2 | 67 | 228 | 31901 | Usher |
| qa_2_068_len29747.json | qa_2 | 68 | 228 | 29747 | Bill Murray |
| qa_2_069_len31334.json | qa_2 | 69 | 228 | 31334 | Carabao Cup |
| qa_2_070_len30409.json | qa_2 | 70 | 218 | 30409 | Teen Titans Go! |
| qa_2_071_len30940.json | qa_2 | 71 | 228 | 30940 | 276,170 inhabitants |
| qa_2_072_len32001.json | qa_2 | 72 | 238 | 32001 | orange |
| qa_2_073_len32695.json | qa_2 | 73 | 228 | 32695 | Tromeo and Juliet |
| qa_2_074_len28897.json | qa_2 | 74 | 218 | 28897 | William Jefferson Clinton |
| qa_2_075_len30720.json | qa_2 | 75 | 238 | 30720 | John John Florence |
| qa_2_076_len31469.json | qa_2 | 76 | 228 | 31469 | Ann |
| qa_2_077_len31196.json | qa_2 | 77 | 218 | 31196 | Apalachees |
| qa_2_078_len30936.json | qa_2 | 78 | 228 | 30936 | British |
| qa_2_079_len32456.json | qa_2 | 79 | 238 | 32456 | 1865 |
| qa_2_080_len30443.json | qa_2 | 80 | 218 | 30443 | Newport |
| qa_2_081_len30467.json | qa_2 | 81 | 228 | 30467 | Bob Seger |
| qa_2_082_len32651.json | qa_2 | 82 | 228 | 32651 | Conscription |
| qa_2_083_len31347.json | qa_2 | 83 | 218 | 31347 | Mondelez International, Inc. |
| qa_2_084_len31142.json | qa_2 | 84 | 228 | 31142 | Monica Lewinsky |
| qa_2_085_len31063.json | qa_2 | 85 | 218 | 31063 | April 1, 1949 |
| qa_2_086_len32396.json | qa_2 | 86 | 228 | 32396 | 1866 |
| qa_2_087_len30960.json | qa_2 | 87 | 218 | 30960 | Canary Islands, Spain |
| qa_2_088_len32564.json | qa_2 | 88 | 238 | 32564 | 250 million |
| qa_2_089_len29446.json | qa_2 | 89 | 228 | 29446 | director |
| qa_2_090_len31022.json | qa_2 | 90 | 228 | 31022 | The Conversation |
| qa_2_091_len29529.json | qa_2 | 91 | 228 | 29529 | John Waters |
| qa_2_092_len32060.json | qa_2 | 92 | 238 | 32060 | Las Vegas Strip in Paradise |
| qa_2_093_len31803.json | qa_2 | 93 | 238 | 31803 | no |
| qa_2_094_len32728.json | qa_2 | 94 | 238 | 32728 | March and April |
| qa_2_095_len30382.json | qa_2 | 95 | 218 | 30382 | Fairfax County |
| qa_2_096_len30281.json | qa_2 | 96 | 228 | 30281 | IT products and services |
| qa_2_097_len31183.json | qa_2 | 97 | 218 | 31183 | Levni Yilmaz |
| qa_2_098_len31177.json | qa_2 | 98 | 238 | 31177 | Beijing |
| qa_2_099_len31084.json | qa_2 | 99 | 228 | 31084 | no |
| qa_2_100_len29933.json | qa_2 | 100 | 228 | 29933 | Coahuila, Mexico |
| qa_2_101_len32041.json | qa_2 | 101 | 238 | 32041 | Larnelle Harris |
| qa_2_102_len31969.json | qa_2 | 102 | 228 | 31969 | 2003 |
| qa_2_103_len29605.json | qa_2 | 103 | 218 | 29605 | North Avenue at Techwood Drive |
| qa_2_104_len31540.json | qa_2 | 104 | 228 | 31540 | Leo Varadkar |
| qa_2_105_len31867.json | qa_2 | 105 | 228 | 31867 | George Raft |
| qa_2_106_len31791.json | qa_2 | 106 | 228 | 31791 | 1887 |
| qa_2_107_len31973.json | qa_2 | 107 | 238 | 31973 | business |
| qa_2_108_len32698.json | qa_2 | 108 | 238 | 32698 | 7 October 1978 |
| qa_2_109_len31517.json | qa_2 | 109 | 228 | 31517 | B-17 Flying Fortress bomber |
| qa_2_110_len32601.json | qa_2 | 110 | 238 | 32601 | Blake Shelton |
| qa_2_111_len30586.json | qa_2 | 111 | 228 | 30586 | three centuries |
| qa_2_112_len31868.json | qa_2 | 112 | 228 | 31868 | 7 January 1936 |
| qa_2_113_len32716.json | qa_2 | 113 | 228 | 32716 | George Archainbaud |
| qa_2_114_len30597.json | qa_2 | 114 | 238 | 30597 | picric acid |
| qa_2_115_len31380.json | qa_2 | 115 | 228 | 31380 | 230 |
| qa_2_116_len31774.json | qa_2 | 116 | 238 | 31774 | Marvel |
| qa_2_117_len31538.json | qa_2 | 117 | 238 | 31538 | United States Senator |
| qa_2_118_len32050.json | qa_2 | 118 | 228 | 32050 | no |
| qa_2_119_len30712.json | qa_2 | 119 | 228 | 30712 | yes |
| qa_2_120_len32159.json | qa_2 | 120 | 228 | 32159 | The A41 |
| qa_2_121_len32108.json | qa_2 | 121 | 238 | 32108 | Princess Jessica |
| qa_2_122_len32398.json | qa_2 | 122 | 238 | 32398 | The Spiderwick Chronicles |
| qa_2_123_len32274.json | qa_2 | 123 | 238 | 32274 | KKR & Co |
| qa_2_124_len31640.json | qa_2 | 124 | 238 | 31640 | The Bye Bye Man |
| qa_2_125_len31802.json | qa_2 | 125 | 228 | 31802 | Richmond |
| qa_2_126_len32646.json | qa_2 | 126 | 228 | 32646 | Owsley Stanley |
| qa_2_127_len32694.json | qa_2 | 127 | 238 | 32694 | the late 12th Century |
| qa_2_128_len32415.json | qa_2 | 128 | 228 | 32415 | San Luis Obispo, California |
| qa_2_129_len30184.json | qa_2 | 129 | 218 | 30184 | Mercedes-Benz Superdome |
| qa_2_130_len32056.json | qa_2 | 130 | 238 | 32056 | LaLee's Kin: The Legacy of Cotton |
| qa_2_131_len31463.json | qa_2 | 131 | 218 | 31463 | Pirate's Cove |
| qa_2_132_len32387.json | qa_2 | 132 | 228 | 32387 | Crystal Dynamics |
| qa_2_133_len31896.json | qa_2 | 133 | 228 | 31896 | "The Marshall Mathers LP 2" |
| qa_2_134_len32449.json | qa_2 | 134 | 238 | 32449 | Suicide Squad |
| qa_2_135_len31438.json | qa_2 | 135 | 218 | 31438 | Jillian Belk |
| qa_2_136_len32293.json | qa_2 | 136 | 238 | 32293 | Texas Tech University |
| qa_2_137_len31037.json | qa_2 | 137 | 228 | 31037 | Lord Combermere |
| qa_2_138_len31463.json | qa_2 | 138 | 228 | 31463 | seasonal television specials, particularly its work in stop  |
| qa_2_139_len32591.json | qa_2 | 139 | 238 | 32591 | 2014 |
| qa_2_140_len32661.json | qa_2 | 140 | 238 | 32661 | Oxford |
| qa_2_141_len32543.json | qa_2 | 141 | 228 | 32543 | Roberta Vinci |
| qa_2_142_len31613.json | qa_2 | 142 | 218 | 31613 | "Teach the Controversy" campaign |
| qa_2_143_len32393.json | qa_2 | 143 | 238 | 32393 | Lostprophets disbanded in 2013 after Watkins was charged wit |
| qa_2_144_len32167.json | qa_2 | 144 | 238 | 32167 | Nebo Zovyot |
| qa_2_145_len30511.json | qa_2 | 145 | 218 | 30511 | Erika Jayne |
| qa_2_146_len32711.json | qa_2 | 146 | 228 | 32711 | 2009 |
| qa_2_147_len30681.json | qa_2 | 147 | 218 | 30681 | Kingdom of the Isles |
| qa_2_148_len31691.json | qa_2 | 148 | 238 | 31691 | The Beatles |
| qa_2_149_len31748.json | qa_2 | 149 | 238 | 31748 | "We'll Burn That Bridge" |
| qa_2_150_len32058.json | qa_2 | 150 | 238 | 32058 | director |
| qa_2_151_len32224.json | qa_2 | 151 | 228 | 32224 | 1901 |
| qa_2_152_len32110.json | qa_2 | 152 | 228 | 32110 | Argand lamp |
| qa_2_153_len30330.json | qa_2 | 153 | 228 | 30330 | seven |
| qa_2_154_len31886.json | qa_2 | 154 | 238 | 31886 | yes |
| qa_2_155_len32709.json | qa_2 | 155 | 238 | 32709 | Brazil |
| qa_2_156_len32375.json | qa_2 | 156 | 228 | 32375 | The Seven Days Battles |
| qa_2_157_len32196.json | qa_2 | 157 | 218 | 32196 | Lush Ltd. |
| qa_2_158_len32037.json | qa_2 | 158 | 238 | 32037 | Manchester United |
| qa_2_159_len32334.json | qa_2 | 159 | 228 | 32334 | 1989 |
| qa_2_160_len32507.json | qa_2 | 160 | 238 | 32507 | Kathryn Jean Martin |
| qa_2_161_len31707.json | qa_2 | 161 | 238 | 31707 | Frank Lowy |
| qa_2_162_len32246.json | qa_2 | 162 | 238 | 32246 | Las Vegas |
| qa_2_163_len30403.json | qa_2 | 163 | 218 | 30403 | Kelly Lee Osbourne |
| qa_2_164_len32441.json | qa_2 | 164 | 228 | 32441 | Sean Yseult. |
| qa_2_165_len31303.json | qa_2 | 165 | 218 | 31303 | Prescription Drug User Fee Act |
| qa_2_166_len32160.json | qa_2 | 166 | 238 | 32160 | Dinosaur |
| qa_2_167_len32056.json | qa_2 | 167 | 238 | 32056 | Wichita |
| qa_2_168_len30372.json | qa_2 | 168 | 218 | 30372 | United States presidential election of 2016 |
| qa_2_169_len32303.json | qa_2 | 169 | 238 | 32303 | Roman |
| qa_2_170_len29727.json | qa_2 | 170 | 228 | 29727 | C. W. Grafton |
| qa_2_171_len31132.json | qa_2 | 171 | 218 | 31132 | Brooklyn, New York |
| qa_2_172_len31576.json | qa_2 | 172 | 228 | 31576 | 48,982 |
| qa_2_173_len32568.json | qa_2 | 173 | 238 | 32568 | Pasek & Paul |
| qa_2_174_len32411.json | qa_2 | 174 | 228 | 32411 | City of Newcastle |
| qa_2_175_len32470.json | qa_2 | 175 | 238 | 32470 | yes |
| qa_2_176_len31431.json | qa_2 | 176 | 228 | 31431 | drawings |
| qa_2_177_len31630.json | qa_2 | 177 | 218 | 31630 | between the 8th and 16th centuries |
| qa_2_178_len30925.json | qa_2 | 178 | 228 | 30925 | Alistair Grant |
| qa_2_179_len32423.json | qa_2 | 179 | 238 | 32423 | Scotland |
| qa_2_180_len29197.json | qa_2 | 180 | 218 | 29197 | Comedy Film Nerds |
| qa_2_181_len32221.json | qa_2 | 181 | 218 | 32221 | Summerlin |
| qa_2_182_len31321.json | qa_2 | 182 | 218 | 31321 | George Harrison |
| qa_2_183_len31596.json | qa_2 | 183 | 238 | 31596 | Canadian professional ice hockey centre |
| qa_2_184_len32380.json | qa_2 | 184 | 228 | 32380 | 1952 |
| qa_2_185_len30717.json | qa_2 | 185 | 228 | 30717 | third |
| qa_2_186_len32520.json | qa_2 | 186 | 238 | 32520 | Mark Neveldine and Brian Taylor |
| qa_2_187_len30471.json | qa_2 | 187 | 218 | 30471 | Hidden America with Jonah Ray |
| qa_2_188_len31954.json | qa_2 | 188 | 238 | 31954 | no |
| qa_2_189_len31141.json | qa_2 | 189 | 228 | 31141 | 1943 |
| qa_2_190_len31498.json | qa_2 | 190 | 228 | 31498 | Oregon Ducks football |
| qa_2_191_len29900.json | qa_2 | 191 | 218 | 29900 | National Broadcasting Company |
| qa_2_192_len31226.json | qa_2 | 192 | 218 | 31226 | North African Arab |
| qa_2_193_len31968.json | qa_2 | 193 | 228 | 31968 | Murray Hill |
| qa_2_194_len32169.json | qa_2 | 194 | 228 | 32169 | severe acute respiratory syndrome |
| qa_2_195_len32576.json | qa_2 | 195 | 228 | 32576 | Wendell Berry |
| qa_2_196_len31556.json | qa_2 | 196 | 228 | 31556 | Tinu Suresh Desai |
| qa_2_197_len29159.json | qa_2 | 197 | 228 | 29159 | Golden Valley, Minnesota, |
| qa_2_198_len32190.json | qa_2 | 198 | 228 | 32190 | $10.5 million |
| qa_2_199_len29142.json | qa_2 | 199 | 208 | 29142 | 1999 Odisha cyclone |
| qa_2_200_len31724.json | qa_2 | 200 | 228 | 31724 | Crackle |
| qa_2_201_len30493.json | qa_2 | 201 | 238 | 30493 | Salford City Football Club |
| qa_2_202_len31148.json | qa_2 | 202 | 238 | 31148 | architect Michael Graves |
| qa_2_203_len32602.json | qa_2 | 203 | 238 | 32602 | multiple awards |
| qa_2_204_len32407.json | qa_2 | 204 | 228 | 32407 | Kevin Spacey |
| qa_2_205_len31610.json | qa_2 | 205 | 228 | 31610 | Ryan Seacrest |
| qa_2_206_len31049.json | qa_2 | 206 | 238 | 31049 | yes |
| qa_2_207_len30987.json | qa_2 | 207 | 238 | 30987 | Colonel Charles Hastings Judd |
| qa_2_208_len30476.json | qa_2 | 208 | 228 | 30476 | Pavel Sergeyevich Alexandrov |
| qa_2_209_len31760.json | qa_2 | 209 | 228 | 31760 | Gal Gadot |
| qa_2_210_len29911.json | qa_2 | 210 | 228 | 29911 | Rear Adm. J. Edward Snyder |
| qa_2_211_len31680.json | qa_2 | 211 | 228 | 31680 | Jackson Storm |
| qa_2_212_len32035.json | qa_2 | 212 | 238 | 32035 | 712 5th Avenue |
| qa_2_213_len31506.json | qa_2 | 213 | 238 | 31506 | right-hand |
| qa_2_214_len31684.json | qa_2 | 214 | 238 | 31684 | Arthur Ignatius Conan Doyle |
| qa_2_215_len32235.json | qa_2 | 215 | 238 | 32235 | Birmingham, Alabama |
| qa_2_216_len32245.json | qa_2 | 216 | 238 | 32245 | 44 |
| qa_2_217_len31129.json | qa_2 | 217 | 228 | 31129 | the Parthian Empire |
| qa_2_218_len32672.json | qa_2 | 218 | 228 | 32672 | Erinsborough |
| qa_2_219_len31578.json | qa_2 | 219 | 228 | 31578 | Samoa |
| qa_2_220_len31878.json | qa_2 | 220 | 238 | 31878 | Kimberly Couger "Kimmy" Schmidt |
| qa_2_221_len32711.json | qa_2 | 221 | 228 | 32711 | "The Comic Strip Presents..." |
| qa_2_222_len31806.json | qa_2 | 222 | 228 | 31806 | no |
| qa_2_223_len32483.json | qa_2 | 223 | 238 | 32483 | nineteenth |
| qa_2_224_len32254.json | qa_2 | 224 | 228 | 32254 | Awake |
| qa_2_225_len32092.json | qa_2 | 225 | 228 | 32092 | French Canadians |
| qa_2_226_len30714.json | qa_2 | 226 | 228 | 30714 | American |
| qa_2_227_len30895.json | qa_2 | 227 | 218 | 30895 | 8 km |
| qa_2_228_len32344.json | qa_2 | 228 | 228 | 32344 | Rawhide |
| qa_2_229_len31901.json | qa_2 | 229 | 228 | 31901 | Marsilea |
| qa_2_230_len31964.json | qa_2 | 230 | 238 | 31964 | Mark Masons' Hall |
| qa_2_231_len30703.json | qa_2 | 231 | 218 | 30703 | Starlite |
| qa_2_232_len31284.json | qa_2 | 232 | 228 | 31284 | Hong Kong |
| qa_2_233_len30764.json | qa_2 | 233 | 228 | 30764 | Craig v. Boren |
| qa_2_234_len31441.json | qa_2 | 234 | 218 | 31441 | "Queen City" |
| qa_2_235_len30237.json | qa_2 | 235 | 218 | 30237 | Arrowhead Stadium |
| qa_2_236_len32094.json | qa_2 | 236 | 238 | 32094 | acidic |
| qa_2_237_len32400.json | qa_2 | 237 | 228 | 32400 | Compliance |
| qa_2_238_len30903.json | qa_2 | 238 | 218 | 30903 | Logan International Airport |
| qa_2_239_len30911.json | qa_2 | 239 | 238 | 30911 | Ehretia |
| qa_2_240_len30810.json | qa_2 | 240 | 218 | 30810 | a pinball machine |
| qa_2_241_len31387.json | qa_2 | 241 | 228 | 31387 | Donna Paige Helmintoller |
| qa_2_242_len31624.json | qa_2 | 242 | 228 | 31624 | Reginald Martinez "Reggie" Jackson |
| qa_2_243_len31589.json | qa_2 | 243 | 228 | 31589 | Thomas Mawson |
| qa_2_244_len31686.json | qa_2 | 244 | 238 | 31686 | strings |
| qa_2_245_len30310.json | qa_2 | 245 | 218 | 30310 | Brad Silberling |
| qa_2_246_len32327.json | qa_2 | 246 | 238 | 32327 | 25 laps |
| qa_2_247_len31801.json | qa_2 | 247 | 228 | 31801 | yes |
| qa_2_248_len31406.json | qa_2 | 248 | 228 | 31406 | Thomas Loren Friedman |
| qa_2_249_len29731.json | qa_2 | 249 | 218 | 29731 | Taylor Swift |
| qa_2_250_len32062.json | qa_2 | 250 | 218 | 32062 | Istanbul, Turkey |
| qa_2_251_len31295.json | qa_2 | 251 | 218 | 31295 | super-regional shopping mall |
| qa_2_252_len32173.json | qa_2 | 252 | 228 | 32173 | Ordos City |
| qa_2_253_len30672.json | qa_2 | 253 | 228 | 30672 | Scott Sidney |
| qa_2_254_len32374.json | qa_2 | 254 | 228 | 32374 | over 600 stores |
| qa_2_255_len30428.json | qa_2 | 255 | 228 | 30428 | Berea College |
| qa_2_256_len32231.json | qa_2 | 256 | 238 | 32231 | 1945 |
| qa_2_257_len32091.json | qa_2 | 257 | 218 | 32091 | Mountbatten family |
| qa_2_258_len30288.json | qa_2 | 258 | 218 | 30288 | James Burke |
| qa_2_259_len32210.json | qa_2 | 259 | 238 | 32210 | Gulangyu |
| qa_2_260_len31376.json | qa_2 | 260 | 228 | 31376 | Slaughterhouse-Five |
| qa_2_261_len32162.json | qa_2 | 261 | 228 | 32162 | the east of Ireland |
| qa_2_262_len31692.json | qa_2 | 262 | 228 | 31692 | Distillery |
| qa_2_263_len32274.json | qa_2 | 263 | 218 | 32274 | Tybalt |
| qa_2_264_len32213.json | qa_2 | 264 | 238 | 32213 | SKUM |
| qa_2_265_len31537.json | qa_2 | 265 | 228 | 31537 | Anabolic steroids |
| qa_2_266_len30372.json | qa_2 | 266 | 218 | 30372 | Don S. Davis |
| qa_2_267_len32695.json | qa_2 | 267 | 238 | 32695 | Charles Kelley |
| qa_2_268_len32712.json | qa_2 | 268 | 238 | 32712 | 3,384,569 |
| qa_2_269_len32098.json | qa_2 | 269 | 228 | 32098 | Anna Clyne |
| qa_2_270_len32727.json | qa_2 | 270 | 238 | 32727 | USS Essex |
| qa_2_271_len30866.json | qa_2 | 271 | 228 | 30866 | yes |
| qa_2_272_len28671.json | qa_2 | 272 | 218 | 28671 | Catwoman |
| qa_2_273_len31791.json | qa_2 | 273 | 238 | 31791 | 7 miles |
| qa_2_274_len31221.json | qa_2 | 274 | 228 | 31221 | Laura Ann Osnes |
| qa_2_275_len30770.json | qa_2 | 275 | 228 | 30770 | 20 |
| qa_2_276_len30627.json | qa_2 | 276 | 228 | 30627 | Eucryphia |
| qa_2_277_len32523.json | qa_2 | 277 | 238 | 32523 | Venice |
| qa_2_278_len32511.json | qa_2 | 278 | 238 | 32511 | Waimea Bay |
| qa_2_279_len31696.json | qa_2 | 279 | 228 | 31696 | evangelical Christian periodical |
| qa_2_280_len31845.json | qa_2 | 280 | 238 | 31845 | James Franco |
| qa_2_281_len32508.json | qa_2 | 281 | 238 | 32508 | DI Humphrey Goodman |
| qa_2_282_len30671.json | qa_2 | 282 | 228 | 30671 | February |
| qa_2_283_len29523.json | qa_2 | 283 | 208 | 29523 | Sacred Planet |
| qa_2_284_len30416.json | qa_2 | 284 | 228 | 30416 | Hanna |
| qa_2_285_len32363.json | qa_2 | 285 | 238 | 32363 | Ferdi Taygan |
| qa_2_286_len31811.json | qa_2 | 286 | 228 | 31811 | paracyclist |
| qa_2_287_len31209.json | qa_2 | 287 | 218 | 31209 | The Dragon School in Oxford |
| qa_2_288_len32482.json | qa_2 | 288 | 228 | 32482 | Suicide Kings |
| qa_2_289_len32306.json | qa_2 | 289 | 238 | 32306 | Hey Pa! There's a Goat on the Roof |
| qa_2_290_len32438.json | qa_2 | 290 | 238 | 32438 | a record of 13–3 |
| qa_2_291_len30709.json | qa_2 | 291 | 208 | 30709 | 8,211 at the 2010 census |
| qa_2_292_len32718.json | qa_2 | 292 | 238 | 32718 | The Crowned Prince of the Philadelphia Mob |
| qa_2_293_len31209.json | qa_2 | 293 | 218 | 31209 | supervisor of the royal works |
| qa_2_294_len31094.json | qa_2 | 294 | 238 | 31094 | Janet Damita Jo Jackson |
| qa_2_295_len29867.json | qa_2 | 295 | 218 | 29867 | Continental Army |
| qa_2_296_len30731.json | qa_2 | 296 | 218 | 30731 | Macaronesia |
| qa_2_297_len32181.json | qa_2 | 297 | 228 | 32181 | Treaty of Gandamak |
| qa_2_298_len32651.json | qa_2 | 298 | 238 | 32651 | Oregon |
| qa_2_299_len32200.json | qa_2 | 299 | 228 | 32200 | early 1970s |
| qa_2_300_len32724.json | qa_2 | 300 | 238 | 32724 | Bury St Edmunds, |
| qa_2_301_len32422.json | qa_2 | 301 | 228 | 32422 | early 20s |
| qa_2_302_len32008.json | qa_2 | 302 | 228 | 32008 | 23 November 1946 |
| qa_2_303_len29376.json | qa_2 | 303 | 218 | 29376 | The Soloist |
| qa_2_304_len32365.json | qa_2 | 304 | 228 | 32365 | no |
| qa_2_305_len30792.json | qa_2 | 305 | 228 | 30792 | 1901 |
| qa_2_306_len31171.json | qa_2 | 306 | 218 | 31171 | Bolton |
| qa_2_307_len31709.json | qa_2 | 307 | 238 | 31709 | George Cayley |
| qa_2_308_len31374.json | qa_2 | 308 | 228 | 31374 | The Leonberger is a giant dog breed. |
| qa_2_309_len32283.json | qa_2 | 309 | 238 | 32283 | yes |
| qa_2_310_len32595.json | qa_2 | 310 | 238 | 32595 | 2013 |
| qa_2_311_len31753.json | qa_2 | 311 | 218 | 31753 | Rolling Stone |
| qa_2_312_len32192.json | qa_2 | 312 | 228 | 32192 | Battle Creek, Michigan |
| qa_2_313_len32236.json | qa_2 | 313 | 228 | 32236 | Am Rong |
| qa_2_314_len31720.json | qa_2 | 314 | 218 | 31720 | 29 |
| qa_2_315_len31422.json | qa_2 | 315 | 238 | 31422 | Bay Ridge, Brooklyn |
| qa_2_316_len31797.json | qa_2 | 316 | 228 | 31797 | Northumbrian |
| qa_2_317_len31695.json | qa_2 | 317 | 228 | 31695 | Oldham County |
| qa_2_318_len32593.json | qa_2 | 318 | 228 | 32593 | no |
| qa_2_319_len30442.json | qa_2 | 319 | 228 | 30442 | The Saturdays |
| qa_2_320_len30695.json | qa_2 | 320 | 228 | 30695 | Serri |
| qa_2_321_len32063.json | qa_2 | 321 | 238 | 32063 | international football competition |
| qa_2_322_len32197.json | qa_2 | 322 | 228 | 32197 | 1944 |
| qa_2_323_len32077.json | qa_2 | 323 | 238 | 32077 | Wilmette |
| qa_2_324_len30779.json | qa_2 | 324 | 228 | 30779 | Oedipus Rex |
| qa_2_325_len32041.json | qa_2 | 325 | 238 | 32041 | Steve Jobs |
| qa_2_326_len32297.json | qa_2 | 326 | 238 | 32297 | Kodos |
| qa_2_327_len32637.json | qa_2 | 327 | 238 | 32637 | My Beautiful Dark Twisted Fantasy |
| qa_2_328_len30871.json | qa_2 | 328 | 228 | 30871 | yes |
| qa_2_329_len32392.json | qa_2 | 329 | 238 | 32392 | Spring city |
| qa_2_330_len31842.json | qa_2 | 330 | 228 | 31842 | Philadelphia |
| qa_2_331_len32686.json | qa_2 | 331 | 238 | 32686 | a minor basilica |
| qa_2_332_len30753.json | qa_2 | 332 | 218 | 30753 | 150 million copies |
| qa_2_333_len32494.json | qa_2 | 333 | 238 | 32494 | Aqua |
| qa_2_334_len32334.json | qa_2 | 334 | 238 | 32334 | 1964 |
| qa_2_335_len32363.json | qa_2 | 335 | 228 | 32363 | CBS |
| qa_2_336_len32139.json | qa_2 | 336 | 238 | 32139 | Pakistani |
| qa_2_337_len30906.json | qa_2 | 337 | 218 | 30906 | The Four Seasons |
| qa_2_338_len30951.json | qa_2 | 338 | 228 | 30951 | no |
| qa_2_339_len31407.json | qa_2 | 339 | 228 | 31407 | 76,416 |
| qa_2_340_len30014.json | qa_2 | 340 | 218 | 30014 | Matt Flynn |
| qa_2_341_len32343.json | qa_2 | 341 | 238 | 32343 | Happy Death Day |
| qa_2_342_len30790.json | qa_2 | 342 | 228 | 30790 | Gerard "Gerry" Adams |
| qa_2_343_len30744.json | qa_2 | 343 | 228 | 30744 | The Battle of Belleau Wood |
| qa_2_344_len31287.json | qa_2 | 344 | 228 | 31287 | Australian Defence Force |
| qa_2_345_len31815.json | qa_2 | 345 | 238 | 31815 | about 450,000 |
| qa_2_346_len30274.json | qa_2 | 346 | 228 | 30274 | West Point Iron and Cannon Foundry |
| qa_2_347_len31704.json | qa_2 | 347 | 238 | 31704 | Jack Ryan |
| qa_2_348_len32642.json | qa_2 | 348 | 238 | 32642 | tomato |
| qa_2_349_len29993.json | qa_2 | 349 | 218 | 29993 | 2009 |
| qa_2_350_len32279.json | qa_2 | 350 | 238 | 32279 | September 8, 2017 |
| qa_2_351_len28661.json | qa_2 | 351 | 218 | 28661 | Edward Franklin Albee III |
| qa_2_352_len32240.json | qa_2 | 352 | 238 | 32240 | yes |
| qa_2_353_len31532.json | qa_2 | 353 | 238 | 31532 | impresario |
| qa_2_354_len31734.json | qa_2 | 354 | 238 | 31734 | no |
| qa_2_355_len32062.json | qa_2 | 355 | 238 | 32062 | no |
| qa_2_356_len31977.json | qa_2 | 356 | 228 | 31977 | Worcester County |
| qa_2_357_len31733.json | qa_2 | 357 | 228 | 31733 | Farrell |
| qa_2_358_len31384.json | qa_2 | 358 | 228 | 31384 | modern-day New York City |
| qa_2_359_len30549.json | qa_2 | 359 | 228 | 30549 | Dan Lin, Roy Lee, Phil Lord and Christopher Miller |
| qa_2_360_len30657.json | qa_2 | 360 | 218 | 30657 | reigning monarch of the United Kingdom |
| qa_2_361_len32689.json | qa_2 | 361 | 228 | 32689 | Saint Louis County |
| qa_2_362_len30847.json | qa_2 | 362 | 218 | 30847 | Phil Collins |
| qa_2_363_len30487.json | qa_2 | 363 | 228 | 30487 | video game |
| qa_2_364_len31579.json | qa_2 | 364 | 228 | 31579 | Cambridge University |
| qa_2_365_len31586.json | qa_2 | 365 | 238 | 31586 | Love Streams |
| qa_2_366_len32034.json | qa_2 | 366 | 238 | 32034 | Lowe's |
| qa_2_367_len30901.json | qa_2 | 367 | 228 | 30901 | first freshman to finish as the runner-up |
| qa_2_368_len30200.json | qa_2 | 368 | 228 | 30200 | 1987 |
| qa_2_369_len30400.json | qa_2 | 369 | 218 | 30400 | James Packer |
| qa_2_370_len31597.json | qa_2 | 370 | 228 | 31597 | documentary |
| qa_2_371_len32331.json | qa_2 | 371 | 228 | 32331 | England |
| qa_2_372_len32423.json | qa_2 | 372 | 228 | 32423 | Jung Yun-ho |
| qa_2_373_len30769.json | qa_2 | 373 | 228 | 30769 | Hot toddy |
| qa_2_374_len32178.json | qa_2 | 374 | 238 | 32178 | 2017 |
| qa_2_375_len32192.json | qa_2 | 375 | 238 | 32192 | The Late Late Show |
| qa_2_376_len31632.json | qa_2 | 376 | 228 | 31632 | Hudson Bay Mining and Smelting Company |
| qa_2_377_len31459.json | qa_2 | 377 | 228 | 31459 | Scandinavian design |
| qa_2_378_len30182.json | qa_2 | 378 | 228 | 30182 | Constantine |
| qa_2_379_len32018.json | qa_2 | 379 | 238 | 32018 | comparable to the seven Wonders of the World |
| qa_2_380_len30800.json | qa_2 | 380 | 238 | 30800 | José Echegaray y Eizaguirre |
| qa_2_381_len32400.json | qa_2 | 381 | 238 | 32400 | Knowsley United Women |
| qa_2_382_len31593.json | qa_2 | 382 | 228 | 31593 | Blue |
| qa_2_383_len32156.json | qa_2 | 383 | 228 | 32156 | Valhalla Highlands Historic District |
| qa_2_384_len31426.json | qa_2 | 384 | 238 | 31426 | Philip K. Dick |
| qa_2_385_len30290.json | qa_2 | 385 | 218 | 30290 | New York Giants |
| qa_2_386_len32496.json | qa_2 | 386 | 218 | 32496 | 1961 |
| qa_2_387_len30670.json | qa_2 | 387 | 228 | 30670 | Copenhagen |
| qa_2_388_len29211.json | qa_2 | 388 | 218 | 29211 | middle America |
| qa_2_389_len32586.json | qa_2 | 389 | 228 | 32586 | Soha Ali Khan |
| qa_2_390_len31363.json | qa_2 | 390 | 228 | 31363 | South Korean horror film |
| qa_2_391_len32544.json | qa_2 | 391 | 228 | 32544 | Lucas Stephen Grabeel |
| qa_2_392_len32501.json | qa_2 | 392 | 238 | 32501 | American supernatural psychological horror film |
| qa_2_393_len32624.json | qa_2 | 393 | 238 | 32624 | Essex |
| qa_2_394_len30477.json | qa_2 | 394 | 218 | 30477 | Peter Hall |
| qa_2_395_len32099.json | qa_2 | 395 | 218 | 32099 | French Open |
| qa_2_396_len32258.json | qa_2 | 396 | 238 | 32258 | Bismarck |
| qa_2_397_len32712.json | qa_2 | 397 | 238 | 32712 | Premier League club Manchester United and the England nation |
| qa_2_398_len32012.json | qa_2 | 398 | 238 | 32012 | Massive Entertainment |
| qa_2_399_len31813.json | qa_2 | 399 | 238 | 31813 | Portal A Interactive |
| qa_2_400_len30106.json | qa_2 | 400 | 228 | 30106 | King George IV |
| qa_2_401_len30823.json | qa_2 | 401 | 228 | 30823 | Icehouse pieces |
| qa_2_402_len30339.json | qa_2 | 402 | 218 | 30339 | ZZ Top |
| qa_2_403_len31037.json | qa_2 | 403 | 218 | 31037 | 1977 |
| qa_2_404_len30171.json | qa_2 | 404 | 218 | 30171 | 1963 to 1991 |
| qa_2_405_len32716.json | qa_2 | 405 | 228 | 32716 | Laurel, Mississippi |
| qa_2_406_len32458.json | qa_2 | 406 | 228 | 32458 | arose on the "Playboy" magazine pattern in the 1970s. |
| qa_2_407_len31436.json | qa_2 | 407 | 228 | 31436 | Langston Hughes |
| qa_2_408_len31901.json | qa_2 | 408 | 228 | 31901 | 728,000 ft² |
| qa_2_409_len31811.json | qa_2 | 409 | 228 | 31811 | Royce da 5'9" (Bad) and Eminem (Evil) |
| qa_2_410_len29792.json | qa_2 | 410 | 218 | 29792 | Shanghai |
| qa_2_411_len31966.json | qa_2 | 411 | 238 | 31966 | Terry Gilliam |
| qa_2_412_len32305.json | qa_2 | 412 | 238 | 32305 | Lithuanian-born French |
| qa_2_413_len32251.json | qa_2 | 413 | 238 | 32251 | a family member |
| qa_2_414_len31315.json | qa_2 | 414 | 218 | 31315 | Douglas Jackson |
| qa_2_415_len32130.json | qa_2 | 415 | 228 | 32130 | Dirty Pretty Things |
| qa_2_416_len31249.json | qa_2 | 416 | 228 | 31249 | The Itchy & Scratchy Show |
| qa_2_417_len30869.json | qa_2 | 417 | 218 | 30869 | Viaport Rotterdam |
| qa_2_418_len31699.json | qa_2 | 418 | 228 | 31699 | signing free agent Chauncey Billups |
| qa_2_419_len30680.json | qa_2 | 419 | 218 | 30680 | 51,271 |
| qa_2_420_len31897.json | qa_2 | 420 | 238 | 31897 | no |
| qa_2_421_len32081.json | qa_2 | 421 | 228 | 32081 | documentary |
| qa_2_422_len30281.json | qa_2 | 422 | 228 | 30281 | fifteen |
| qa_2_423_len31789.json | qa_2 | 423 | 228 | 31789 | defensive coordinator |
| qa_2_424_len31814.json | qa_2 | 424 | 238 | 31814 | 1.95 m |
| qa_2_425_len32263.json | qa_2 | 425 | 228 | 32263 | MGM Resorts International |
| qa_2_426_len32571.json | qa_2 | 426 | 238 | 32571 | Bedknobs and Broomsticks |
| qa_2_427_len30073.json | qa_2 | 427 | 228 | 30073 | Holston River |
| qa_2_428_len31366.json | qa_2 | 428 | 228 | 31366 | British Conservative Party |
| qa_2_429_len30809.json | qa_2 | 429 | 228 | 30809 | Wes Archer |
| qa_2_430_len30063.json | qa_2 | 430 | 218 | 30063 | Geelong Football Club |
| qa_2_431_len31766.json | qa_2 | 431 | 228 | 31766 | no |
| qa_2_432_len31899.json | qa_2 | 432 | 238 | 31899 | Delmer Lawrence Daves |
| qa_2_433_len30921.json | qa_2 | 433 | 238 | 30921 | X Ambassadors |
| qa_2_434_len31739.json | qa_2 | 434 | 228 | 31739 | Walt Disney Productions |
| qa_2_435_len31935.json | qa_2 | 435 | 228 | 31935 | 2844 km |
| qa_2_436_len29289.json | qa_2 | 436 | 218 | 29289 | located in Olathe, Kansas |
| qa_2_437_len31634.json | qa_2 | 437 | 228 | 31634 | "The Ring" magazine |
| qa_2_438_len31155.json | qa_2 | 438 | 228 | 31155 | 1909 |
| qa_2_439_len31826.json | qa_2 | 439 | 238 | 31826 | Sun Valley, Idaho |
| qa_2_440_len30894.json | qa_2 | 440 | 228 | 30894 | President of the United States |
| qa_2_441_len32504.json | qa_2 | 441 | 228 | 32504 | The Bears and I |
| qa_2_442_len32307.json | qa_2 | 442 | 228 | 32307 | Patterns of Sexual Behavior |
| qa_2_443_len30994.json | qa_2 | 443 | 218 | 30994 | Larry Richard Drake |
| qa_2_444_len31171.json | qa_2 | 444 | 218 | 31171 | Bedknobs and Broomsticks |
| qa_2_445_len32294.json | qa_2 | 445 | 238 | 32294 | 46 Albert Square, Walford, London E20 |
| qa_2_446_len30741.json | qa_2 | 446 | 228 | 30741 | Darkroom |
| qa_2_447_len30964.json | qa_2 | 447 | 228 | 30964 | Phoenix Television |
| qa_2_448_len32129.json | qa_2 | 448 | 238 | 32129 | Planet Terror |
| qa_2_449_len31097.json | qa_2 | 449 | 218 | 31097 | "Peshwa" (Prime Minister) |
| qa_2_450_len32073.json | qa_2 | 450 | 228 | 32073 | Donald Trump Jr. |
| qa_2_451_len31098.json | qa_2 | 451 | 218 | 31098 | Royal Governor Lord Dunmore |
| qa_2_452_len31288.json | qa_2 | 452 | 238 | 31288 | Graduados |
| qa_2_453_len31527.json | qa_2 | 453 | 238 | 31527 | Private Secretary and Treasurer |
| qa_2_454_len32323.json | qa_2 | 454 | 228 | 32323 | no |
| qa_2_455_len32629.json | qa_2 | 455 | 238 | 32629 | Ogallala Aquifer |
| qa_2_456_len30405.json | qa_2 | 456 | 218 | 30405 | ibnʿArabī |
| qa_2_457_len30636.json | qa_2 | 457 | 228 | 30636 | various deities, beings, and heroes |
| qa_2_458_len30336.json | qa_2 | 458 | 218 | 30336 | 1984 South Asian Games |
| qa_2_459_len32590.json | qa_2 | 459 | 238 | 32590 | Viscount Cranborne |
| qa_2_460_len31300.json | qa_2 | 460 | 228 | 31300 | Kansas City crime family |
| qa_2_461_len32396.json | qa_2 | 461 | 238 | 32396 | Albert Park |
| qa_2_462_len32626.json | qa_2 | 462 | 238 | 32626 | in 1896 |
| qa_2_463_len32603.json | qa_2 | 463 | 228 | 32603 | Unbreakable |
| qa_2_464_len32563.json | qa_2 | 464 | 228 | 32563 | Apatosaurus |
| qa_2_465_len32400.json | qa_2 | 465 | 238 | 32400 | BraveStarr |
| qa_2_466_len32409.json | qa_2 | 466 | 228 | 32409 | Sanskrit |
| qa_2_467_len31961.json | qa_2 | 467 | 228 | 31961 | Battle of Guam |
| qa_2_468_len32109.json | qa_2 | 468 | 228 | 32109 | Martha Coolidge |
| qa_2_469_len31172.json | qa_2 | 469 | 228 | 31172 | Mr. Church |
| qa_2_470_len31027.json | qa_2 | 470 | 228 | 31027 | director |
| qa_2_471_len31395.json | qa_2 | 471 | 238 | 31395 | Executive Search, Board Consulting |
| qa_2_472_len31297.json | qa_2 | 472 | 228 | 31297 | author |
| qa_2_473_len32728.json | qa_2 | 473 | 228 | 32728 | The Catholic Church in Ireland |
| qa_2_474_len30988.json | qa_2 | 474 | 228 | 30988 | Edmonton, Alberta |
| qa_2_475_len32191.json | qa_2 | 475 | 238 | 32191 | Charles Joseph |
| qa_2_476_len32476.json | qa_2 | 476 | 238 | 32476 | NCAA Division I FBS football |
| qa_2_477_len31917.json | qa_2 | 477 | 228 | 31917 | Tea Moderna |
| qa_2_478_len31602.json | qa_2 | 478 | 228 | 31602 | Armenian |
| qa_2_479_len31352.json | qa_2 | 479 | 218 | 31352 | Kathleen O'Brien |
| qa_2_480_len30469.json | qa_2 | 480 | 228 | 30469 | Warner Bros |
| qa_2_481_len32538.json | qa_2 | 481 | 238 | 32538 | Motörhead |
| qa_2_482_len32153.json | qa_2 | 482 | 238 | 32153 | commentary on Isaac Newton's book "Principia" |
| qa_2_483_len32469.json | qa_2 | 483 | 228 | 32469 | no |
| qa_2_484_len30121.json | qa_2 | 484 | 228 | 30121 | four months in jail |
| qa_2_485_len31805.json | qa_2 | 485 | 238 | 31805 | 1963 |
| qa_2_486_len29655.json | qa_2 | 486 | 218 | 29655 | rock |
| qa_2_487_len28811.json | qa_2 | 487 | 228 | 28811 | Gateways |
| qa_2_488_len32737.json | qa_2 | 488 | 238 | 32737 | zona glomerulosa |
| qa_2_489_len30469.json | qa_2 | 489 | 218 | 30469 | 33 |
| qa_2_490_len32711.json | qa_2 | 490 | 228 | 32711 | Mississippi University for Women |
| qa_2_491_len32494.json | qa_2 | 491 | 228 | 32494 | Stacey Kent |
| qa_2_492_len31992.json | qa_2 | 492 | 218 | 31992 | 1966 |
| qa_2_493_len32699.json | qa_2 | 493 | 228 | 32699 | DTM |
| qa_2_494_len31659.json | qa_2 | 494 | 228 | 31659 | 2002 |
| qa_2_495_len32069.json | qa_2 | 495 | 218 | 32069 | 4,613 |
| qa_2_496_len32405.json | qa_2 | 496 | 228 | 32405 | lower Manhattan |
| qa_2_497_len31175.json | qa_2 | 497 | 228 | 31175 | 137th |
| qa_2_498_len30886.json | qa_2 | 498 | 218 | 30886 | Ronald Wilson Reagan |
| qa_2_499_len31501.json | qa_2 | 499 | 228 | 31501 | documentary |

qa_2 长度分布：min=28661, 中位=31724, max=32737, 均值=31545

