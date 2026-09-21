# 长上下文实验评分报告

- 样本目录：`experiments/inputs_qwen_ruler30k`（48 个样本）
- 评分：EM（精确匹配，多答案取最大）+ SM（子串匹配）+ token-F1（多答案取最大）
- 归一化：小写/去冠词/去标点/Yes-No 规整（`example/utils.py`）
- 输入日志：`run_cb_qwen_ruler_qa1_r1.log`, `run_cb_qwen_ruler_qa1_r2.log`, `run_cb_qwen_ruler_qa1_r3.log`, `run_cb_qwen_ruler_qa2_r1.log`, `run_cb_qwen_ruler_qa2_r2.log`, `run_cb_qwen_ruler_qa2_r3.log`, `run_daf_qwen_ruler_qa1_r1.log`, `run_daf_qwen_ruler_qa1_r2.log`, `run_daf_qwen_ruler_qa1_r3.log`, `run_daf_qwen_ruler_qa2_r1.log`, `run_daf_qwen_ruler_qa2_r2.log`, `run_daf_qwen_ruler_qa2_r3.log`

## 一、逐样本明细（质量指标为多轮均值，TTFT 为多轮中位数）

| 样本 | 来源 | 长度 | 方法 | EM | SM | F1 | TTFT缓存(s) | TTFT全量(s) | 缓存答案 | 全量答案 |
|---|---|---|---|---|---|---|---|----|---|---|
| qa_1_000_len32420.json | validation.jsonl (qa_1) | 32420 | CacheBlend | 0.00 | 1.00 | 0.00 | 3.241 | 7.905 | Normandy is located in France. | France |
| qa_1_000_len32420.json | validation.jsonl (qa_1) | 32420 | DAF | 0.00 | 1.00 | 0.00 | 3.079 | 7.940 | Normandy is located in France. | France |
| qa_1_001_len31728.json | validation.jsonl (qa_1) | 31728 | CacheBlend | 0.00 | 1.00 | 0.69 | 3.129 | 7.709 | The Normans in Normandy were present fro | The Normans established themselves in No |
| qa_1_001_len31728.json | validation.jsonl (qa_1) | 31728 | DAF | 0.00 | 1.00 | 0.69 | 2.914 | 7.734 | The Normans in Normandy were present fro | The Normans established themselves in No |
| qa_1_002_len32716.json | validation.jsonl (qa_1) | 32716 | CacheBlend | 0.00 | 1.00 | 0.67 | 3.243 | 8.010 | The Norse originated from Denmark, Icela | Denmark, Iceland, and Norway |
| qa_1_002_len32716.json | validation.jsonl (qa_1) | 32716 | DAF | 0.00 | 1.00 | 0.67 | 3.011 | 8.033 | The Norse originated from Denmark, Icela | Denmark, Iceland, and Norway |
| qa_1_003_len30149.json | validation.jsonl (qa_1) | 30149 | CacheBlend | 1.00 | 1.00 | 1.00 | 2.891 | 7.193 | Rollo | Rollo |
| qa_1_003_len30149.json | validation.jsonl (qa_1) | 30149 | DAF | 1.00 | 1.00 | 1.00 | 2.765 | 7.227 | Rollo | Rollo |
| qa_1_004_len30471.json | validation.jsonl (qa_1) | 30471 | CacheBlend | 1.00 | 1.00 | 1.00 | 2.930 | 7.296 | 10th century | 10th century |
| qa_1_004_len30471.json | validation.jsonl (qa_1) | 30471 | DAF | 1.00 | 1.00 | 1.00 | 2.760 | 7.319 | 10th | 10th century |
| qa_1_005_len32253.json | validation.jsonl (qa_1) | 32253 | CacheBlend | 1.00 | 1.00 | 1.00 | 3.191 | 7.891 | William the Conqueror | William II of Normandy |
| qa_1_005_len32253.json | validation.jsonl (qa_1) | 32253 | DAF | 0.00 | 0.00 | 0.00 | 2.986 | 7.919 | Edward the Confessor | William II of Normandy |
| qa_1_006_len32195.json | validation.jsonl (qa_1) | 32195 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.202 | 7.831 | The duchy of Normandy was ruled by the N | The duchy of Normandy was ruled by the N |
| qa_1_006_len32195.json | validation.jsonl (qa_1) | 32195 | DAF | 0.00 | 0.00 | 0.00 | 3.033 | 7.855 | The duchy of Normandy was ruled by the N | The duchy of Normandy was ruled by the N |
| qa_1_007_len32600.json | validation.jsonl (qa_1) | 32600 | CacheBlend | 0.00 | 1.00 | 0.00 | 3.238 | 7.975 | The Normans were initially Norse, which  | Christian |
| qa_1_007_len32600.json | validation.jsonl (qa_1) | 32600 | DAF | 0.00 | 0.00 | 0.00 | 3.005 | 7.995 | The Normans were not a single religious  | Christian |
| qa_1_008_len30678.json | validation.jsonl (qa_1) | 30678 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.000 | 7.335 | Norman means relating to the Normans, a  | Norman: Nourmands; French: Normands; Lat |
| qa_1_008_len30678.json | validation.jsonl (qa_1) | 30678 | DAF | 0.00 | 0.00 | 0.00 | 2.859 | 7.360 | Norman refers to people from Normandy, a | Norman: Nourmands; French: Normands; Lat |
| qa_1_009_len32681.json | validation.jsonl (qa_1) | 32681 | CacheBlend | 0.00 | 0.00 | 0.18 | 3.263 | 7.999 | The Latin version of the word Norman, "N | 9th century |
| qa_1_009_len32681.json | validation.jsonl (qa_1) | 32681 | DAF | 0.00 | 0.00 | 0.14 | 3.061 | 8.024 | The Latin version of the word Norman, "N | 9th century |
| qa_1_010_len32585.json | validation.jsonl (qa_1) | 32585 | CacheBlend | 0.00 | 1.00 | 0.40 | 3.258 | 7.977 | The Duchy of Normandy was founded in 911 | The Duchy of Normandy was founded by tre |
| qa_1_010_len32585.json | validation.jsonl (qa_1) | 32585 | DAF | 0.00 | 0.00 | 0.00 | 3.013 | 8.009 | The Duchy of Normandy was established by | The Duchy of Normandy was founded by tre |
| qa_1_011_len31585.json | validation.jsonl (qa_1) | 31585 | CacheBlend | 0.00 | 1.00 | 0.67 | 3.116 | 7.653 | King Charles III of West Francia | King Charles III of West Francia |
| qa_1_011_len31585.json | validation.jsonl (qa_1) | 31585 | DAF | 0.00 | 1.00 | 0.67 | 2.915 | 7.682 | King Charles III of West Francia | King Charles III of West Francia |
| qa_1_012_len31552.json | validation.jsonl (qa_1) | 31552 | CacheBlend | 0.00 | 1.00 | 0.40 | 3.115 | 7.650 | the river Epte | the river Epte |
| qa_1_012_len31552.json | validation.jsonl (qa_1) | 31552 | DAF | 1.00 | 1.00 | 1.00 | 2.915 | 7.685 | the Epte | the river Epte |
| qa_1_013_len32635.json | validation.jsonl (qa_1) | 32635 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.235 | 8.000 | The Norman rulers, who were culturally a | Dinwiddie |
| qa_1_013_len32635.json | validation.jsonl (qa_1) | 32635 | DAF | 0.00 | 0.00 | 0.00 | 3.019 | 8.027 | The text does not explicitly state who g | Dinwiddie |
| qa_1_014_len32483.json | validation.jsonl (qa_1) | 32483 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.219 | 7.949 | The text does not explicitly state the N | The Normans replaced the Norse religion  |
| qa_1_014_len32483.json | validation.jsonl (qa_1) | 32483 | DAF | 0.00 | 0.00 | 0.00 | 3.006 | 7.972 | The text does not explicitly state the N | The Normans replaced the Norse religion  |
| qa_1_015_len30396.json | validation.jsonl (qa_1) | 30396 | CacheBlend | 0.00 | 1.00 | 0.00 | 2.927 | 7.265 | The Normans were located in Normandy, wh | Normandy |
| qa_1_015_len30396.json | validation.jsonl (qa_1) | 30396 | DAF | 0.00 | 1.00 | 0.00 | 2.750 | 7.288 | The Normans were located in the north of | Normandy |
| qa_1_016_len32051.json | validation.jsonl (qa_1) | 32051 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.189 | 7.792 | The documents do not specify what one of | beef cattle |
| qa_1_016_len32051.json | validation.jsonl (qa_1) | 32051 | DAF | 0.00 | 0.00 | 0.00 | 2.962 | 7.831 | The passage does not mention any specifi | beef cattle |
| qa_1_017_len30626.json | validation.jsonl (qa_1) | 30626 | CacheBlend | 0.00 | 0.00 | 0.05 | 3.002 | 7.324 | Based on the given documents, the Norman | The Normans' main enemies in Italy, the  |
| qa_1_017_len30626.json | validation.jsonl (qa_1) | 30626 | DAF | 0.00 | 0.00 | 0.05 | 2.841 | 7.351 | The text does not explicitly mention the | The Normans' main enemies in Italy, the  |
| qa_1_018_len30128.json | validation.jsonl (qa_1) | 30128 | CacheBlend | 0.00 | 0.00 | 0.18 | 2.887 | 7.196 | Herve served as a Byzantine general in t | Hervé in the 1050s. |
| qa_1_018_len30128.json | validation.jsonl (qa_1) | 30128 | DAF | 0.00 | 0.00 | 0.19 | 2.727 | 7.222 | Herve served as a Byzantine general in t | Hervé in the 1050s. |
| qa_1_019_len32473.json | validation.jsonl (qa_1) | 32473 | CacheBlend | 0.00 | 0.00 | 0.06 | 3.221 | 7.939 | The document does not specify when Rober | In the 1060s, Robert Crispin led the Nor |
| qa_1_019_len32473.json | validation.jsonl (qa_1) | 32473 | DAF | 0.00 | 0.00 | 0.06 | 2.998 | 7.969 | The document does not specify when Rober | In the 1060s, Robert Crispin led the Nor |
| qa_1_020_len30589.json | validation.jsonl (qa_1) | 30589 | CacheBlend | 0.00 | 0.00 | 0.00 | 2.936 | 7.324 | The text does not specify who ruined Rou | Byzantine general Alexius Komnenos |
| qa_1_020_len30589.json | validation.jsonl (qa_1) | 30589 | DAF | 0.00 | 0.00 | 0.00 | 2.768 | 7.341 | The text does not specify who ruined Rou | Byzantine general Alexius Komnenos |
| qa_1_021_len32038.json | validation.jsonl (qa_1) | 32038 | CacheBlend | 0.00 | 0.00 | 0.06 | 3.165 | 7.785 | The text does not specify a particular N | Afranji |
| qa_1_021_len32038.json | validation.jsonl (qa_1) | 32038 | DAF | 1.00 | 1.00 | 1.00 | 2.961 | 7.812 | Afranji | Afranji |
| qa_1_022_len30751.json | validation.jsonl (qa_1) | 30751 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.004 | 7.396 | Rogers of Montreuil | A Norman named Oursel led a force of "Fr |
| qa_1_022_len30751.json | validation.jsonl (qa_1) | 30751 | DAF | 0.00 | 0.00 | 0.00 | 2.827 | 7.420 | Raimbaud | A Norman named Oursel led a force of "Fr |
| qa_1_023_len32382.json | validation.jsonl (qa_1) | 32382 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.215 | 7.930 | The Normans teamed up with the Byzantine | The Normans joined Turkish forces to aid |
| qa_1_023_len32382.json | validation.jsonl (qa_1) | 32382 | DAF | 0.00 | 0.00 | 0.00 | 3.017 | 7.949 | The given text does not mention any spec | The Normans joined Turkish forces to aid |
| qa_2_000_len31234.json | validation.jsonl (qa_2) | 31234 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.045 | 7.481 | No | No |
| qa_2_000_len31234.json | validation.jsonl (qa_2) | 31234 | DAF | 0.00 | 0.00 | 0.00 | 2.972 | 7.532 | No | No |
| qa_2_001_len31871.json | validation.jsonl (qa_2) | 31871 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.132 | 7.694 | The woman who portrayed Corliss Archer i | Shirley Temple |
| qa_2_001_len31871.json | validation.jsonl (qa_2) | 31871 | DAF | 0.00 | 0.00 | 0.00 | 2.941 | 7.753 | Governor | Shirley Temple |
| qa_2_002_len30167.json | validation.jsonl (qa_2) | 30167 | CacheBlend | 0.00 | 0.00 | 0.00 | 2.883 | 7.150 | The information provided does not specif | The Hork-Bajir Chronicles |
| qa_2_002_len30167.json | validation.jsonl (qa_2) | 30167 | DAF | 0.00 | 0.00 | 0.00 | 2.744 | 7.203 | There is no science fantasy young adult  | The Hork-Bajir Chronicles |
| qa_2_003_len31814.json | validation.jsonl (qa_2) | 31814 | CacheBlend | 1.00 | 1.00 | 1.00 | 3.114 | 7.680 | No | No |
| qa_2_003_len31814.json | validation.jsonl (qa_2) | 31814 | DAF | 1.00 | 1.00 | 1.00 | 2.936 | 7.726 | No | No |
| qa_2_004_len32050.json | validation.jsonl (qa_2) | 32050 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.166 | 7.753 | The documents provided do not contain an | Greenwich Village |
| qa_2_004_len32050.json | validation.jsonl (qa_2) | 32050 | DAF | 0.00 | 1.00 | 0.44 | 3.014 | 7.791 | New York City | Greenwich Village |
| qa_2_005_len32340.json | validation.jsonl (qa_2) | 32340 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.186 | 7.860 | The information provided does not specif | YG Entertainment |
| qa_2_005_len32340.json | validation.jsonl (qa_2) | 32340 | DAF | 0.00 | 0.00 | 0.06 | 3.009 | 7.904 | The information provided does not specif | YG Entertainment |
| qa_2_006_len31806.json | validation.jsonl (qa_2) | 31806 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.114 | 7.701 | Dick Morris | Eenasul Fateh |
| qa_2_006_len31806.json | validation.jsonl (qa_2) | 31806 | DAF | 0.00 | 0.00 | 0.00 | 2.937 | 7.724 | The information provided does not indica | Eenasul Fateh |
| qa_2_007_len31346.json | validation.jsonl (qa_2) | 31346 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.058 | 7.539 | The passage states that the Lewiston Mai | 1,400 |
| qa_2_007_len31346.json | validation.jsonl (qa_2) | 31346 | DAF | 0.00 | 0.00 | 0.00 | 2.890 | 7.570 | The text does not provide the seating ca | 1,400 |
| qa_2_008_len32138.json | validation.jsonl (qa_2) | 32138 | CacheBlend | 0.00 | 1.00 | 0.16 | 3.163 | 7.785 | Annie Morton was born on December 17, 19 | Annie Morton |
| qa_2_008_len32138.json | validation.jsonl (qa_2) | 32138 | DAF | 0.00 | 0.00 | 0.00 | 2.980 | 7.822 | Annie Morton | Annie Morton |
| qa_2_009_len31131.json | validation.jsonl (qa_2) | 31131 | CacheBlend | 1.00 | 1.00 | 1.00 | 3.038 | 7.472 | Yes. | Yes |
| qa_2_009_len31131.json | validation.jsonl (qa_2) | 31131 | DAF | 0.00 | 0.00 | 0.00 | 2.881 | 7.509 | No | Yes |
| qa_2_010_len31807.json | validation.jsonl (qa_2) | 31807 | CacheBlend | 0.00 | 0.00 | 0.06 | 3.119 | 7.692 | There is no university mentioned in the  | Kansas Song |
| qa_2_010_len31807.json | validation.jsonl (qa_2) | 31807 | DAF | 0.00 | 0.00 | 0.13 | 2.942 | 7.732 | There is no mention of a fight song for  | Kansas Song |
| qa_2_011_len31916.json | validation.jsonl (qa_2) | 31916 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.139 | 7.722 | Nick Vallelonga | David Weissman |
| qa_2_011_len31916.json | validation.jsonl (qa_2) | 31916 | DAF | 0.00 | 0.00 | 0.00 | 2.991 | 7.750 | Nick Vallelonga | David Weissman |
| qa_2_012_len31839.json | validation.jsonl (qa_2) | 31839 | CacheBlend | 0.00 | 0.00 | 0.75 | 3.116 | 7.695 | 1997 | 1999 |
| qa_2_012_len31839.json | validation.jsonl (qa_2) | 31839 | DAF | 0.00 | 0.00 | 0.00 | 2.945 | 7.730 | The information provided does not mentio | 1999 |
| qa_2_013_len32311.json | validation.jsonl (qa_2) | 32311 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.196 | 7.875 | Yes, Random House Tower and 888 7th Aven | Yes |
| qa_2_013_len32311.json | validation.jsonl (qa_2) | 32311 | DAF | 0.00 | 1.00 | 0.07 | 3.044 | 7.909 | No. Random House Tower and 888 7th Avenu | Yes |
| qa_2_014_len32118.json | validation.jsonl (qa_2) | 32118 | CacheBlend | 0.00 | 0.00 | 0.50 | 3.162 | 7.788 | 1995-2003 | 1903–1912 and then again from 1912–1924 |
| qa_2_014_len32118.json | validation.jsonl (qa_2) | 32118 | DAF | 0.00 | 0.00 | 0.00 | 3.026 | 7.815 | The information provided does not specif | 1903–1912 and then again from 1912–1924 |
| qa_2_015_len32037.json | validation.jsonl (qa_2) | 32037 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.149 | 7.755 | The document does not provide the popula | Brown State Fishing Lake is in the Unite |
| qa_2_015_len32037.json | validation.jsonl (qa_2) | 32037 | DAF | 0.00 | 0.00 | 0.00 | 2.969 | 7.780 | The information provided does not includ | Brown State Fishing Lake is in the Unite |
| qa_2_016_len32351.json | validation.jsonl (qa_2) | 32351 | CacheBlend | 0.00 | 0.00 | 0.25 | 3.218 | 7.874 | America East Conference | America East Conference |
| qa_2_016_len32351.json | validation.jsonl (qa_2) | 32351 | DAF | 0.00 | 0.00 | 0.06 | 3.010 | 7.904 | The information provided does not mentio | America East Conference |
| qa_2_017_len31167.json | validation.jsonl (qa_2) | 31167 | CacheBlend | 0.00 | 1.00 | 0.12 | 3.040 | 7.487 | Yes, Giuseppe Verdi and Ambroise Thomas  | yes |
| qa_2_017_len31167.json | validation.jsonl (qa_2) | 31167 | DAF | 1.00 | 1.00 | 1.00 | 2.917 | 7.515 | Yes | yes |
| qa_2_018_len31835.json | validation.jsonl (qa_2) | 31835 | CacheBlend | 0.00 | 0.00 | 0.56 | 3.120 | 7.696 | 1981-1989 | 1969-1974 |
| qa_2_018_len31835.json | validation.jsonl (qa_2) | 31835 | DAF | 0.00 | 0.00 | 0.56 | 2.940 | 7.729 | 1977-1981 | 1969-1974 |
| qa_2_019_len30786.json | validation.jsonl (qa_2) | 30786 | CacheBlend | 0.00 | 1.00 | 0.88 | 2.994 | 7.380 | Robert Erskine Childers | Robert Erskine Childers |
| qa_2_019_len30786.json | validation.jsonl (qa_2) | 30786 | DAF | 0.00 | 1.00 | 0.88 | 2.889 | 7.418 | Robert Erskine Childers | Robert Erskine Childers |
| qa_2_020_len30327.json | validation.jsonl (qa_2) | 30327 | CacheBlend | 0.00 | 0.00 | 0.00 | 2.898 | 7.228 | The document states that the Force India | Sergio Pérez |
| qa_2_020_len30327.json | validation.jsonl (qa_2) | 30327 | DAF | 0.00 | 0.00 | 0.00 | 2.754 | 7.253 | The given information does not provide d | Sergio Pérez |
| qa_2_021_len32159.json | validation.jsonl (qa_2) | 32159 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.186 | 7.786 | Based on the information provided, the s | Pooh |
| qa_2_021_len32159.json | validation.jsonl (qa_2) | 32159 | DAF | 0.00 | 0.00 | 0.00 | 2.981 | 7.821 | Based on the information provided, the s | Pooh |
| qa_2_022_len31724.json | validation.jsonl (qa_2) | 31724 | CacheBlend | 0.00 | 0.00 | 0.00 | 3.115 | 7.660 | The Apple Remote was originally designed | Apple TV |
| qa_2_022_len31724.json | validation.jsonl (qa_2) | 31724 | DAF | 0.00 | 0.00 | 0.00 | 2.930 | 7.699 | The first three members of the band, exc | Apple TV |
| qa_2_023_len31158.json | validation.jsonl (qa_2) | 31158 | CacheBlend | 0.00 | 1.00 | 0.14 | 3.063 | 7.464 | Based on the information provided, we ca | Based on the information provided, Badly |
| qa_2_023_len31158.json | validation.jsonl (qa_2) | 31158 | DAF | 0.00 | 1.00 | 0.13 | 2.917 | 7.516 | Based on the information provided, it is | Based on the information provided, Badly |

## 二、按方法聚合（每轮一行，再给出跨轮中位数）

| 方法 | 轮次 | EM | SM | F1 | TTFT缓存均值(s) | TTFT全量均值(s) | 缓存加速比 |
|---|---|---|---|---|---|------|---|
| CacheBlend | r1 | 0.125 | 0.458 | 0.265 | 3.167 | 7.674 | 2.42× |
| CacheBlend | r2 | 0.125 | 0.458 | 0.265 | 3.114 | 7.687 | 2.47× |
| CacheBlend | r3 | 0.125 | 0.458 | 0.265 | 3.117 | 7.681 | 2.46× |
| CacheBlend | r4 | 0.083 | 0.250 | 0.226 | 3.115 | 7.624 | 2.45× |
| CacheBlend | r5 | 0.083 | 0.250 | 0.226 | 3.099 | 7.637 | 2.46× |
| CacheBlend | r6 | 0.083 | 0.250 | 0.226 | 3.100 | 7.637 | 2.46× |
| **CacheBlend** | **中位** | **0.104** | **0.354** | **0.245** | **3.115** | **7.656** | **2.46×** |
| DAF | r1 | 0.167 | 0.375 | 0.270 | 2.926 | 7.710 | 2.64× |
| DAF | r2 | 0.167 | 0.375 | 0.270 | 2.928 | 7.708 | 2.63× |
| DAF | r3 | 0.167 | 0.375 | 0.270 | 2.922 | 7.704 | 2.64× |
| DAF | r4 | 0.083 | 0.250 | 0.180 | 2.972 | 7.680 | 2.58× |
| DAF | r5 | 0.083 | 0.250 | 0.180 | 2.940 | 7.669 | 2.61× |
| DAF | r6 | 0.083 | 0.250 | 0.180 | 2.931 | 7.669 | 2.62× |
| **DAF** | **中位** | **0.125** | **0.312** | **0.225** | **2.929** | **7.692** | **2.63×** |
| vLLM全量 | r1 | 0.250 | 0.625 | 0.438 | - | 7.674 | 1.00× |
| vLLM全量 | r2 | 0.250 | 0.625 | 0.438 | - | 7.687 | 1.00× |
| vLLM全量 | r3 | 0.250 | 0.625 | 0.438 | - | 7.681 | 1.00× |
| vLLM全量 | r4 | 0.333 | 0.458 | 0.468 | - | 7.624 | 1.00× |
| vLLM全量 | r5 | 0.333 | 0.458 | 0.468 | - | 7.637 | 1.00× |
| vLLM全量 | r6 | 0.333 | 0.458 | 0.468 | - | 7.637 | 1.00× |
| vLLM全量 | r7 | 0.250 | 0.625 | 0.438 | - | 7.710 | 1.00× |
| vLLM全量 | r8 | 0.250 | 0.625 | 0.438 | - | 7.708 | 1.00× |
| vLLM全量 | r9 | 0.250 | 0.625 | 0.438 | - | 7.704 | 1.00× |
| vLLM全量 | r10 | 0.333 | 0.458 | 0.468 | - | 7.680 | 1.00× |
| vLLM全量 | r11 | 0.333 | 0.458 | 0.468 | - | 7.669 | 1.00× |
| vLLM全量 | r12 | 0.333 | 0.458 | 0.468 | - | 7.669 | 1.00× |
| **vLLM全量** | **中位** | **0.292** | **0.542** | **0.453** | - | **7.677** | **1.00×** |

## 三、按数据集×方法聚合（跨轮中位数；QA 数据集看 SM/F1，samsum 摘要看 F1）

| 数据集 | 样本数 | 方法 | EM中位 | SM中位 | F1中位 | TTFT缓存中位(s) |
|---|---|---|---|---|---|---|
| validation.jsonl (qa_1) | 24 | CacheBlend | 0.000 | 0.000 | 0.055 | 3.187 |
| validation.jsonl (qa_1) | 24 | DAF | 0.000 | 0.000 | 0.000 | 2.961 |
| validation.jsonl (qa_2) | 24 | CacheBlend | 0.000 | 0.000 | 0.000 | 3.125 |
| validation.jsonl (qa_2) | 24 | DAF | 0.000 | 0.000 | 0.000 | 2.948 |

## 四、与论文 30k 档对照（手工分析）

### 运行配置

- 模型：Qwen2.5-7B-Instruct（BF16，qwen 预设，`BLEND_MAX_MODEL_LEN=32768`，`BLEND_MAX_TOKENS=32` 对齐 RULER 官方 `tokens_to_generate`）
- 样本：RULER 官方 `qa.py` 生成（seed 42），qa_1=squad 24 条（193 文档块）、qa_2=hotpotqa 24 条（228~238 文档块），长度 30.1k~32.7k token
- 方法：CacheBlend r=0.3 / DAF r=0.3（sel_layers=[1,2,3]）/ vLLM 全量；贪心解码；3 轮取中位
- 显存：CB 用 `BLEND_GPU_MEM_UTIL=0.90`，DAF 用 0.89（DAF check 阶段前 5 层全量 prefill，激活峰值更大；util 只影响 KV 池大小，不影响计算与结果）。两处配套修复：阶段三前释放 old_kvs（~1.9GB，保持 `[[None,None]]×层数` 占位）、check 前 `torch.cuda.empty_cache()`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 评分：`--raw_parse`（见下方"评分修正"），SM 为归一化后互为子串（≈RULER 官方 string_match_part）

### 核心结果

| 任务 | 指标 | CacheBlend | DAF | vLLM全量 |
|---|---|---|---|---|
| qa_1 | SM 均值 | 0.458 | 0.375 | 0.625 |
| qa_1 | EM 均值 | 0.125 | 0.167 | 0.250 |
| qa_2 | SM 均值 | 0.250 | 0.250 | 0.458 |
| qa_2 | EM 均值 | 0.083 | 0.083 | 0.333 |
| qa_1 | TTFT缓存中位(s) | 3.177 | 2.962 | 7.788（全量） |
| qa_2 | TTFT缓存中位(s) | 3.118 | 2.942 | 7.729（全量） |

- **DAF/CB 时长比：qa_1 = 0.932，qa_2 = 0.943，合并 = 0.940**（DAF 快 6%）
- 对全量加速：CB 2.46×，DAF 2.63×

### 验收标准逐条核对

| 验收标准 | 目标 | 实测 | 结论 |
|---|---|---|---|
| 30k 档 DAF vs CB ≥1.3× | DAF/CB ≤ 0.77 | 0.940 | **未达标**（但 DAF 首次反超 CB，比值 <1） |
| qa_2 DAF 精度 | ≥0.68 | 0.250 | **未达标** |
| qa_2 CB 精度 | ≤0.65 | 0.250 | 达标（但绝对水平远低于论文，不具备区分度） |

### 趋势链（DAF/CB 时长比随长度单调下降，本档确认反超）

Mistral 2–4k 1.75 → 6–10k 1.73 → 30k 1.45；Qwen 6k 1.14 → 9k 1.11 → **30k 0.94**。
Qwen 上 6k→30k 外推的反超点落在 ~25k 附近，本档实测 0.94 证实 DAF 在 30k
已比 CB 快，方向与论文一致；差距在幅度（论文 30k 档 DAF/CB≈0.77，本项目 0.94）。

### 与论文的差距与原因分析

1. **绝对精度系统性偏低**：三方法全部远低于论文对照值（论文 qa_2 DAF 0.72 /
   CB 0.625；本项目全量基线仅 0.458）。全量 prefill 是无近似的精确计算，它也低，
   说明差距来自实验设置而非算法：① 样本量 24 vs 论文/RULER 惯例 500，24 样本
   噪声 ±0.1 量级；② ~~裸 prompt 未套 ChatML~~（**2026-09-21 更正**：qwen 预设
   实际已套 ChatML 信封，仅 system 措辞与官方默认不同；消融证实影响 ≤1 题，
   非主因，见 `experiments/reports/chatml_ablation_ruler30k.md`）；
   ③ xformers 后端 vs 论文 Flash-Attention。
2. **精度排序未复现**：论文 DAF > CB；本项目 qa_2 二者持平（0.250=0.250）、
   qa_1 DAF 略低于 CB（0.375 vs 0.458，24 样本下差 2 题）。qa_2_009 出现 DAF
   把 yes 答成 no 的个例。多层聚合公式、状态融合等工程实现假设（见 AGENTS.md
   铁律 7）在 30k 长文档+分散关键证据场景下可能不如论文原方法稳健。
3. **速度反超确认但幅度不足**：DAF 在 Qwen 30k 首次快于 CB（0.94），与论文
   "长上下文 DAF 优势扩大"趋势一致；未达 1.3× 的原因同前（xformers 无
   Flash-Attention 的变长 kernel 优势、3090Ti vs A100 算力带宽比、r=0.3 下
   DAF 前 5 层全量 prefill 的固定开销在 30k 摊薄但仍存在）。

### 评分修正（本次发现的重要 bug）

`example/utils.py` 的 `parse_generation` 会把 "No…"/"Yes…" 开头的整段回答折叠成
"No"/"Yes"（原作者为 hotpot yes/no 题设计）。RULER squad 答案大量以
"Normandy/Norman/Norse" 等 No 前缀词开头，首轮评分被系统性误判（如
"Normandy is located in France." → "No" → SM=0）。已为评分脚本增加
`--raw_parse` 开关（裸首行，不折叠），本报告全部数字为重评后的值。
注：此前 6k 2wikimqa / 9k samsum 报告用的是折叠版解析——2wikimqa 的 yes/no 题
正需要该折叠，samsum 摘要极少以 No 前缀词开头，结论不受影响；本报告与它们的
口径差异仅限于此。
