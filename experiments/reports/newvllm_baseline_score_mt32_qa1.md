# 长上下文实验评分报告

- 样本目录：`experiments/inputs_qwen_ruler30k_500`（1000 个样本）
- 评分：EM（精确匹配，多答案取最大）+ SM（子串匹配）+ token-F1（多答案取最大）
- 归一化：小写/去冠词/去标点/Yes-No 规整（`example/utils.py`）
- 输入日志：`run_newvllm_mt32_qa1_c00.log`, `run_newvllm_mt32_qa1_c01.log`, `run_newvllm_mt32_qa1_c02.log`, `run_newvllm_mt32_qa1_c03.log`, `run_newvllm_mt32_qa1_c04.log`

## 一、逐样本明细（质量指标为多轮均值，TTFT 为多轮中位数）

| 样本 | 来源 | 长度 | 方法 | EM | SM | F1 | TTFT缓存(s) | TTFT全量(s) | 缓存答案 | 全量答案 |
|---|---|---|---|---|---|---|---|----|---|---|
| qa_1_000_len32420.json | validation.jsonl (qa_1) | 32420 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.429 | 8.429 | France | France |
| qa_1_001_len31728.json | validation.jsonl (qa_1) | 31728 | CacheBlend | 0.00 | 0.00 | 0.22 | 9.181 | 9.181 | The Normans in Normandy began in the 10t | The Normans in Normandy began in the 10t |
| qa_1_002_len32716.json | validation.jsonl (qa_1) | 32716 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.686 | 8.686 | Denmark, Iceland, and Norway | Denmark, Iceland, and Norway |
| qa_1_003_len30149.json | validation.jsonl (qa_1) | 30149 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.636 | 7.636 | Rollo | Rollo |
| qa_1_004_len30471.json | validation.jsonl (qa_1) | 30471 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.770 | 7.770 | 10th century | 10th century |
| qa_1_005_len32253.json | validation.jsonl (qa_1) | 32253 | CacheBlend | 0.00 | 0.00 | 0.40 | 8.451 | 8.451 | William II of Normandy | William II of Normandy |
| qa_1_006_len32195.json | validation.jsonl (qa_1) | 32195 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.364 | 9.364 | Rollo ruled the Duchy of Normandy initia | Rollo ruled the Duchy of Normandy initia |
| qa_1_007_len32600.json | validation.jsonl (qa_1) | 32600 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.495 | 8.495 | Christian | Christian |
| qa_1_008_len30678.json | validation.jsonl (qa_1) | 30678 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.391 | 8.391 | Norman: Nourmands; French: Normands; Lat | Norman: Nourmands; French: Normands; Lat |
| qa_1_009_len32681.json | validation.jsonl (qa_1) | 32681 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.591 | 8.591 | 9th century | 9th century |
| qa_1_010_len32585.json | validation.jsonl (qa_1) | 32585 | CacheBlend | 0.00 | 1.00 | 0.19 | 9.490 | 9.490 | The Duchy of Normandy was founded by tre | The Duchy of Normandy was founded by tre |
| qa_1_011_len31585.json | validation.jsonl (qa_1) | 31585 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.268 | 8.268 | King Charles III of West Francia | King Charles III of West Francia |
| qa_1_012_len31552.json | validation.jsonl (qa_1) | 31552 | CacheBlend | 0.00 | 1.00 | 0.40 | 8.225 | 8.225 | the river Epte | the river Epte |
| qa_1_013_len32635.json | validation.jsonl (qa_1) | 32635 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.616 | 8.616 | Dinwiddie | Dinwiddie |
| qa_1_014_len32483.json | validation.jsonl (qa_1) | 32483 | CacheBlend | 0.00 | 1.00 | 0.11 | 8.968 | 8.968 | The Normans replaced the Norse religion  | The Normans replaced the Norse religion  |
| qa_1_015_len30396.json | validation.jsonl (qa_1) | 30396 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.653 | 7.653 | Normandy | Normandy |
| qa_1_016_len32051.json | validation.jsonl (qa_1) | 32051 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.282 | 8.282 | dairy products | dairy products |
| qa_1_017_len30626.json | validation.jsonl (qa_1) | 30626 | CacheBlend | 0.00 | 0.00 | 0.50 | 8.842 | 8.842 | The Normans' main enemies in Italy, the  | The Normans' main enemies in Italy, the  |
| qa_1_018_len30128.json | validation.jsonl (qa_1) | 30128 | CacheBlend | 0.00 | 1.00 | 0.71 | 7.948 | 7.948 | Hervé in the 1050s. | Hervé in the 1050s. |
| qa_1_019_len32473.json | validation.jsonl (qa_1) | 32473 | CacheBlend | 0.00 | 1.00 | 0.54 | 9.286 | 9.286 | In the 1060s, Robert Crispin led the Nor | In the 1060s, Robert Crispin led the Nor |
| qa_1_020_len30589.json | validation.jsonl (qa_1) | 30589 | CacheBlend | 0.00 | 1.00 | 0.57 | 7.986 | 7.986 | the Byzantine general Alexius Komnenos | the Byzantine general Alexius Komnenos |
| qa_1_021_len32038.json | validation.jsonl (qa_1) | 32038 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.323 | 8.323 | Afranji | Afranji |
| qa_1_022_len30751.json | validation.jsonl (qa_1) | 30751 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.679 | 8.679 | A Norman named Oursel led a force of "Fr | A Norman named Oursel led a force of "Fr |
| qa_1_023_len32382.json | validation.jsonl (qa_1) | 32382 | CacheBlend | 0.00 | 1.00 | 0.12 | 9.488 | 9.488 | The Normans joined Turkish forces to aid | The Normans joined Turkish forces to aid |
| qa_1_024_len31374.json | validation.jsonl (qa_1) | 31374 | CacheBlend | 0.00 | 1.00 | 0.61 | 8.610 | 8.610 | The Raoulii family was descended from an | The Raoulii family was descended from an |
| qa_1_025_len30462.json | validation.jsonl (qa_1) | 30462 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.744 | 7.744 | Robert Guiscard | Robert Guiscard |
| qa_1_026_len32340.json | validation.jsonl (qa_1) | 32340 | CacheBlend | 0.00 | 0.00 | 0.38 | 8.973 | 8.973 | Dyrrachium fell to the Normans in 1081. | Dyrrachium fell to the Normans in 1081. |
| qa_1_027_len30910.json | validation.jsonl (qa_1) | 30910 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.989 | 7.989 | 30,000 | 30,000 |
| qa_1_028_len32017.json | validation.jsonl (qa_1) | 32017 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.355 | 8.355 | Dyrrachium | Dyrrachium |
| qa_1_029_len31291.json | validation.jsonl (qa_1) | 31291 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.034 | 8.034 | Bohemond | Bohemond |
| qa_1_030_len31310.json | validation.jsonl (qa_1) | 31310 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.164 | 8.164 | the low Seine valley | the low Seine valley |
| qa_1_031_len32496.json | validation.jsonl (qa_1) | 32496 | CacheBlend | 0.00 | 1.00 | 0.40 | 9.162 | 9.162 | The Normans attacked Dyrrachium in 1081  | The Normans attacked Dyrrachium in 1081  |
| qa_1_032_len32245.json | validation.jsonl (qa_1) | 32245 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.438 | 8.438 | Afranji | Afranji |
| qa_1_033_len30549.json | validation.jsonl (qa_1) | 30549 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.096 | 8.096 | Dyrrachium was located in modern-day Alb | Dyrrachium was located in modern-day Alb |
| qa_1_034_len31855.json | validation.jsonl (qa_1) | 31855 | CacheBlend | 0.00 | 0.00 | 0.17 | 8.386 | 8.386 | Duke Richard II of Normandy | Duke Richard II of Normandy |
| qa_1_035_len32730.json | validation.jsonl (qa_1) | 32730 | CacheBlend | 0.00 | 1.00 | 0.77 | 8.717 | 8.717 | Duke Richard II of Normandy | Duke Richard II of Normandy |
| qa_1_036_len30672.json | validation.jsonl (qa_1) | 30672 | CacheBlend | 0.00 | 1.00 | 0.40 | 7.809 | 7.809 | to Normandy | to Normandy |
| qa_1_037_len32088.json | validation.jsonl (qa_1) | 32088 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.413 | 8.413 | Sweyn Forkbeard | Sweyn Forkbeard |
| qa_1_038_len31185.json | validation.jsonl (qa_1) | 31185 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.044 | 8.044 | Harthacnut | Harthacnut |
| qa_1_039_len30640.json | validation.jsonl (qa_1) | 30640 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.888 | 7.888 | 1041 | 1041 |
| qa_1_040_len31793.json | validation.jsonl (qa_1) | 31793 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.294 | 8.294 | Robert of Jumièges | Robert of Jumièges |
| qa_1_041_len32112.json | validation.jsonl (qa_1) | 32112 | CacheBlend | 0.00 | 1.00 | 0.57 | 8.555 | 8.555 | Harold II died at the Battle of Hastings | Harold II died at the Battle of Hastings |
| qa_1_042_len28768.json | validation.jsonl (qa_1) | 28768 | CacheBlend | 0.00 | 1.00 | 0.59 | 7.449 | 7.449 | Duke William II of Normandy killed Harol | Duke William II of Normandy killed Harol |
| qa_1_043_len29775.json | validation.jsonl (qa_1) | 29775 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.567 | 7.567 | 1066 | 1066 |
| qa_1_044_len32613.json | validation.jsonl (qa_1) | 32613 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.507 | 8.507 | Frankish | Frankish |
| qa_1_045_len32597.json | validation.jsonl (qa_1) | 32597 | CacheBlend | 0.00 | 0.00 | 0.07 | 9.499 | 9.499 | The Anglo-Norman language was eventually | The Anglo-Norman language was eventually |
| qa_1_046_len31139.json | validation.jsonl (qa_1) | 31139 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.018 | 8.018 | 1169 | 1169 |
| qa_1_047_len32728.json | validation.jsonl (qa_1) | 32728 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.542 | 8.542 | Ireland | Ireland |
| qa_1_048_len32119.json | validation.jsonl (qa_1) | 32119 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.804 | 8.804 | Catholicism and the Gallo-Romance langua | Catholicism and the Gallo-Romance langua |
| qa_1_049_len31169.json | validation.jsonl (qa_1) | 31169 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.029 | 8.029 | Edgar Atheling | Edgar Atheling |
| qa_1_050_len32145.json | validation.jsonl (qa_1) | 32145 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.406 | 8.406 | Edgar Atheling | Edgar Atheling |
| qa_1_051_len32137.json | validation.jsonl (qa_1) | 32137 | CacheBlend | 0.00 | 0.00 | 0.50 | 8.404 | 8.404 | 1066 | 1066 |
| qa_1_052_len29728.json | validation.jsonl (qa_1) | 29728 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.548 | 7.548 | Duncan | Duncan |
| qa_1_053_len30282.json | validation.jsonl (qa_1) | 30282 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.794 | 7.794 | Sybilla of Normandy | Sybilla of Normandy |
| qa_1_054_len30042.json | validation.jsonl (qa_1) | 30042 | CacheBlend | 0.00 | 1.00 | 0.27 | 8.124 | 8.124 | Norman culture's arrival in Scotland is  | Norman culture's arrival in Scotland is  |
| qa_1_055_len30991.json | validation.jsonl (qa_1) | 30991 | CacheBlend | 0.00 | 1.00 | 0.22 | 8.180 | 8.180 | Ralph earl of Hereford. | Ralph earl of Hereford. |
| qa_1_056_len31411.json | validation.jsonl (qa_1) | 31411 | CacheBlend | 0.00 | 1.00 | 0.14 | 8.441 | 8.441 | Ralph was in charge of being at war with | Ralph was in charge of being at war with |
| qa_1_057_len29615.json | validation.jsonl (qa_1) | 29615 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.827 | 7.827 | King Ethelred II of England made Ralph e | King Ethelred II of England made Ralph e |
| qa_1_058_len32630.json | validation.jsonl (qa_1) | 32630 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.450 | 8.450 | Wales | Wales |
| qa_1_059_len32398.json | validation.jsonl (qa_1) | 32398 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.458 | 8.458 | 1018 | 1018 |
| qa_1_060_len32336.json | validation.jsonl (qa_1) | 32336 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.482 | 8.482 | William of Montreuil | William of Montreuil |
| qa_1_061_len30969.json | validation.jsonl (qa_1) | 30969 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.950 | 7.950 | 1097 | 1097 |
| qa_1_062_len29765.json | validation.jsonl (qa_1) | 29765 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.515 | 7.515 | Tancred | Tancred |
| qa_1_063_len31774.json | validation.jsonl (qa_1) | 31774 | CacheBlend | 0.00 | 1.00 | 0.08 | 9.210 | 9.210 | Tancred played a role in the conquest of | Tancred played a role in the conquest of |
| qa_1_064_len32621.json | validation.jsonl (qa_1) | 32621 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.433 | 9.433 | The Normans under the command of Bohemon | The Normans under the command of Bohemon |
| qa_1_065_len31954.json | validation.jsonl (qa_1) | 31954 | CacheBlend | 0.00 | 1.00 | 0.13 | 8.821 | 8.821 | A storm dispersed the fleet, making it d | A storm dispersed the fleet, making it d |
| qa_1_066_len32699.json | validation.jsonl (qa_1) | 32699 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.585 | 8.585 | Berengaria | Berengaria |
| qa_1_067_len32231.json | validation.jsonl (qa_1) | 32231 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.441 | 8.441 | 1191 | 1191 |
| qa_1_068_len31186.json | validation.jsonl (qa_1) | 31186 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.131 | 8.131 | Isaac Komnenos | Isaac Komnenos |
| qa_1_069_len31696.json | validation.jsonl (qa_1) | 31696 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.350 | 8.350 | Conrad of Montferrat | Conrad of Montferrat |
| qa_1_070_len31505.json | validation.jsonl (qa_1) | 31505 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.075 | 8.075 | silver | silver |
| qa_1_071_len31228.json | validation.jsonl (qa_1) | 31228 | CacheBlend | 0.00 | 0.00 | 0.11 | 8.281 | 8.281 | Richard de Camville and Robert de Thornh | Richard de Camville and Robert de Thornh |
| qa_1_072_len32173.json | validation.jsonl (qa_1) | 32173 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.281 | 8.281 | Africa | Africa |
| qa_1_073_len31989.json | validation.jsonl (qa_1) | 31989 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.283 | 8.283 | Bethencourt | Bethencourt |
| qa_1_074_len30148.json | validation.jsonl (qa_1) | 30148 | CacheBlend | 0.00 | 1.00 | 0.37 | 8.605 | 8.605 | Jean's nephew Maciot de Bethencourt boug | Jean's nephew Maciot de Bethencourt boug |
| qa_1_075_len32155.json | validation.jsonl (qa_1) | 32155 | CacheBlend | 0.00 | 1.00 | 0.41 | 9.309 | 9.309 | Maciot de Bethencourt sold the rights to | Maciot de Bethencourt sold the rights to |
| qa_1_076_len31246.json | validation.jsonl (qa_1) | 31246 | CacheBlend | 0.00 | 1.00 | 0.17 | 8.364 | 8.364 | Jersey and Guernsey are in the Channel I | Jersey and Guernsey are in the Channel I |
| qa_1_077_len32170.json | validation.jsonl (qa_1) | 32170 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.319 | 8.319 | 2 | 2 |
| qa_1_078_len31327.json | validation.jsonl (qa_1) | 31327 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.779 | 8.779 | The Norman architecture idiom is charact | The Norman architecture idiom is charact |
| qa_1_079_len30646.json | validation.jsonl (qa_1) | 30646 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.229 | 8.229 | Norman architecture has rounded arches,  | Norman architecture has rounded arches,  |
| qa_1_080_len30668.json | validation.jsonl (qa_1) | 30668 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.851 | 7.851 | Anglo-Saxon | Anglo-Saxon |
| qa_1_081_len32397.json | validation.jsonl (qa_1) | 32397 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.518 | 8.518 | Anglo-Saxon | Anglo-Saxon |
| qa_1_082_len29758.json | validation.jsonl (qa_1) | 29758 | CacheBlend | 0.00 | 0.00 | 0.10 | 8.000 | 8.000 | The Duchy of Normandy and southern Italy | The Duchy of Normandy and southern Italy |
| qa_1_083_len30969.json | validation.jsonl (qa_1) | 30969 | CacheBlend | 0.00 | 1.00 | 0.71 | 8.256 | 8.256 | The church reform began in the early 11t | The church reform began in the early 11t |
| qa_1_084_len30551.json | validation.jsonl (qa_1) | 30551 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.021 | 8.021 | The Normans used the church to unify the | The Normans used the church to unify the |
| qa_1_085_len32000.json | validation.jsonl (qa_1) | 32000 | CacheBlend | 0.00 | 1.00 | 0.56 | 8.710 | 8.710 | The French Wars of Religion occurred in  | The French Wars of Religion occurred in  |
| qa_1_086_len31039.json | validation.jsonl (qa_1) | 31039 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.934 | 7.934 | embroidery | embroidery |
| qa_1_087_len28340.json | validation.jsonl (qa_1) | 28340 | CacheBlend | 0.00 | 1.00 | 0.42 | 7.554 | 7.554 | The most well-known piece of Norman art  | The most well-known piece of Norman art  |
| qa_1_088_len31102.json | validation.jsonl (qa_1) | 31102 | CacheBlend | 0.00 | 1.00 | 0.12 | 8.537 | 8.537 | Odo, the Bishop of Bayeux and first Earl | Odo, the Bishop of Bayeux and first Earl |
| qa_1_089_len31470.json | validation.jsonl (qa_1) | 31470 | CacheBlend | 0.00 | 1.00 | 0.06 | 9.034 | 9.034 | Norman artwork survives plentifully in f | Norman artwork survives plentifully in f |
| qa_1_090_len30235.json | validation.jsonl (qa_1) | 30235 | CacheBlend | 0.00 | 1.00 | 0.86 | 7.712 | 7.712 | 11th century | 11th century |
| qa_1_091_len31074.json | validation.jsonl (qa_1) | 31074 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.186 | 8.186 | William of Volpiano and John of Ravenna | William of Volpiano and John of Ravenna |
| qa_1_092_len31265.json | validation.jsonl (qa_1) | 31265 | CacheBlend | 0.00 | 1.00 | 0.33 | 8.231 | 8.231 | The monks fled to southern Italy. | The monks fled to southern Italy. |
| qa_1_093_len32404.json | validation.jsonl (qa_1) | 32404 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.553 | 8.553 | Sant'Eufemia | Sant'Eufemia |
| qa_1_094_len30691.json | validation.jsonl (qa_1) | 30691 | CacheBlend | 0.00 | 1.00 | 0.57 | 8.058 | 8.058 | Robert Guiscard patronized the monks in  | Robert Guiscard patronized the monks in  |
| qa_1_095_len30985.json | validation.jsonl (qa_1) | 30985 | CacheBlend | 0.00 | 0.00 | 0.09 | 8.688 | 8.688 | The Saint-Evroul monks were known for de | The Saint-Evroul monks were known for de |
| qa_1_096_len31883.json | validation.jsonl (qa_1) | 31883 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.327 | 8.327 | Computational complexity theory | Computational complexity theory |
| qa_1_097_len31198.json | validation.jsonl (qa_1) | 31198 | CacheBlend | 0.00 | 1.00 | 0.00 | 7.994 | 7.994 | difficulty | difficulty |
| qa_1_098_len32639.json | validation.jsonl (qa_1) | 32639 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.554 | 8.554 | computational problem | computational problem |
| qa_1_099_len30722.json | validation.jsonl (qa_1) | 30722 | CacheBlend | 0.00 | 0.00 | 0.10 | 8.545 | 8.545 | The measure of a computational problem t | The measure of a computational problem t |
| qa_1_100_len30867.json | validation.jsonl (qa_1) | 30867 | CacheBlend | 0.00 | 0.00 | 0.07 | 8.738 | 8.738 | The amount of resources required to solv | The amount of resources required to solv |
| qa_1_101_len32721.json | validation.jsonl (qa_1) | 32721 | CacheBlend | 0.00 | 0.00 | 0.67 | 8.526 | 8.526 | time and space | time and space |
| qa_1_102_len32234.json | validation.jsonl (qa_1) | 32234 | CacheBlend | 0.00 | 0.00 | 0.24 | 9.318 | 9.318 | The unit measured to determine circuit c | The unit measured to determine circuit c |
| qa_1_103_len31221.json | validation.jsonl (qa_1) | 31221 | CacheBlend | 0.00 | 0.00 | 0.12 | 8.978 | 8.978 | Defining the complexity of problems help | Defining the complexity of problems help |
| qa_1_104_len29976.json | validation.jsonl (qa_1) | 29976 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.786 | 7.786 | analysis of algorithms and computability | analysis of algorithms and computability |
| qa_1_105_len32495.json | validation.jsonl (qa_1) | 32495 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.519 | 8.519 | Analysis of algorithms | Analysis of algorithms |
| qa_1_106_len31241.json | validation.jsonl (qa_1) | 31241 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.101 | 8.101 | computational complexity theory | computational complexity theory |
| qa_1_107_len30414.json | validation.jsonl (qa_1) | 30414 | CacheBlend | 0.00 | 0.00 | 0.57 | 7.824 | 7.824 | computational complexity theory | computational complexity theory |
| qa_1_108_len29609.json | validation.jsonl (qa_1) | 29609 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.500 | 7.500 | problem instance | problem instance |
| qa_1_109_len30615.json | validation.jsonl (qa_1) | 30615 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.783 | 7.783 | problem | problem |
| qa_1_110_len32263.json | validation.jsonl (qa_1) | 32263 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.354 | 8.354 | concrete | concrete |
| qa_1_111_len32402.json | validation.jsonl (qa_1) | 32402 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.414 | 8.414 | resource | resource |
| qa_1_112_len30238.json | validation.jsonl (qa_1) | 30238 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.658 | 7.658 | solution | solution |
| qa_1_113_len32383.json | validation.jsonl (qa_1) | 32383 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.696 | 8.696 | at most 2000 kilometers | at most 2000 kilometers |
| qa_1_114_len31725.json | validation.jsonl (qa_1) | 31725 | CacheBlend | 0.00 | 0.00 | 0.35 | 9.094 | 9.094 | Is there a route of at most 2000 kilomet | Is there a route of at most 2000 kilomet |
| qa_1_115_len31106.json | validation.jsonl (qa_1) | 31106 | CacheBlend | 0.00 | 1.00 | 0.24 | 8.881 | 8.881 | Computational complexity theory most spe | Computational complexity theory most spe |
| qa_1_116_len31489.json | validation.jsonl (qa_1) | 31489 | CacheBlend | 0.00 | 1.00 | 0.15 | 8.681 | 8.681 | In a computational problem, an instance  | In a computational problem, an instance  |
| qa_1_117_len32334.json | validation.jsonl (qa_1) | 32334 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.491 | 8.491 | binary alphabet | binary alphabet |
| qa_1_118_len30653.json | validation.jsonl (qa_1) | 30653 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.880 | 7.880 | input string | input string |
| qa_1_119_len31885.json | validation.jsonl (qa_1) | 31885 | CacheBlend | 0.00 | 1.00 | 0.20 | 8.525 | 8.525 | integers are commonly expressed in binar | integers are commonly expressed in binar |
| qa_1_120_len30497.json | validation.jsonl (qa_1) | 30497 | CacheBlend | 0.00 | 1.00 | 0.53 | 8.052 | 8.052 | graphs can be encoded directly via their | graphs can be encoded directly via their |
| qa_1_121_len30432.json | validation.jsonl (qa_1) | 30432 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.773 | 7.773 | Decision problems | Decision problems |
| qa_1_122_len29351.json | validation.jsonl (qa_1) | 29351 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.471 | 7.471 | yes or no | yes or no |
| qa_1_123_len31399.json | validation.jsonl (qa_1) | 31399 | CacheBlend | 0.00 | 0.00 | 0.29 | 8.132 | 8.132 | yes or no | yes or no |
| qa_1_124_len31034.json | validation.jsonl (qa_1) | 31034 | CacheBlend | 0.00 | 1.00 | 0.50 | 8.017 | 8.017 | yes or no | yes or no |
| qa_1_125_len32527.json | validation.jsonl (qa_1) | 32527 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.578 | 8.578 | the algorithm returns yes | the algorithm returns yes |
| qa_1_126_len31350.json | validation.jsonl (qa_1) | 31350 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.081 | 8.081 | connected graphs | connected graphs |
| qa_1_127_len31881.json | validation.jsonl (qa_1) | 31881 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.416 | 8.416 | The set of all connected graphs | The set of all connected graphs |
| qa_1_128_len31494.json | validation.jsonl (qa_1) | 31494 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.319 | 8.319 | the encoding of different objects into b | the encoding of different objects into b |
| qa_1_129_len31707.json | validation.jsonl (qa_1) | 31707 | CacheBlend | 0.00 | 1.00 | 0.22 | 9.210 | 9.210 | a computational problem where a single o | a computational problem where a single o |
| qa_1_130_len30722.json | validation.jsonl (qa_1) | 30722 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.028 | 8.028 | more complex than just yes or no | more complex than just yes or no |
| qa_1_131_len30888.json | validation.jsonl (qa_1) | 30888 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.947 | 7.947 | function problem | function problem |
| qa_1_132_len31804.json | validation.jsonl (qa_1) | 31804 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.347 | 8.347 | The integer factorization problem | The integer factorization problem |
| qa_1_133_len30956.json | validation.jsonl (qa_1) | 30956 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.923 | 7.923 | complex | complex |
| qa_1_134_len29660.json | validation.jsonl (qa_1) | 29660 | CacheBlend | 0.00 | 1.00 | 0.31 | 7.831 | 7.831 | Function problems can typically be resta | Function problems can typically be resta |
| qa_1_135_len31906.json | validation.jsonl (qa_1) | 31906 | CacheBlend | 0.00 | 0.00 | 0.67 | 8.329 | 8.329 | set of integers | set of integers |
| qa_1_136_len30284.json | validation.jsonl (qa_1) | 30284 | CacheBlend | 0.00 | 1.00 | 0.67 | 7.710 | 7.710 | time complexity | time complexity |
| qa_1_137_len30021.json | validation.jsonl (qa_1) | 30021 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.695 | 7.695 | the size of the instance | the size of the instance |
| qa_1_138_len31003.json | validation.jsonl (qa_1) | 31003 | CacheBlend | 0.00 | 1.00 | 0.37 | 8.963 | 8.963 | The time needed to obtain the solution t | The time needed to obtain the solution t |
| qa_1_139_len29185.json | validation.jsonl (qa_1) | 29185 | CacheBlend | 0.00 | 1.00 | 0.00 | 7.602 | 7.602 | The size of the input is measured in bit | The size of the input is measured in bit |
| qa_1_140_len32724.json | validation.jsonl (qa_1) | 32724 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.582 | 8.582 | time and space | time and space |
| qa_1_141_len30850.json | validation.jsonl (qa_1) | 30850 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.023 | 8.023 | Cobham's thesis | Cobham's thesis |
| qa_1_142_len30924.json | validation.jsonl (qa_1) | 30924 | CacheBlend | 0.00 | 1.00 | 0.07 | 8.929 | 8.929 | Since the time taken on different inputs | Since the time taken on different inputs |
| qa_1_143_len30440.json | validation.jsonl (qa_1) | 30440 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.874 | 7.874 | worst-case time complexity | worst-case time complexity |
| qa_1_144_len31849.json | validation.jsonl (qa_1) | 31849 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.297 | 8.297 | T(n) | T(n) |
| qa_1_145_len31762.json | validation.jsonl (qa_1) | 31762 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.288 | 8.288 | polynomial time algorithm | polynomial time algorithm |
| qa_1_146_len31153.json | validation.jsonl (qa_1) | 31153 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.056 | 8.056 | Turing machine | Turing machine |
| qa_1_147_len30808.json | validation.jsonl (qa_1) | 30808 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.054 | 8.054 | anything that can be computed by an algo | anything that can be computed by an algo |
| qa_1_148_len30919.json | validation.jsonl (qa_1) | 30919 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.004 | 8.004 | Turing machine | Turing machine |
| qa_1_149_len29541.json | validation.jsonl (qa_1) | 29541 | CacheBlend | 0.00 | 1.00 | 0.00 | 7.948 | 7.948 | A Turing machine manipulates symbols con | A Turing machine manipulates symbols con |
| qa_1_150_len31416.json | validation.jsonl (qa_1) | 31416 | CacheBlend | 0.00 | 1.00 | 0.33 | 8.616 | 8.616 | The most basic iteration of a Turing mac | The most basic iteration of a Turing mac |
| qa_1_151_len30560.json | validation.jsonl (qa_1) | 30560 | CacheBlend | 0.00 | 1.00 | 0.62 | 7.951 | 7.951 | a fixed set of rules | a fixed set of rules |
| qa_1_152_len31940.json | validation.jsonl (qa_1) | 31940 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.430 | 8.430 | Probabilistic Turing machine | Probabilistic Turing machine |
| qa_1_153_len31478.json | validation.jsonl (qa_1) | 31478 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.297 | 8.297 | Non-deterministic Turing machine | Non-deterministic Turing machine |
| qa_1_154_len31973.json | validation.jsonl (qa_1) | 31973 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.358 | 8.358 | Randomized algorithms | Randomized algorithms |
| qa_1_155_len32311.json | validation.jsonl (qa_1) | 32311 | CacheBlend | 0.00 | 1.00 | 0.15 | 8.842 | 8.842 | Turing machines are commonly employed to | Turing machines are commonly employed to |
| qa_1_156_len30908.json | validation.jsonl (qa_1) | 30908 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.073 | 9.073 | The ability to make probabilistic decisi | The ability to make probabilistic decisi |
| qa_1_157_len30931.json | validation.jsonl (qa_1) | 30931 | CacheBlend | 0.00 | 0.00 | 0.58 | 8.295 | 8.295 | deterministic Turing machine and probabi | deterministic Turing machine and probabi |
| qa_1_158_len30549.json | validation.jsonl (qa_1) | 30549 | CacheBlend | 0.00 | 0.00 | 0.09 | 8.771 | 8.771 | An example of a machine model that devia | An example of a machine model that devia |
| qa_1_159_len31675.json | validation.jsonl (qa_1) | 31675 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.267 | 8.267 | computational power | computational power |
| qa_1_160_len30636.json | validation.jsonl (qa_1) | 30636 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.941 | 7.941 | time and memory consumption | time and memory consumption |
| qa_1_161_len31439.json | validation.jsonl (qa_1) | 31439 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.514 | 8.514 | They all can be converted to another wit | They all can be converted to another wit |
| qa_1_162_len31743.json | validation.jsonl (qa_1) | 31743 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.361 | 8.361 | Non-deterministic Turing machine | Non-deterministic Turing machine |
| qa_1_163_len30939.json | validation.jsonl (qa_1) | 30939 | CacheBlend | 0.00 | 1.00 | 0.38 | 8.388 | 8.388 | more unusual resources often affect or f | more unusual resources often affect or f |
| qa_1_164_len31356.json | validation.jsonl (qa_1) | 31356 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.247 | 8.247 | a non-deterministic time facet | a non-deterministic time facet |
| qa_1_165_len28460.json | validation.jsonl (qa_1) | 28460 | CacheBlend | 0.00 | 0.00 | 0.67 | 7.199 | 7.199 | non-determinism | non-determinism |
| qa_1_166_len31388.json | validation.jsonl (qa_1) | 31388 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.335 | 8.335 | the total number of state transitions, o | the total number of state transitions, o |
| qa_1_167_len30875.json | validation.jsonl (qa_1) | 30875 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.945 | 7.945 | resource consumption | resource consumption |
| qa_1_168_len30998.json | validation.jsonl (qa_1) | 30998 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.081 | 8.081 | DTIME(f(n)) | DTIME(f(n)) |
| qa_1_169_len31532.json | validation.jsonl (qa_1) | 31532 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.134 | 8.134 | time | time |
| qa_1_170_len30390.json | validation.jsonl (qa_1) | 30390 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.794 | 7.794 | complexity resources | complexity resources |
| qa_1_171_len31008.json | validation.jsonl (qa_1) | 31008 | CacheBlend | 0.00 | 1.00 | 0.15 | 8.323 | 8.323 | A complexity resource can also be descri | A complexity resource can also be descri |
| qa_1_172_len30865.json | validation.jsonl (qa_1) | 30865 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.087 | 8.087 | The Blum complexity axioms. | The Blum complexity axioms. |
| qa_1_173_len31870.json | validation.jsonl (qa_1) | 31870 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.293 | 8.293 | a complexity measure | a complexity measure |
| qa_1_174_len31211.json | validation.jsonl (qa_1) | 31211 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.995 | 7.995 | function | function |
| qa_1_175_len30506.json | validation.jsonl (qa_1) | 30506 | CacheBlend | 0.00 | 0.00 | 0.45 | 8.416 | 8.416 | The three primary expressions used to re | The three primary expressions used to re |
| qa_1_176_len31439.json | validation.jsonl (qa_1) | 31439 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.126 | 8.126 | time and space | time and space |
| qa_1_177_len31057.json | validation.jsonl (qa_1) | 31057 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.939 | 7.939 | time | time |
| qa_1_178_len32448.json | validation.jsonl (qa_1) | 32448 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.439 | 8.439 | time | time |
| qa_1_179_len29277.json | validation.jsonl (qa_1) | 29277 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.418 | 7.418 | quicksort | quicksort |
| qa_1_180_len30144.json | validation.jsonl (qa_1) | 30144 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.700 | 7.700 | worst-case | worst-case |
| qa_1_181_len31234.json | validation.jsonl (qa_1) | 31234 | CacheBlend | 0.00 | 1.00 | 0.40 | 8.267 | 8.267 | T(n) = O(n^2) | T(n) = O(n^2) |
| qa_1_182_len30516.json | validation.jsonl (qa_1) | 30516 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.998 | 7.998 | the most efficient algorithm solving a g | the most efficient algorithm solving a g |
| qa_1_183_len30110.json | validation.jsonl (qa_1) | 30110 | CacheBlend | 0.00 | 1.00 | 0.50 | 7.796 | 7.796 | the field of analysis of algorithms | the field of analysis of algorithms |
| qa_1_184_len31384.json | validation.jsonl (qa_1) | 31384 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.092 | 8.092 | lower bounds | lower bounds |
| qa_1_185_len31464.json | validation.jsonl (qa_1) | 31464 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.173 | 8.173 | a specific time complexity | a specific time complexity |
| qa_1_186_len32249.json | validation.jsonl (qa_1) | 32249 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.436 | 8.436 | polynomial time algorithm | polynomial time algorithm |
| qa_1_187_len29855.json | validation.jsonl (qa_1) | 29855 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.607 | 7.607 | big O notation | big O notation |
| qa_1_188_len31697.json | validation.jsonl (qa_1) | 31697 | CacheBlend | 0.00 | 1.00 | 0.57 | 8.480 | 8.480 | big O notation hides constant factors an | big O notation hides constant factors an |
| qa_1_189_len30490.json | validation.jsonl (qa_1) | 30490 | CacheBlend | 0.00 | 0.00 | 0.67 | 7.995 | 7.995 | T(n) = O(n²) | T(n) = O(n²) |
| qa_1_190_len31632.json | validation.jsonl (qa_1) | 31632 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.210 | 8.210 | computational model | computational model |
| qa_1_191_len30933.json | validation.jsonl (qa_1) | 30933 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.993 | 7.993 | Some complexity classes | Some complexity classes |
| qa_1_192_len31924.json | validation.jsonl (qa_1) | 31924 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.591 | 8.591 | Complexity classes are generally classif | Complexity classes are generally classif |
| qa_1_193_len30679.json | validation.jsonl (qa_1) | 30679 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.809 | 7.809 | resources | resources |
| qa_1_194_len30507.json | validation.jsonl (qa_1) | 30507 | CacheBlend | 0.00 | 0.00 | 0.57 | 7.861 | 7.861 | concrete machine model | concrete machine model |
| qa_1_195_len32452.json | validation.jsonl (qa_1) | 32452 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.509 | 8.509 | a time complexity | a time complexity |
| qa_1_196_len31923.json | validation.jsonl (qa_1) | 31923 | CacheBlend | 0.00 | 0.00 | 0.33 | 8.385 | 8.385 | deterministic Turing machine | deterministic Turing machine |
| qa_1_197_len31358.json | validation.jsonl (qa_1) | 31358 | CacheBlend | 0.00 | 0.00 | 0.67 | 8.168 | 8.168 | Cobham's thesis | Cobham's thesis |
| qa_1_198_len30981.json | validation.jsonl (qa_1) | 30981 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.922 | 7.922 | P | P |
| qa_1_199_len31501.json | validation.jsonl (qa_1) | 31501 | CacheBlend | 0.00 | 0.00 | 0.67 | 8.141 | 8.141 | time and space | time and space |
| qa_1_200_len30958.json | validation.jsonl (qa_1) | 30958 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.014 | 8.014 | computational model | computational model |
| qa_1_201_len29409.json | validation.jsonl (qa_1) | 29409 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.442 | 7.442 | complexity classes | complexity classes |
| qa_1_202_len29532.json | validation.jsonl (qa_1) | 29532 | CacheBlend | 0.00 | 0.00 | 0.89 | 7.595 | 7.595 | BPP, ZPP, RP | BPP, ZPP, RP |
| qa_1_203_len28703.json | validation.jsonl (qa_1) | 28703 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.102 | 7.102 | Boolean circuits | Boolean circuits |
| qa_1_204_len31871.json | validation.jsonl (qa_1) | 31871 | CacheBlend | 0.00 | 1.00 | 0.57 | 8.325 | 8.325 | quantum Turing machines | quantum Turing machines |
| qa_1_205_len32732.json | validation.jsonl (qa_1) | 32732 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.542 | 8.542 | #P | #P |
| qa_1_206_len31259.json | validation.jsonl (qa_1) | 31259 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.400 | 8.400 | IP and AM are most commonly defined usin | IP and AM are most commonly defined usin |
| qa_1_207_len31512.json | validation.jsonl (qa_1) | 31512 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.197 | 8.197 | time and space | time and space |
| qa_1_208_len31177.json | validation.jsonl (qa_1) | 31177 | CacheBlend | 0.00 | 0.00 | 0.29 | 8.463 | 8.463 | DTIME(n) can be found in the context of  | DTIME(n) can be found in the context of  |
| qa_1_209_len30901.json | validation.jsonl (qa_1) | 30901 | CacheBlend | 0.00 | 1.00 | 0.58 | 8.536 | 8.536 | The time and space hierarchy theorems ar | The time and space hierarchy theorems ar |
| qa_1_210_len31797.json | validation.jsonl (qa_1) | 31797 | CacheBlend | 0.00 | 0.00 | 0.29 | 8.267 | 8.267 | complexity classes | complexity classes |
| qa_1_211_len32719.json | validation.jsonl (qa_1) | 32719 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.093 | 9.093 | The effort involves making upper and low | The effort involves making upper and low |
| qa_1_212_len32063.json | validation.jsonl (qa_1) | 32063 | CacheBlend | 0.00 | 1.00 | 0.61 | 8.900 | 8.900 | The time and space hierarchy theorems fo | The time and space hierarchy theorems fo |
| qa_1_213_len30545.json | validation.jsonl (qa_1) | 30545 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.854 | 7.854 | a more powerful machine | a more powerful machine |
| qa_1_214_len31652.json | validation.jsonl (qa_1) | 31652 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.153 | 8.153 | space | space |
| qa_1_215_len30827.json | validation.jsonl (qa_1) | 30827 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.924 | 7.924 | reduction | reduction |
| qa_1_216_len30674.json | validation.jsonl (qa_1) | 30674 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.001 | 8.001 | one problem into another problem. | one problem into another problem. |
| qa_1_217_len30986.json | validation.jsonl (qa_1) | 30986 | CacheBlend | 0.00 | 0.00 | 0.55 | 8.183 | 8.183 | X is reducible to Y | X is reducible to Y |
| qa_1_218_len31722.json | validation.jsonl (qa_1) | 31722 | CacheBlend | 0.00 | 0.00 | 0.91 | 8.355 | 8.355 | Cook reductions and Karp reductions | Cook reductions and Karp reductions |
| qa_1_219_len31630.json | validation.jsonl (qa_1) | 31630 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.211 | 8.211 | reductions | reductions |
| qa_1_220_len31566.json | validation.jsonl (qa_1) | 31566 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.235 | 8.235 | polynomial-time reduction | polynomial-time reduction |
| qa_1_221_len32243.json | validation.jsonl (qa_1) | 32243 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.438 | 8.438 | multiplying two integers | multiplying two integers |
| qa_1_222_len30846.json | validation.jsonl (qa_1) | 30846 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.918 | 7.918 | polynomial | polynomial |
| qa_1_223_len31405.json | validation.jsonl (qa_1) | 31405 | CacheBlend | 0.00 | 0.00 | 0.00 | 9.082 | 9.082 | The algorithm would need to remain consi | The algorithm would need to remain consi |
| qa_1_224_len31646.json | validation.jsonl (qa_1) | 31646 | CacheBlend | 0.00 | 0.00 | 0.33 | 8.272 | 8.272 | multiplying two integers | multiplying two integers |
| qa_1_225_len31833.json | validation.jsonl (qa_1) | 31833 | CacheBlend | 0.00 | 0.00 | 0.47 | 8.740 | 8.740 | the type of reduction used and the resou | the type of reduction used and the resou |
| qa_1_226_len31015.json | validation.jsonl (qa_1) | 31015 | CacheBlend | 0.00 | 0.00 | 0.38 | 8.957 | 8.957 | If a problem X is hard for a class of pr | If a problem X is hard for a class of pr |
| qa_1_227_len31324.json | validation.jsonl (qa_1) | 31324 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.173 | 8.173 | solve any problem in C | solve any problem in C |
| qa_1_228_len30643.json | validation.jsonl (qa_1) | 30643 | CacheBlend | 0.00 | 1.00 | 0.21 | 8.384 | 8.384 | A problem set that is hard for the expre | A problem set that is hard for the expre |
| qa_1_229_len31179.json | validation.jsonl (qa_1) | 31179 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.020 | 8.020 | NP-complete | NP-complete |
| qa_1_230_len30565.json | validation.jsonl (qa_1) | 30565 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.771 | 7.771 | P | P |
| qa_1_231_len31371.json | validation.jsonl (qa_1) | 31371 | CacheBlend | 0.00 | 0.00 | 0.12 | 9.040 | 9.040 | If P = NP is unsolved and a known NP-com | If P = NP is unsolved and a known NP-com |
| qa_1_232_len32512.json | validation.jsonl (qa_1) | 32512 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.521 | 8.521 | P = NP | P = NP |
| qa_1_233_len29898.json | validation.jsonl (qa_1) | 29898 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.541 | 7.541 | P | P |
| qa_1_234_len30965.json | validation.jsonl (qa_1) | 30965 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.129 | 8.129 | Cobham–Edmonds thesis | Cobham–Edmonds thesis |
| qa_1_235_len31067.json | validation.jsonl (qa_1) | 31067 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.953 | 7.953 | NP | NP |
| qa_1_236_len32345.json | validation.jsonl (qa_1) | 32345 | CacheBlend | 0.00 | 0.00 | 0.25 | 8.518 | 8.518 | The traveling salesman problem | The traveling salesman problem |
| qa_1_237_len31474.json | validation.jsonl (qa_1) | 31474 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.142 | 8.142 | Turing machine | Turing machine |
| qa_1_238_len29505.json | validation.jsonl (qa_1) | 29505 | CacheBlend | 0.00 | 0.00 | 0.15 | 8.419 | 8.419 | If P were ultimately proven to be equal  | If P were ultimately proven to be equal  |
| qa_1_239_len31551.json | validation.jsonl (qa_1) | 31551 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.236 | 8.236 | Protein structure prediction | Protein structure prediction |
| qa_1_240_len30933.json | validation.jsonl (qa_1) | 30933 | CacheBlend | 0.00 | 1.00 | 0.54 | 8.788 | 8.788 | The Clay Mathematics Institute offers a  | The Clay Mathematics Institute offers a  |
| qa_1_241_len31312.json | validation.jsonl (qa_1) | 31312 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.090 | 8.090 | Ladner | Ladner |
| qa_1_242_len32560.json | validation.jsonl (qa_1) | 32560 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.542 | 8.542 | NP-intermediate | NP-intermediate |
| qa_1_243_len31183.json | validation.jsonl (qa_1) | 31183 | CacheBlend | 0.00 | 1.00 | 0.38 | 8.820 | 8.820 | The graph isomorphism problem is an exam | The graph isomorphism problem is an exam |
| qa_1_244_len31226.json | validation.jsonl (qa_1) | 31226 | CacheBlend | 0.00 | 1.00 | 0.31 | 8.998 | 8.998 | The graph isomorphism problem is the com | The graph isomorphism problem is the com |
| qa_1_245_len31295.json | validation.jsonl (qa_1) | 31295 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.055 | 8.055 | NP-complete | NP-complete |
| qa_1_246_len32112.json | validation.jsonl (qa_1) | 32112 | CacheBlend | 0.00 | 1.00 | 0.07 | 9.296 | 9.296 | It does not imply the graph isomorphism  | It does not imply the graph isomorphism  |
| qa_1_247_len32297.json | validation.jsonl (qa_1) | 32297 | CacheBlend | 0.00 | 1.00 | 0.11 | 9.038 | 9.038 | If graph isomorphism is NP-complete, the | If graph isomorphism is NP-complete, the |
| qa_1_248_len31640.json | validation.jsonl (qa_1) | 31640 | CacheBlend | 0.00 | 1.00 | 0.70 | 8.468 | 8.468 | László Babai and Eugene Luks | László Babai and Eugene Luks |
| qa_1_249_len31866.json | validation.jsonl (qa_1) | 31866 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.346 | 8.346 | The integer factorization problem | The integer factorization problem |
| qa_1_250_len31586.json | validation.jsonl (qa_1) | 31586 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.127 | 8.127 | k | k |
| qa_1_251_len28664.json | validation.jsonl (qa_1) | 28664 | CacheBlend | 0.00 | 1.00 | 0.67 | 7.125 | 7.125 | RSA | RSA |
| qa_1_252_len31535.json | validation.jsonl (qa_1) | 31535 | CacheBlend | 0.00 | 1.00 | 0.38 | 8.602 | 8.602 | The best known algorithm for integer fac | The best known algorithm for integer fac |
| qa_1_253_len31320.json | validation.jsonl (qa_1) | 31320 | CacheBlend | 0.00 | 0.00 | 0.21 | 8.782 | 8.782 | The unproven assumption generally ascrib | The unproven assumption generally ascrib |
| qa_1_254_len31413.json | validation.jsonl (qa_1) | 31413 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.388 | 8.388 | P ⊆ NP ⊆ PP ⊆ PSPACE | P ⊆ NP ⊆ PP ⊆ PSPACE |
| qa_1_255_len30881.json | validation.jsonl (qa_1) | 30881 | CacheBlend | 0.00 | 0.00 | 0.09 | 8.781 | 8.781 | The complexity classes RP, BPP, PP, BQP, | The complexity classes RP, BPP, PP, BQP, |
| qa_1_256_len31282.json | validation.jsonl (qa_1) | 31282 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.641 | 8.641 | If P is not equal to NP, then the polyno | If P is not equal to NP, then the polyno |
| qa_1_257_len32623.json | validation.jsonl (qa_1) | 32623 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.549 | 8.549 | co-NP | co-NP |
| qa_1_258_len30674.json | validation.jsonl (qa_1) | 30674 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.197 | 8.197 | The yes/no answers of a complement probl | The yes/no answers of a complement probl |
| qa_1_259_len29117.json | validation.jsonl (qa_1) | 29117 | CacheBlend | 0.00 | 1.00 | 0.11 | 7.908 | 7.908 | P is not equal to co-NP is commonly beli | P is not equal to co-NP is commonly beli |
| qa_1_260_len31335.json | validation.jsonl (qa_1) | 31335 | CacheBlend | 0.00 | 0.00 | 0.16 | 8.778 | 8.778 | If P and co-NP are established to be une | If P and co-NP are established to be une |
| qa_1_261_len31149.json | validation.jsonl (qa_1) | 31149 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.073 | 8.073 | logarithmic space | logarithmic space |
| qa_1_262_len31104.json | validation.jsonl (qa_1) | 31104 | CacheBlend | 0.00 | 1.00 | 0.41 | 8.924 | 8.924 | L (the set of all problems that can be s | L (the set of all problems that can be s |
| qa_1_263_len30795.json | validation.jsonl (qa_1) | 30795 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.841 | 7.841 | NL | NL |
| qa_1_264_len31684.json | validation.jsonl (qa_1) | 31684 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.225 | 8.225 | NL and NC | NL and NC |
| qa_1_265_len32121.json | validation.jsonl (qa_1) | 32121 | CacheBlend | 0.00 | 0.00 | 0.11 | 9.316 | 9.316 | It is unknown if L is strictly contained | It is unknown if L is strictly contained |
| qa_1_266_len31937.json | validation.jsonl (qa_1) | 31937 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.352 | 8.352 | intractable problems | intractable problems |
| qa_1_267_len31146.json | validation.jsonl (qa_1) | 31146 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.092 | 8.092 | polynomial time algorithm | polynomial time algorithm |
| qa_1_268_len31853.json | validation.jsonl (qa_1) | 31853 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.298 | 8.298 | NP-complete problems | NP-complete problems |
| qa_1_269_len31835.json | validation.jsonl (qa_1) | 31835 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.233 | 8.233 | NP | NP |
| qa_1_270_len30737.json | validation.jsonl (qa_1) | 30737 | CacheBlend | 0.00 | 0.00 | 0.14 | 9.062 | 9.062 | For problems that can be solved in theor | For problems that can be solved in theor |
| qa_1_271_len30881.json | validation.jsonl (qa_1) | 30881 | CacheBlend | 0.00 | 0.00 | 0.26 | 9.159 | 9.159 | The knapsack problem is an example where | The knapsack problem is an example where |
| qa_1_272_len32130.json | validation.jsonl (qa_1) | 32130 | CacheBlend | 0.00 | 0.00 | 0.07 | 9.308 | 9.308 | The best known algorithm for the knapsac | The best known algorithm for the knapsac |
| qa_1_273_len30842.json | validation.jsonl (qa_1) | 30842 | CacheBlend | 0.00 | 0.00 | 0.20 | 8.010 | 8.010 | The knapsack problem | The knapsack problem |
| qa_1_274_len30868.json | validation.jsonl (qa_1) | 30868 | CacheBlend | 0.00 | 0.00 | 0.10 | 8.905 | 8.905 | The most influential among these was the | The most influential among these was the |
| qa_1_275_len32013.json | validation.jsonl (qa_1) | 32013 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.282 | 8.282 | Alan Turing | Alan Turing |
| qa_1_276_len31867.json | validation.jsonl (qa_1) | 31867 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.295 | 8.295 | Turing machine | Turing machine |
| qa_1_277_len31901.json | validation.jsonl (qa_1) | 31901 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.342 | 8.342 | 1936 | 1936 |
| qa_1_278_len29925.json | validation.jsonl (qa_1) | 29925 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.963 | 7.963 | a theoretical device that manipulates sy | a theoretical device that manipulates sy |
| qa_1_279_len31001.json | validation.jsonl (qa_1) | 31001 | CacheBlend | 0.00 | 1.00 | 0.48 | 8.542 | 8.542 | "On the Computational Complexity of Algo | "On the Computational Complexity of Algo |
| qa_1_280_len30550.json | validation.jsonl (qa_1) | 30550 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.108 | 8.108 | Juris Hartmanis and Richard Stearns | Juris Hartmanis and Richard Stearns |
| qa_1_281_len29974.json | validation.jsonl (qa_1) | 29974 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.693 | 7.693 | 1965 | 1965 |
| qa_1_282_len31053.json | validation.jsonl (qa_1) | 31053 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.050 | 8.050 | time and space | time and space |
| qa_1_283_len30808.json | validation.jsonl (qa_1) | 30808 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.026 | 8.026 | 1965 | 1965 |
| qa_1_284_len31855.json | validation.jsonl (qa_1) | 31855 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.335 | 8.335 | John Myhill | John Myhill |
| qa_1_285_len32677.json | validation.jsonl (qa_1) | 32677 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.660 | 8.660 | 1961 | 1961 |
| qa_1_286_len30262.json | validation.jsonl (qa_1) | 30262 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.952 | 7.952 | Boris Trakhtenbrot | Boris Trakhtenbrot |
| qa_1_287_len30107.json | validation.jsonl (qa_1) | 30107 | CacheBlend | 0.00 | 1.00 | 0.25 | 7.804 | 7.804 | the concrete choice of input encoding | the concrete choice of input encoding |
| qa_1_288_len31437.json | validation.jsonl (qa_1) | 31437 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.197 | 8.197 | the choice of input encoding | the choice of input encoding |
| qa_1_289_len31295.json | validation.jsonl (qa_1) | 31295 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.127 | 8.127 | Manuel Blum | Manuel Blum |
| qa_1_290_len31128.json | validation.jsonl (qa_1) | 31128 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.054 | 8.054 | speed-up theorem | speed-up theorem |
| qa_1_291_len30987.json | validation.jsonl (qa_1) | 30987 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.201 | 8.201 | "Reducibility Among Combinatorial Proble | "Reducibility Among Combinatorial Proble |
| qa_1_292_len32118.json | validation.jsonl (qa_1) | 32118 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.330 | 8.330 | 21 | 21 |
| qa_1_293_len32519.json | validation.jsonl (qa_1) | 32519 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.487 | 8.487 | SoCal | SoCal |
| qa_1_294_len29827.json | validation.jsonl (qa_1) | 29827 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.593 | 7.593 | 10 counties | 10 counties |
| qa_1_295_len30052.json | validation.jsonl (qa_1) | 30052 | CacheBlend | 0.00 | 1.00 | 0.22 | 8.167 | 8.167 | Southern California is a major economic  | Southern California is a major economic  |
| qa_1_296_len30599.json | validation.jsonl (qa_1) | 30599 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.910 | 7.910 | demographics and economic ties | demographics and economic ties |
| qa_1_297_len32212.json | validation.jsonl (qa_1) | 32212 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.423 | 8.423 | historical political divisions | historical political divisions |
| qa_1_298_len32464.json | validation.jsonl (qa_1) | 32464 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.652 | 8.652 | The Southern California Megaregion | The Southern California Megaregion |
| qa_1_299_len31144.json | validation.jsonl (qa_1) | 31144 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.020 | 8.020 | 11 | 11 |
| qa_1_300_len32131.json | validation.jsonl (qa_1) | 32131 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.427 | 8.427 | Nevada | Nevada |
| qa_1_301_len31150.json | validation.jsonl (qa_1) | 31150 | CacheBlend | 0.00 | 1.00 | 0.86 | 8.043 | 8.043 | the Mexican border | the Mexican border |
| qa_1_302_len32292.json | validation.jsonl (qa_1) | 32292 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.452 | 8.452 | Tijuana | Tijuana |
| qa_1_303_len31781.json | validation.jsonl (qa_1) | 31781 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.230 | 8.230 | Pacific Coast | Pacific Coast |
| qa_1_304_len32194.json | validation.jsonl (qa_1) | 32194 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.304 | 8.304 | seven | seven |
| qa_1_305_len32172.json | validation.jsonl (qa_1) | 32172 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.387 | 8.387 | 12 million | 12 million |
| qa_1_306_len32614.json | validation.jsonl (qa_1) | 32614 | CacheBlend | 0.00 | 1.00 | 0.57 | 8.845 | 8.845 | The San Diego area with over 3 million i | The San Diego area with over 3 million i |
| qa_1_307_len30463.json | validation.jsonl (qa_1) | 30463 | CacheBlend | 0.00 | 0.00 | 0.33 | 8.058 | 8.058 | 17,786,419 | 17,786,419 |
| qa_1_308_len32638.json | validation.jsonl (qa_1) | 32638 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.554 | 8.554 | The Colorado River | The Colorado River |
| qa_1_309_len29928.json | validation.jsonl (qa_1) | 29928 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.594 | 7.594 | Colorado Desert | Colorado Desert |
| qa_1_310_len28522.json | validation.jsonl (qa_1) | 28522 | CacheBlend | 0.00 | 0.00 | 0.29 | 7.135 | 7.135 | Colorado Desert | Colorado Desert |
| qa_1_311_len31561.json | validation.jsonl (qa_1) | 31561 | CacheBlend | 0.00 | 0.00 | 0.43 | 8.285 | 8.285 | Mexico-Mexican border | Mexico-Mexican border |
| qa_1_312_len31698.json | validation.jsonl (qa_1) | 31698 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.168 | 8.168 | California | California |
| qa_1_313_len32213.json | validation.jsonl (qa_1) | 32213 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.626 | 8.626 | 3,792,621 | 3,792,621 |
| qa_1_314_len29980.json | validation.jsonl (qa_1) | 29980 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.603 | 7.603 | Los Angeles | Los Angeles |
| qa_1_315_len31592.json | validation.jsonl (qa_1) | 31592 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.175 | 8.175 | San Diego | San Diego |
| qa_1_316_len31813.json | validation.jsonl (qa_1) | 31813 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.257 | 8.257 | southwest | southwest |
| qa_1_317_len31451.json | validation.jsonl (qa_1) | 31451 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.105 | 8.105 | Ventura | Ventura |
| qa_1_318_len29239.json | validation.jsonl (qa_1) | 29239 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.336 | 7.336 | United States | United States |
| qa_1_319_len31507.json | validation.jsonl (qa_1) | 31507 | CacheBlend | 0.00 | 1.00 | 0.23 | 9.081 | 9.081 | Los Angeles, Orange, San Diego, San Bern | Los Angeles, Orange, San Diego, San Bern |
| qa_1_320_len31578.json | validation.jsonl (qa_1) | 31578 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.174 | 8.174 | San Diego | San Diego |
| qa_1_321_len31108.json | validation.jsonl (qa_1) | 31108 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.693 | 8.693 | The smallest geographical region discuss | The smallest geographical region discuss |
| qa_1_322_len31775.json | validation.jsonl (qa_1) | 31775 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.253 | 8.253 | Hollywood | Hollywood |
| qa_1_323_len31335.json | validation.jsonl (qa_1) | 31335 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.068 | 8.068 | Los Angeles | Los Angeles |
| qa_1_324_len30926.json | validation.jsonl (qa_1) | 30926 | CacheBlend | 0.00 | 1.00 | 0.83 | 8.117 | 8.117 | The Walt Disney Company owns ABC. | The Walt Disney Company owns ABC. |
| qa_1_325_len31401.json | validation.jsonl (qa_1) | 31401 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.164 | 8.164 | The music industry. | The music industry. |
| qa_1_326_len31852.json | validation.jsonl (qa_1) | 31852 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.266 | 8.266 | Sony Pictures | Sony Pictures |
| qa_1_327_len31029.json | validation.jsonl (qa_1) | 31029 | CacheBlend | 0.00 | 1.00 | 0.86 | 8.047 | 8.047 | skateboard culture | skateboard culture |
| qa_1_328_len30992.json | validation.jsonl (qa_1) | 30992 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.957 | 7.957 | Tony Hawk | Tony Hawk |
| qa_1_329_len30654.json | validation.jsonl (qa_1) | 30654 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.864 | 7.864 | Shaun White | Shaun White |
| qa_1_330_len30417.json | validation.jsonl (qa_1) | 30417 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.756 | 7.756 | Oahu | Oahu |
| qa_1_331_len31615.json | validation.jsonl (qa_1) | 31615 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.181 | 8.181 | Transpac | Transpac |
| qa_1_332_len29893.json | validation.jsonl (qa_1) | 29893 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.612 | 7.612 | Palm Springs | Palm Springs |
| qa_1_333_len32011.json | validation.jsonl (qa_1) | 32011 | CacheBlend | 0.00 | 1.00 | 0.38 | 8.688 | 8.688 | many locals and tourists frequent the so | many locals and tourists frequent the so |
| qa_1_334_len31569.json | validation.jsonl (qa_1) | 31569 | CacheBlend | 0.00 | 1.00 | 0.86 | 8.236 | 8.236 | southern California | southern California |
| qa_1_335_len30554.json | validation.jsonl (qa_1) | 30554 | CacheBlend | 0.00 | 1.00 | 0.29 | 8.122 | 8.122 | Palm Springs is popular for its nearby o | Palm Springs is popular for its nearby o |
| qa_1_336_len32077.json | validation.jsonl (qa_1) | 32077 | CacheBlend | 0.00 | 1.00 | 0.95 | 8.729 | 8.729 | 37° 9' 58.23" latitude | 37° 9' 58.23" latitude |
| qa_1_337_len31075.json | validation.jsonl (qa_1) | 31075 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.027 | 8.027 | 11 miles | 11 miles |
| qa_1_338_len29543.json | validation.jsonl (qa_1) | 29543 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.464 | 7.464 | 10 | 10 |
| qa_1_339_len31626.json | validation.jsonl (qa_1) | 31626 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.285 | 8.285 | the Tehachapi Mountains | the Tehachapi Mountains |
| qa_1_340_len31882.json | validation.jsonl (qa_1) | 31882 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.290 | 8.290 | northern boundary | northern boundary |
| qa_1_341_len31066.json | validation.jsonl (qa_1) | 31066 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.953 | 7.953 | Mexico | Mexico |
| qa_1_342_len31418.json | validation.jsonl (qa_1) | 31418 | CacheBlend | 0.00 | 1.00 | 0.17 | 8.407 | 8.407 | Los Angeles is in the lower part of Alta | Los Angeles is in the lower part of Alta |
| qa_1_343_len29351.json | validation.jsonl (qa_1) | 29351 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.466 | 7.466 | Monterey | Monterey |
| qa_1_344_len30649.json | validation.jsonl (qa_1) | 30649 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.073 | 8.073 | The Compromise of 1850 | The Compromise of 1850 |
| qa_1_345_len31504.json | validation.jsonl (qa_1) | 31504 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.111 | 8.111 | free state | free state |
| qa_1_346_len31588.json | validation.jsonl (qa_1) | 31588 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.317 | 8.317 | pro-slavery southerners | pro-slavery southerners |
| qa_1_347_len32298.json | validation.jsonl (qa_1) | 32298 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.541 | 8.541 | the "Cow Counties" | the "Cow Counties" |
| qa_1_348_len30576.json | validation.jsonl (qa_1) | 30576 | CacheBlend | 0.00 | 1.00 | 0.67 | 7.804 | 7.804 | three times | three times |
| qa_1_349_len30863.json | validation.jsonl (qa_1) | 30863 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.951 | 7.951 | 75% | 75% |
| qa_1_350_len31952.json | validation.jsonl (qa_1) | 31952 | CacheBlend | 0.00 | 1.00 | 0.40 | 8.352 | 8.352 | Senator Milton Latham | Senator Milton Latham |
| qa_1_351_len31055.json | validation.jsonl (qa_1) | 31055 | CacheBlend | 0.00 | 1.00 | 0.23 | 8.953 | 8.953 | The Los Angeles Times defined southern C | The Los Angeles Times defined southern C |
| qa_1_352_len31369.json | validation.jsonl (qa_1) | 31369 | CacheBlend | 0.00 | 0.00 | 0.50 | 8.137 | 8.137 | 1999 | 1999 |
| qa_1_353_len30108.json | validation.jsonl (qa_1) | 30108 | CacheBlend | 0.00 | 0.00 | 0.25 | 7.732 | 7.732 | 2010 | 2010 |
| qa_1_354_len31497.json | validation.jsonl (qa_1) | 31497 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.107 | 8.107 | Imperial | Imperial |
| qa_1_355_len31182.json | validation.jsonl (qa_1) | 31182 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.021 | 8.021 | 10 | 10 |
| qa_1_356_len30802.json | validation.jsonl (qa_1) | 30802 | CacheBlend | 0.00 | 1.00 | 0.15 | 8.864 | 8.864 | The two AAA Auto Clubs of the state, the | The two AAA Auto Clubs of the state, the |
| qa_1_357_len28766.json | validation.jsonl (qa_1) | 28766 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.282 | 7.282 | California State Automobile Association | California State Automobile Association |
| qa_1_358_len31443.json | validation.jsonl (qa_1) | 31443 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.199 | 8.199 | three-region point of view | three-region point of view |
| qa_1_359_len31784.json | validation.jsonl (qa_1) | 31784 | CacheBlend | 0.00 | 0.00 | 0.67 | 8.351 | 8.351 | Tehachapi Mountains | Tehachapi Mountains |
| qa_1_360_len32287.json | validation.jsonl (qa_1) | 32287 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.507 | 8.507 | southern California | southern California |
| qa_1_361_len31826.json | validation.jsonl (qa_1) | 31826 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.211 | 8.211 | third | third |
| qa_1_362_len31764.json | validation.jsonl (qa_1) | 31764 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.496 | 8.496 | Much of it has been left undeveloped. | Much of it has been left undeveloped. |
| qa_1_363_len30889.json | validation.jsonl (qa_1) | 30889 | CacheBlend | 0.00 | 0.00 | 0.43 | 8.049 | 8.049 | use heavy automobiles and highways | use heavy automobiles and highways |
| qa_1_364_len31108.json | validation.jsonl (qa_1) | 31108 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.618 | 8.618 | Southern California is famous for using  | Southern California is famous for using  |
| qa_1_365_len32328.json | validation.jsonl (qa_1) | 32328 | CacheBlend | 0.00 | 0.00 | 0.07 | 9.415 | 9.415 | The urban area of southern California co | The urban area of southern California co |
| qa_1_366_len31038.json | validation.jsonl (qa_1) | 31038 | CacheBlend | 0.00 | 1.00 | 0.31 | 8.328 | 8.328 | The main gap to continued urbanization i | The main gap to continued urbanization i |
| qa_1_367_len30408.json | validation.jsonl (qa_1) | 30408 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.904 | 7.904 | the San Diego metropolitan area | the San Diego metropolitan area |
| qa_1_368_len30559.json | validation.jsonl (qa_1) | 30559 | CacheBlend | 0.00 | 1.00 | 0.28 | 9.021 | 9.021 | The United States Census Bureau consider | The United States Census Bureau consider |
| qa_1_369_len32256.json | validation.jsonl (qa_1) | 32256 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.388 | 8.388 | Orange County | Orange County |
| qa_1_370_len31498.json | validation.jsonl (qa_1) | 31498 | CacheBlend | 0.00 | 0.00 | 0.80 | 8.259 | 8.259 | 1980s | 1980s |
| qa_1_371_len30083.json | validation.jsonl (qa_1) | 30083 | CacheBlend | 0.00 | 1.00 | 0.46 | 7.961 | 7.961 | southern California maintains a Mediterr | southern California maintains a Mediterr |
| qa_1_372_len30547.json | validation.jsonl (qa_1) | 30547 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.828 | 7.828 | hot summers | hot summers |
| qa_1_373_len32203.json | validation.jsonl (qa_1) | 32203 | CacheBlend | 0.00 | 0.00 | 0.14 | 8.896 | 8.896 | 83.0 °F (28.3 °C) | 83.0 °F (28.3 °C) |
| qa_1_374_len32014.json | validation.jsonl (qa_1) | 32014 | CacheBlend | 0.00 | 1.00 | 0.20 | 8.696 | 8.696 | Snow is very rare in the Southwest of th | Snow is very rare in the Southwest of th |
| qa_1_375_len32550.json | validation.jsonl (qa_1) | 32550 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.853 | 8.853 | 15 °C (59 °F) | 15 °C (59 °F) |
| qa_1_376_len32319.json | validation.jsonl (qa_1) | 32319 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.617 | 8.617 | one of the more varied collections | one of the more varied collections |
| qa_1_377_len31609.json | validation.jsonl (qa_1) | 31609 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.199 | 8.199 | Pacific Ocean | Pacific Ocean |
| qa_1_378_len31522.json | validation.jsonl (qa_1) | 31522 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.186 | 8.186 | urban landscapes | urban landscapes |
| qa_1_379_len31519.json | validation.jsonl (qa_1) | 31519 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.311 | 8.311 | Peninsular Ranges | Peninsular Ranges |
| qa_1_380_len31072.json | validation.jsonl (qa_1) | 31072 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.008 | 8.008 | deserts | deserts |
| qa_1_381_len31168.json | validation.jsonl (qa_1) | 31168 | CacheBlend | 0.00 | 1.00 | 0.77 | 8.364 | 8.364 | About 10,000 earthquakes. | About 10,000 earthquakes. |
| qa_1_382_len30045.json | validation.jsonl (qa_1) | 30045 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.839 | 8.839 | The earthquakes that hit southern Califo | The earthquakes that hit southern Califo |
| qa_1_383_len31908.json | validation.jsonl (qa_1) | 31908 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.346 | 8.346 | 6.7 | 6.7 |
| qa_1_384_len32179.json | validation.jsonl (qa_1) | 32179 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.373 | 8.373 | property damage | property damage |
| qa_1_385_len31185.json | validation.jsonl (qa_1) | 31185 | CacheBlend | 0.00 | 1.00 | 0.28 | 9.220 | 9.220 | The 1994 Northridge earthquake caused th | The 1994 Northridge earthquake caused th |
| qa_1_386_len32048.json | validation.jsonl (qa_1) | 32048 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.360 | 8.360 | San Andreas Fault | San Andreas Fault |
| qa_1_387_len32039.json | validation.jsonl (qa_1) | 32039 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.490 | 8.490 | magnitude 6.7+ | magnitude 6.7+ |
| qa_1_388_len29885.json | validation.jsonl (qa_1) | 29885 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.678 | 7.678 | Puente Hills Fault | Puente Hills Fault |
| qa_1_389_len32356.json | validation.jsonl (qa_1) | 32356 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.467 | 8.467 | USGS | USGS |
| qa_1_390_len30911.json | validation.jsonl (qa_1) | 30911 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.636 | 8.636 | The earthquake forecast models features  | The earthquake forecast models features  |
| qa_1_391_len31443.json | validation.jsonl (qa_1) | 31443 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.151 | 8.151 | economically | economically |
| qa_1_392_len30635.json | validation.jsonl (qa_1) | 30635 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.771 | 8.771 | some other kind of recognition include b | some other kind of recognition include b |
| qa_1_393_len31652.json | validation.jsonl (qa_1) | 31652 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.215 | 8.215 | economic activity | economic activity |
| qa_1_394_len29458.json | validation.jsonl (qa_1) | 29458 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.520 | 7.520 | 2010 | 2010 |
| qa_1_395_len30255.json | validation.jsonl (qa_1) | 30255 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.748 | 7.748 | high growth rates | high growth rates |
| qa_1_396_len28472.json | validation.jsonl (qa_1) | 28472 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.226 | 7.226 | 10.0% | 10.0% |
| qa_1_397_len30756.json | validation.jsonl (qa_1) | 30756 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.861 | 8.861 | The documents do not provide specific in | The documents do not provide specific in |
| qa_1_398_len31715.json | validation.jsonl (qa_1) | 31715 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.386 | 8.386 | northern part of the state | northern part of the state |
| qa_1_399_len32553.json | validation.jsonl (qa_1) | 32553 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.509 | 8.509 | eight counties | eight counties |
| qa_1_400_len30519.json | validation.jsonl (qa_1) | 30519 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.777 | 7.777 | 2 | 2 |
| qa_1_401_len30995.json | validation.jsonl (qa_1) | 30995 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.923 | 7.923 | five million | five million |
| qa_1_402_len31777.json | validation.jsonl (qa_1) | 31777 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.313 | 8.313 | the Southern Border Region | the Southern Border Region |
| qa_1_403_len30928.json | validation.jsonl (qa_1) | 30928 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.215 | 8.215 | 17,786,419 | 17,786,419 |
| qa_1_404_len30596.json | validation.jsonl (qa_1) | 30596 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.800 | 7.800 | Los Angeles | Los Angeles |
| qa_1_405_len31678.json | validation.jsonl (qa_1) | 31678 | CacheBlend | 0.00 | 0.00 | 0.40 | 8.445 | 8.445 | 1,307,402 | 1,307,402 |
| qa_1_406_len31009.json | validation.jsonl (qa_1) | 31009 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.960 | 7.960 | 12 | 12 |
| qa_1_407_len30930.json | validation.jsonl (qa_1) | 30930 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.105 | 8.105 | 100,000 | 100,000 |
| qa_1_408_len31508.json | validation.jsonl (qa_1) | 31508 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.145 | 8.145 | Riverside | Riverside |
| qa_1_409_len32603.json | validation.jsonl (qa_1) | 32603 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.523 | 8.523 | petroleum | petroleum |
| qa_1_410_len30976.json | validation.jsonl (qa_1) | 30976 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.956 | 7.956 | Hollywood | Hollywood |
| qa_1_411_len30867.json | validation.jsonl (qa_1) | 30867 | CacheBlend | 0.00 | 1.00 | 0.11 | 8.591 | 8.591 | The region was a leader in the housing b | The region was a leader in the housing b |
| qa_1_412_len31735.json | validation.jsonl (qa_1) | 31735 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.459 | 8.459 | dominated and heavily dependent upon abu | dominated and heavily dependent upon abu |
| qa_1_413_len31568.json | validation.jsonl (qa_1) | 31568 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.479 | 8.479 | The housing crash had a heavy impact on  | The housing crash had a heavy impact on  |
| qa_1_414_len32711.json | validation.jsonl (qa_1) | 32711 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.656 | 8.656 | 1920s | 1920s |
| qa_1_415_len32163.json | validation.jsonl (qa_1) | 32163 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.351 | 8.351 | vast | vast |
| qa_1_416_len30849.json | validation.jsonl (qa_1) | 30849 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.916 | 7.916 | cattle | cattle |
| qa_1_417_len30527.json | validation.jsonl (qa_1) | 30527 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.792 | 7.792 | citrus | citrus |
| qa_1_418_len30567.json | validation.jsonl (qa_1) | 30567 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.839 | 7.839 | aerospace | aerospace |
| qa_1_419_len30925.json | validation.jsonl (qa_1) | 30925 | CacheBlend | 0.00 | 1.00 | 0.67 | 7.942 | 7.942 | business districts | business districts |
| qa_1_420_len31772.json | validation.jsonl (qa_1) | 31772 | CacheBlend | 0.00 | 1.00 | 0.67 | 8.275 | 8.275 | Central Business District | Central Business District |
| qa_1_421_len31550.json | validation.jsonl (qa_1) | 31550 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.198 | 8.198 | South Coast Metro | South Coast Metro |
| qa_1_422_len30644.json | validation.jsonl (qa_1) | 30644 | CacheBlend | 0.00 | 1.00 | 0.67 | 7.829 | 7.829 | business district | business district |
| qa_1_423_len31812.json | validation.jsonl (qa_1) | 31812 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.277 | 8.277 | Los Angeles Area | Los Angeles Area |
| qa_1_424_len31379.json | validation.jsonl (qa_1) | 31379 | CacheBlend | 0.00 | 1.00 | 0.43 | 8.386 | 8.386 | Warner Center is located in the San Fern | Warner Center is located in the San Fern |
| qa_1_425_len32255.json | validation.jsonl (qa_1) | 32255 | CacheBlend | 0.00 | 1.00 | 0.50 | 8.433 | 8.433 | Downtown Los Angeles | Downtown Los Angeles |
| qa_1_426_len30733.json | validation.jsonl (qa_1) | 30733 | CacheBlend | 0.00 | 1.00 | 0.06 | 8.825 | 8.825 | The Sand Bernardino - Riverside area mai | The Sand Bernardino - Riverside area mai |
| qa_1_427_len30853.json | validation.jsonl (qa_1) | 30853 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.951 | 7.951 | Riverside | Riverside |
| qa_1_428_len31749.json | validation.jsonl (qa_1) | 31749 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.248 | 8.248 | Downtown Riverside | Downtown Riverside |
| qa_1_429_len31042.json | validation.jsonl (qa_1) | 31042 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.974 | 7.974 | Orange County | Orange County |
| qa_1_430_len30297.json | validation.jsonl (qa_1) | 30297 | CacheBlend | 0.00 | 1.00 | 0.50 | 8.066 | 8.066 | international corporations are headquart | international corporations are headquart |
| qa_1_431_len30087.json | validation.jsonl (qa_1) | 30087 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.640 | 7.640 | West Irvine | West Irvine |
| qa_1_432_len31340.json | validation.jsonl (qa_1) | 31340 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.947 | 8.947 | The Orange County business districts als | The Orange County business districts als |
| qa_1_433_len30556.json | validation.jsonl (qa_1) | 30556 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.802 | 7.802 | rapidly | rapidly |
| qa_1_434_len30846.json | validation.jsonl (qa_1) | 30846 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.984 | 7.984 | Downtown San Diego | Downtown San Diego |
| qa_1_435_len30516.json | validation.jsonl (qa_1) | 30516 | CacheBlend | 0.00 | 0.00 | 0.12 | 8.771 | 8.771 | The majority of San Diego's business dis | The majority of San Diego's business dis |
| qa_1_436_len30213.json | validation.jsonl (qa_1) | 30213 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.650 | 8.650 | Downtown San Bernardino, Downtown Rivers | Downtown San Bernardino, Downtown Rivers |
| qa_1_437_len30183.json | validation.jsonl (qa_1) | 30183 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.675 | 7.675 | San Diego | San Diego |
| qa_1_438_len31529.json | validation.jsonl (qa_1) | 31529 | CacheBlend | 0.00 | 0.00 | 0.40 | 8.234 | 8.234 | San Diego International Airport | San Diego International Airport |
| qa_1_439_len29042.json | validation.jsonl (qa_1) | 29042 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.335 | 7.335 | passenger volume | passenger volume |
| qa_1_440_len32641.json | validation.jsonl (qa_1) | 32641 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.484 | 8.484 | third | third |
| qa_1_441_len31400.json | validation.jsonl (qa_1) | 31400 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.160 | 8.160 | San Diego International Airport | San Diego International Airport |
| qa_1_442_len29771.json | validation.jsonl (qa_1) | 29771 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.577 | 7.577 | Van Nuys Airport | Van Nuys Airport |
| qa_1_443_len32302.json | validation.jsonl (qa_1) | 32302 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.470 | 8.470 | Metrolink | Metrolink |
| qa_1_444_len30420.json | validation.jsonl (qa_1) | 30420 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.688 | 7.688 | Six | Six |
| qa_1_445_len30842.json | validation.jsonl (qa_1) | 30842 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.882 | 7.882 | six | six |
| qa_1_446_len30929.json | validation.jsonl (qa_1) | 30929 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.909 | 7.909 | Orange | Orange |
| qa_1_447_len31793.json | validation.jsonl (qa_1) | 31793 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.349 | 8.349 | The Port of Los Angeles | The Port of Los Angeles |
| qa_1_448_len30896.json | validation.jsonl (qa_1) | 30896 | CacheBlend | 0.00 | 1.00 | 0.38 | 8.432 | 8.432 | The second busiest container port in the | The second busiest container port in the |
| qa_1_449_len30936.json | validation.jsonl (qa_1) | 30936 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.953 | 7.953 | Southern California | Southern California |
| qa_1_450_len32193.json | validation.jsonl (qa_1) | 32193 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.352 | 8.352 | Tech Coast | Tech Coast |
| qa_1_451_len31782.json | validation.jsonl (qa_1) | 31782 | CacheBlend | 0.00 | 1.00 | 0.00 | 9.223 | 9.223 | The region is famous for its large numbe | The region is famous for its large numbe |
| qa_1_452_len31468.json | validation.jsonl (qa_1) | 31468 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.420 | 8.420 | The California Institute of Technology i | The California Institute of Technology i |
| qa_1_453_len31085.json | validation.jsonl (qa_1) | 31085 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.985 | 7.985 | 10 | 10 |
| qa_1_454_len31497.json | validation.jsonl (qa_1) | 31497 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.104 | 8.104 | 12 | 12 |
| qa_1_455_len30461.json | validation.jsonl (qa_1) | 30461 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.729 | 7.729 | NFL | NFL |
| qa_1_456_len30509.json | validation.jsonl (qa_1) | 30509 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.793 | 7.793 | basketball | basketball |
| qa_1_457_len31031.json | validation.jsonl (qa_1) | 31031 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.972 | 7.972 | MLB | MLB |
| qa_1_458_len30982.json | validation.jsonl (qa_1) | 30982 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.991 | 7.991 | Los Angeles Kings | Los Angeles Kings |
| qa_1_459_len30810.json | validation.jsonl (qa_1) | 30810 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.915 | 7.915 | LA Galaxy | LA Galaxy |
| qa_1_460_len31244.json | validation.jsonl (qa_1) | 31244 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.063 | 8.063 | Chivas USA | Chivas USA |
| qa_1_461_len31020.json | validation.jsonl (qa_1) | 31020 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.938 | 7.938 | Two | Two |
| qa_1_462_len30989.json | validation.jsonl (qa_1) | 30989 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.025 | 8.025 | 2014 | 2014 |
| qa_1_463_len30766.json | validation.jsonl (qa_1) | 30766 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.923 | 7.923 | StubHub Center | StubHub Center |
| qa_1_464_len30944.json | validation.jsonl (qa_1) | 30944 | CacheBlend | 0.00 | 1.00 | 0.48 | 8.512 | 8.512 | The suspended team, Chivas, is scheduled | The suspended team, Chivas, is scheduled |
| qa_1_465_len31899.json | validation.jsonl (qa_1) | 31899 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.284 | 8.284 | rugby | rugby |
| qa_1_466_len31771.json | validation.jsonl (qa_1) | 31771 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.242 | 8.242 | UCLA | UCLA |
| qa_1_467_len30828.json | validation.jsonl (qa_1) | 30828 | CacheBlend | 0.00 | 1.00 | 0.57 | 8.023 | 8.023 | USC Trojans | USC Trojans |
| qa_1_468_len30653.json | validation.jsonl (qa_1) | 30653 | CacheBlend | 0.00 | 1.00 | 0.86 | 7.967 | 7.967 | the Pac-12 Conference | the Pac-12 Conference |
| qa_1_469_len29430.json | validation.jsonl (qa_1) | 29430 | CacheBlend | 0.00 | 1.00 | 0.33 | 7.516 | 7.516 | NCAA Division I | NCAA Division I |
| qa_1_470_len30347.json | validation.jsonl (qa_1) | 30347 | CacheBlend | 0.00 | 1.00 | 0.27 | 8.288 | 8.288 | Rugby is a growing sport in southern Cal | Rugby is a growing sport in southern Cal |
| qa_1_471_len31303.json | validation.jsonl (qa_1) | 31303 | CacheBlend | 0.00 | 1.00 | 0.80 | 8.094 | 8.094 | high school level | high school level |
| qa_1_472_len32030.json | validation.jsonl (qa_1) | 32030 | CacheBlend | 0.00 | 0.00 | 0.22 | 8.956 | 8.956 | Rugby is rapidly becoming a growing spor | Rugby is rapidly becoming a growing spor |
| qa_1_473_len32001.json | validation.jsonl (qa_1) | 32001 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.330 | 8.330 | BSkyB | BSkyB |
| qa_1_474_len32455.json | validation.jsonl (qa_1) | 32455 | CacheBlend | 0.00 | 1.00 | 0.14 | 8.861 | 8.861 | Sky UK Limited (formerly British Sky Bro | Sky UK Limited (formerly British Sky Bro |
| qa_1_475_len31865.json | validation.jsonl (qa_1) | 31865 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.333 | 8.333 | 2014 | 2014 |
| qa_1_476_len31147.json | validation.jsonl (qa_1) | 31147 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.972 | 7.972 | Sky plc | Sky plc |
| qa_1_477_len31000.json | validation.jsonl (qa_1) | 31000 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.933 | 7.933 | Sky UK Limited | Sky UK Limited |
| qa_1_478_len32650.json | validation.jsonl (qa_1) | 32650 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.589 | 8.589 | 2006 | 2006 |
| qa_1_479_len29309.json | validation.jsonl (qa_1) | 29309 | CacheBlend | 0.00 | 1.00 | 0.00 | 7.858 | 7.858 | Setanta was awarded two of the six Premi | Setanta was awarded two of the six Premi |
| qa_1_480_len31175.json | validation.jsonl (qa_1) | 31175 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.009 | 8.009 | BSkyB | BSkyB |
| qa_1_481_len31842.json | validation.jsonl (qa_1) | 31842 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.327 | 8.327 | £1.3bn | £1.3bn |
| qa_1_482_len31313.json | validation.jsonl (qa_1) | 31313 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.055 | 8.055 | ONdigital | ONdigital |
| qa_1_483_len31429.json | validation.jsonl (qa_1) | 31429 | CacheBlend | 0.00 | 1.00 | 0.11 | 8.574 | 8.574 | BSkyB teamed up with ITV Digital's free- | BSkyB teamed up with ITV Digital's free- |
| qa_1_484_len31369.json | validation.jsonl (qa_1) | 31369 | CacheBlend | 0.00 | 0.00 | 0.00 | 8.033 | 8.033 | 3 | 3 |
| qa_1_485_len32004.json | validation.jsonl (qa_1) | 32004 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.297 | 8.297 | Sky Three | Sky Three |
| qa_1_486_len30152.json | validation.jsonl (qa_1) | 30152 | CacheBlend | 0.00 | 0.00 | 0.00 | 7.624 | 7.624 | Sky Three | Sky Three |
| qa_1_487_len31993.json | validation.jsonl (qa_1) | 31993 | CacheBlend | 0.00 | 1.00 | 0.21 | 8.837 | 8.837 | BSkyB initially charged additional subsc | BSkyB initially charged additional subsc |
| qa_1_488_len31846.json | validation.jsonl (qa_1) | 31846 | CacheBlend | 0.00 | 0.00 | 0.62 | 8.349 | 8.349 | In January 2007 | In January 2007 |
| qa_1_489_len31273.json | validation.jsonl (qa_1) | 31273 | CacheBlend | 0.00 | 1.00 | 0.10 | 8.746 | 8.746 | customers that do not subscribe to BSkyB | customers that do not subscribe to BSkyB |
| qa_1_490_len32200.json | validation.jsonl (qa_1) | 32200 | CacheBlend | 0.00 | 1.00 | 0.35 | 9.309 | 9.309 | In February 2011, BSkyB discontinued the | In February 2011, BSkyB discontinued the |
| qa_1_491_len31410.json | validation.jsonl (qa_1) | 31410 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.112 | 8.112 | Sky+HD Box | Sky+HD Box |
| qa_1_492_len32464.json | validation.jsonl (qa_1) | 32464 | CacheBlend | 0.00 | 1.00 | 0.40 | 8.748 | 8.748 | BSkyB uses the VideoGuard pay-TV scrambl | BSkyB uses the VideoGuard pay-TV scrambl |
| qa_1_493_len32303.json | validation.jsonl (qa_1) | 32303 | CacheBlend | 0.00 | 1.00 | 0.00 | 8.735 | 8.735 | VideoGuard is owned by NDS, a Cisco Syst | VideoGuard is owned by NDS, a Cisco Syst |
| qa_1_494_len30821.json | validation.jsonl (qa_1) | 30821 | CacheBlend | 1.00 | 1.00 | 1.00 | 7.834 | 7.834 | Cisco Systems | Cisco Systems |
| qa_1_495_len31257.json | validation.jsonl (qa_1) | 31257 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.046 | 8.046 | BSkyB | BSkyB |
| qa_1_496_len31906.json | validation.jsonl (qa_1) | 31906 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.282 | 8.282 | Sky+ | Sky+ |
| qa_1_497_len31565.json | validation.jsonl (qa_1) | 31565 | CacheBlend | 1.00 | 1.00 | 1.00 | 8.166 | 8.166 | 2007 | 2007 |
| qa_1_498_len31165.json | validation.jsonl (qa_1) | 31165 | CacheBlend | 0.00 | 1.00 | 0.25 | 8.478 | 8.478 | The basic channels were removed from the | The basic channels were removed from the |
| qa_1_499_len30934.json | validation.jsonl (qa_1) | 30934 | CacheBlend | 0.00 | 1.00 | 0.77 | 8.394 | 8.394 | Virgin Media claimed that BSkyB had subs | Virgin Media claimed that BSkyB had subs |

## 二、按方法聚合（每轮一行，再给出跨轮中位数）

| 方法 | 轮次 | EM | SM | F1 | TTFT缓存均值(s) | TTFT全量均值(s) | 缓存加速比 |
|---|---|---|---|---|---|------|---|
| CacheBlend | r1 | 0.370 | 0.700 | 0.517 | 8.375 | 8.375 | 1.00× |
| CacheBlend | r2 | 0.400 | 0.640 | 0.555 | 8.212 | 8.212 | 1.00× |
| CacheBlend | r3 | 0.420 | 0.690 | 0.562 | 8.324 | 8.324 | 1.00× |
| CacheBlend | r4 | 0.420 | 0.730 | 0.615 | 8.242 | 8.242 | 1.00× |
| CacheBlend | r5 | 0.540 | 0.820 | 0.654 | 8.197 | 8.197 | 1.00× |
| **CacheBlend** | **中位** | **0.420** | **0.700** | **0.562** | **8.242** | **8.242** | **1.00×** |
| vLLM全量 | r1 | 0.370 | 0.700 | 0.517 | - | 8.375 | 1.00× |
| vLLM全量 | r2 | 0.400 | 0.640 | 0.555 | - | 8.212 | 1.00× |
| vLLM全量 | r3 | 0.420 | 0.690 | 0.562 | - | 8.324 | 1.00× |
| vLLM全量 | r4 | 0.420 | 0.730 | 0.615 | - | 8.242 | 1.00× |
| vLLM全量 | r5 | 0.540 | 0.820 | 0.654 | - | 8.197 | 1.00× |
| **vLLM全量** | **中位** | **0.420** | **0.700** | **0.562** | - | **8.242** | **1.00×** |

## 三、按数据集×方法聚合（跨轮中位数；QA 数据集看 SM/F1，samsum 摘要看 F1）

| 数据集 | 样本数 | 方法 | EM中位 | SM中位 | F1中位 | TTFT缓存中位(s) |
|---|---|---|---|---|---|---|
| validation.jsonl (qa_1) | 500 | CacheBlend | 0.000 | 1.000 | 0.667 | 8.263 |
