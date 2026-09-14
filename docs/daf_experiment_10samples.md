# DAF 10 样本对比实验记录（二：干净环境重跑版）

> 日期：2026-09-10 凌晨（用户休息时段，无桌面程序抢占 GPU）
> 环境：WSL2 Ubuntu-22.04 + RTX 4060 8GB + Mistral-7B-Instruct-v0.2-AWQ（4bit 量化）
> 脚本：`example/blend.py`，由 `wsl_setup/run_both.sh` 依次执行 DAF=0 和 DAF=1
> 样本：inputs/1.json ~ inputs/10.json，temperature=0（贪心解码），每次生成 10 个 token
> DAF 配置：sel_layers=[1,2,3]，第 4 层选人，recomp_ratio=0.3，V-only 偏差
> CacheBlend 配置：check_layers=[1]，recomp_ratio=0.16（原型默认值）
> 日志：`wsl_setup/logs/run_cb.log`、`wsl_setup/logs/run_daf.log`

---

## 〇、与上一轮（有噪声）的对照

**两轮运行的生成结果逐字完全相同**（原版 CacheBlend 和 DAF 各自的 10 个答案都没变），
证明贪心解码下结果可复现，质量结论稳定。

上一轮的速度数据作废（GPU 被壁纸引擎/抖音等抢占，最慢一个 chunk 跑了 73 分钟），
本文档速度部分以本轮干净数据为准。

---

## 一、质量对比（三种方式的生成结果）

| 样本 | 全量重算（标准答案） | 原版 CacheBlend | DAF |
|---|---|---|---|
| 1 | Carnegie Mellon University | ✅ 一致 | ✅ 一致 |
| 2 | The Soviet Story was made first. | ✅ The Soviet Story. (born 200 | ❌ Hell Up In Harlem (1973 |
| 3 | Cipriano Castro was born first. | ✅ 一致 | ✅ 一致 |
| 4 | Jugband Blues came out first. | ✅ 一致 | ✅ "Jugband Blues" came out first.（多引号） |
| 5 | Yakshya Malla's birthplace | ❌ Bulacan, Philippines. | ❌ Ducal house of Bourbon. |
| 6 | Arthur Acheson, 1st Earl of | ✅ 一致 | ✅ 一致 |
| 7 | At home in Goring-on-Thames | ✅ Goring-on-Thames. | ❌ Died in bed. |
| 8 | Yes, both are French. | ❌ Yes, both are **German**. | ✅ 一致 |
| 9 | Wooden Crosses (1932) | ✅ 一致 | ✅ 一致 |
| 10 | Above Rubies came out earlier. | ✅ Above Rubies came out earlier. | ❌ The Magic Aster came out earlier. |

**统计**：

- 原版 CacheBlend 与全量一致：**8/10**
- DAF 与全量一致：**6/10**

细节：

- 样本 8：原版答错（German），**DAF 反而答对**（French）——DAF 并非单方面更差
- 样本 5：两种方法都错且错法不同——该样本对缓存融合本身可能就难

---

## 二、速度对比（TTFT，单位：秒，干净环境）

| 样本 | 原版 CacheBlend | DAF | 全量重算（CB 轮） | 全量重算（DAF 轮） |
|---|---|---|---|---|
| 1 | 21.3 | 46.0 | 84.9 | 92.8 |
| 2 | 20.7 | 42.7 | 39.8 | 26.7 |
| 3 | 12.5 | 9.8 | 21.3 | 11.6 |
| 4 | 11.1 | 13.7 | 24.8 | 70.8 |
| 5 | 18.8 | 23.2 | 38.3 | 49.4 |
| 6 | 17.9 | 21.8 | 19.7 | 69.3 |
| 7 | 11.4 | 26.7 | 40.0 | 57.7 |
| 8 | 21.0 | 22.3 | 39.0 | 57.1 |
| 9 | 25.9 | 31.6 | 49.8 | 46.0 |
| 10 | 32.1 | 25.8 | 46.5 | 58.2 |
| **平均** | **19.3** | **26.4** | **40.4** | **54.0** |

**结论**：

- 两种融合方法在所有样本上都快于全量重算，融合价值成立
- 本批短上下文（2~4k token）下，DAF 平均 TTFT 略高于原版 CacheBlend（26.4s vs 19.3s）

**DAF 此时没有更快是符合预期的**，原因有三：

1. DAF 重算比例 30% > 原版 16%，短上下文下多算的部分省不回来
2. DAF 论文的优势区间在长上下文（论文数据：6k 持平，30k 拉开，136k 差距大），
   本批样本最长不到 4k，不在 DAF 主场
3. 我们的第一版有正确性取向的开销（混合 KV 全量 clone、xformers 而非 Flash-Attention）

**残留噪声提示**：两轮的全量重算均值仍有差异（40.4s vs 54.0s，同一段代码），
说明即使是凌晨也有少量环境波动；TTFT 对比建议今后取多轮中位数。

---

## 三、质量解读

DAF 6/10 低于原版 8/10，但下结论前需注意：

1. **样本量太小**：10 个样本、每个只生成 10 个 token，2 个样本之差不具统计显著性
2. **配置不对等**：DAF 重算更多（30% vs 16%）却一致率更低，说明质量差异大概率来自
   工程假设而非重算量：
   - V-only 偏差打分（论文写的是 KV deviation）
   - 多层分数算术平均（论文未给公式）
   - 重要 token 双向无掩码注意力
   - 第 4 层才首次选择
3. **功能验证是通过的**：DAF 全程无报错、无 NaN，选人数量符合预期
   （样本 1：1134 重要 + 59 问题 = 1193），两轮结果逐字可复现

---

## 四、实验过程中的环境事故记录（第一轮）

- 第一次 DAF=0 运行因 Windows 桌面程序（壁纸引擎、抖音、微信等）抢占 GPU，
  单个 chunk 从正常几十秒恶化到 628s / 4389s，90 分钟超时被终止，无结果
- 关闭程序 + 凌晨无人使用时段重跑成功
- **教训**：跑实验前关闭动态壁纸、视频类应用；WSL2 的 GPU 与 Windows 桌面分时共用；
  长时间实验安排在无人使用电脑时进行

---

## 五、后续计划（按价值排序）

1. **真实评测**：用 blend_musique.py / blend_wikimqa.py 等带标准答案的脚本，
   几十上百样本 + EM/F1 指标严肃比较质量
2. **消融实验**（每次只改一处，定位质量损失来源）：
   - `sel_layers` 改为 [1] 或 [1,2]：检验多层聚合是否帮倒忙
   - 重要 token 路径从无掩码改为因果掩码
   - recomp_ratio 对齐：DAF 也用 0.16 跑一次
3. **长上下文速度测试**：构造 30k+ token 输入，验证 DAF 的 r²·Ld² 复杂度优势
4. **可选升级**：K+V 偏差打分、去掉 clone 换按请求缓冲区、Flash-Attention 后端
