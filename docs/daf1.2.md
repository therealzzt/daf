# DAF 1.2 实验总结：长上下文第一档（6k–10k，官方数据集样本）

> 日期：2026-09-15
> **路径变更（2026-09-16 整理）**：实验文件已从仓库根目录迁入 `experiments/`，本文中的旧路径对应关系：
> 样本 `inputs_long_6k10k_official/` → `experiments/inputs_long_6k10k_official/`；
> 脚本（make_inputs…/score_em…/resume.sh）→ `experiments/scripts/`；
> 日志 `run_*_long6k10k_*.log` → `experiments/logs/long6k10k/`；
> 报告 `em_report_long_6k10k.md` → `experiments/reports/`；
> 作废档 `logs_long6k10k_maxtok10截断作废/` → `experiments/logs/archived_maxtok10截断作废/`。
> 总索引见 `experiments/README.md`。
> 环境：Ubuntu 24.04 服务器 + RTX 3090 Ti 24GB（实验用 GPU 0）+ vllm 0.4.1（本仓库 vllm_blend）+ torch 2.2.1/cu121 + xformers 0.0.25 + transformers 4.41.2
> 模型：Mistral-7B-Instruct-v0.2 官方 bf16 非量化版（15GB）
> 前作：DAF 1.1（bf16、2–4k 样本）见 `docs/daf1.1.md`；AWQ 版见 `docs/daf_experiment_10samples.md`
> 数据：`inputs_long_6k10k_official/`（24 样本，带标准答案），挑选明细 `_selection_report.md`
> 运行：每方法 3 轮（幂等续跑脚本 `run_long6k10k_resume.sh`，nohup 启动可断线续跑）
> 日志：`run_{cb,daf}_long6k10k_r{1,2,3}.log`；评分报告 `em_report_long_6k10k.md`

---

## 一、本档做了什么

把输入长度从 1.x 的 2–4k 拉到 **6k–10k**（论文 Table 1 里 6k/9k 档位的区间），验证两个论断：
1. 论文说 9k 时 DAF 与 CacheBlend **速度持平**——我们的实现是否复现？
2. 用带标准答案的真实数据集（而非 1.x 的"与全量重算对比"）严格评测质量。

### 新增/改动文件（命名均含出处）

| 文件 | 作用 |
|---|---|
| `example/make_inputs_long_6k10k_official.py` | 从 `inputs/{musique_s,wikimqa_s,samsum}.json` 挑 6k–10k 样本并转成 blend.py 格式；文件名 `{数据集}_{原下标}_len{token数}.json` 一眼可溯源 |
| `inputs_long_6k10k_official/` | 24 个样本（musique/wikimqa/samsum 各 8）+ `_selection_report.md` 挑选报告 |
| `example/blend.py` 改动① | `max_model_len` 4096→**16384**（vllm 0.4.1 要求 max_model_len ≤ KV 池容量；bf16 权重 15GB 在 24GB 卡上池约 1.9 万 token，设 32768 会启动报错；16384 对本档 9.6k 上限留足余量） |
| `example/blend.py` 改动② | 生成 `max_tokens` 10→**32**（对齐 `blend_musique.py`；10 token 会把答案截成残句，EM 恒为 0——见第三节教训） |
| `example/blend.py` 改动③ | 支持 `BLEND_INPUT_DIR` 环境变量指定样本目录（默认 `inputs/`，行为与原版完全一致） |
| `run_long6k10k_resume.sh` | 幂等续跑：每轮 24 样本数够即跳过；nohup 启动后 SSH 断开不影响，重跑同脚本自动补齐缺轮 |
| `example/score_em_long_6k10k.py` | EM/SM/F1 评分：按日志解析 → 按方法/按数据集聚合 → 输出 `em_report_long_6k10k.md` |
| `logs_long6k10k_maxtok10截断作废/` | 第一轮实验的 6 份日志 + 评分报告（max_tokens=10 截断导致 EM 无效，作废留档不删除） |

## 二、实验设置

- 样本：24 个，来自 CacheBlend 官方数据集文件（多跳 QA / 维基 QA / 对话摘要各 8），**每个都带标准答案**
- 质量指标：
  - **EM**（精确匹配）：归一化后与标准答案完全一致。结果**恒为 0**——Mistral-7B 回答爱带补充语（"Charles university in Prague." ≠ gold "Charles University"），EM 对这个模型-提示词组合失去区分度，仅作记录
  - **SM**（子串匹配，RULER 官方判分风格）：归一化后互为子串即对。能正确给"答对但多带了修饰"的样本记分，是本档 QA 的主要指标
  - **F1**（token 级，仓库 `blend_musique.py` 的原版指标），samsum 摘要任务主要看它
- 速度指标：TTFT（首 token 延迟），每组 **3 轮取中位数**
- 方法配置：CacheBlend 用原型默认（r=0.16）；DAF 用论文配置（r=0.3、第 2~4 层打分、第 4 层选人）——与 1.x 保持一致，**仍不对等**（论文对比时两边同为 r=0.3）

## 三、结果

### 3.1 质量（24 样本，EM 全 0 不列；SM/F1 为跨轮中位数）

| 方法 | SM | F1 | 与全量基线的 SM 差 |
|---|---|---|---|
| 全量重算（标准） | 0.208 | 0.308 | — |
| CacheBlend | 0.167 | 0.283 | −0.04 |
| DAF | 0.083 | 0.224 | −0.12 |

按数据集拆（SM / F1）：

| 数据集 | CacheBlend | DAF | 全量基线 |
|---|---|---|---|
| musique_s（多跳QA，8 个） | 0.000 / 0.082 | 0.000 / 0.090 | 0.250 / 0.279 |
| wikimqa_s（维基QA，8 个） | 0.375 / 0.291 | 0.000 / 0.142 | 0.375 / 0.374 |
| samsum（摘要，8 个） | 0.000 / 0.330 | 0.000 / 0.261 | 0.000 / 0.270 |

绝对值普遍不高是正常的：musique 多跳题对 7B 模型本身就难（全量重算也只有 SM 0.25），CacheBlend 论文在同族任务上的绝对指标同样只有零点几。有意义的信号是**相对排序**。

### 3.2 速度（TTFT 中位数，秒）

| 方法 | TTFT 缓存 | TTFT 全量 | 相对全量加速比 | DAF/CB 时长比 |
|---|---|---|---|---|
| CacheBlend | 0.346 | 1.381 | **3.99×** | 1.73×（慢） |
| DAF | 0.597 | 1.384 | **2.32×** | — |

与 1.1 档（2–4k）对比：CB 0.189→0.346、DAF 0.331→0.597、全量 0.718→1.381——长度翻倍时间约翻倍，**DAF 比 CB 慢的比例几乎不变（1.75×→1.73×）**，三数据集各自也是稳定的 1.73~1.75×。

### 3.3 典型错例

- `wikimqa_s_032`（"George Michael 逝于何处"）：CB 答 "Goring-on-Thames."（对），**DAF 答 "Christmas Day 2016."**（张冠李戴）；全量重算也答对——DAF 的融合近似在这题上直接翻车
- `musique_s_093`：CB 答 "Charles university in Prague."（SM 判对），DAF 答 "University of Prague"（近似但错）
- `musique_s_063`：CB 和 DAF 都答 "Caceres"（全量答 Vila Franca de Xira，gold 是 Lisbon District）——模型连全量重算都错，方法无罪

## 四、分析与结论

1. **论文"9k 持平"未复现**：论文 9k 档 DAF 1.9× vs CB 1.8×（基本持平），我们 6–10k 档 DAF 仍比 CB 慢 73%，且该比例不随长度收窄。原因按可能性排序：
   - 我们是 xformers 后端 + Python 级两路注意力拼装，DAF 的固定开销（clone 混合 KV、两次独立 attention 调用、GQA 头对齐）远大于论文的 Flash-Attention 稠密内核；论文的工程红利（FA kernel）我们没吃到
   - DAF r=0.3 重算量近 CB（0.16）两倍，理论复杂度优势 O(r²Ld²) 在 10k 长度下省的时间抵不过多算的 token
   - DAF/CB 时长比恒定 ≈1.75 说明现在**固定开销主导**，长度得再上一个量级（30k+）才轮到复杂度项说话
2. **质量排序与 1.x 一致**：全量 > CB > DAF。CB 贴全量（SM 差 0.04），DAF 掉队明显（差 0.12）；wikimqa 上 CB 与全量打平而 DAF 全错（SM 0）。样本量 24 仍然偏小，单样本翻转 4 个百分点，趋势参考级。
3. **方法论教训（本档最大收获之一）**：
   - **评测先于跑批**：第一轮 6 轮跑完才发现 max_tokens=10 截断答案、EM 恒 0——白跑一小时。教训：新实验先跑 1 个样本人工检查输出，再放量
   - **指标要匹配模型行为**：Mistral 不遵守"5 词以内"时会带补充语，裸 EM 会系统性低估正确率；SM（RULER 风格子串匹配）是更合适的 QA 指标
   - **作废数据要归档不删除**：`logs_long6k10k_maxtok10截断作废/` 保留了原始证据链
4. **与论文的差距没有缩小**：6–10k 仍是 DAF 的"客场"（论文主场在 30k+）。本档的价值在于把质量评测升级为带标准答案的严格指标，并建立了可复用、可断线续跑的实验管线——为 30k 档铺平了路。

## 五、复现命令

```bash
conda activate cacheblend && cd /media/yunlin/6t/zhenzengtao/daf/zzt
# 1. 生成样本（已生成则幂等跳过）
python experiments/scripts/make_inputs_long_6k10k_official.py
# 2. 跑 6 轮（nohup，可断线；已完整的轮次自动跳过）
nohup bash experiments/scripts/run_long6k10k_resume.sh \
    > experiments/logs/long6k10k/resume.out 2>&1 &
# 3. 评分
python experiments/scripts/score_em_long_6k10k.py \
  --logs experiments/logs/long6k10k/run_cb_long6k10k_r*.log \
         experiments/logs/long6k10k/run_daf_long6k10k_r*.log
```

## 六、后续（按价值排序）

1. **30k 档**（论文主场）：用 RULER 风格合成样本（复现指南 2.3 节），`max_model_len` 16384 已够；预期 DAF/CB 时长比开始 <1——这是判定 DAF 价值的关键实验
2. **r 对齐**：DAF 用 0.16 跑一组，剥离重算比例的混杂因子
3. **扩大样本量**：每数据集 8→30+，把 SM/F1 的误差棒压下去
4. **Flash-Attention 后端**（工程量大）：把 DAF 两路注意力迁到 FA 内核，吃掉论文的工程红利
