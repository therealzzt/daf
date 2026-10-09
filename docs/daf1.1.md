# DAF 1.1 实验总结：bf16 非量化模型复现

> 日期：2026-09-14
> **路径变更（2026-09-16 整理）**：本实验的日志已从仓库根目录迁入 `experiments/logs/`：
> `run_cb.log`/`run_daf.log` → `experiments/logs/short_2k4k_awq/`，
> `run_cb_bf16.log`/`run_daf_bf16.log` → `experiments/logs/short_2k4k_bf16/`。总索引见 `experiments/README.md`。
> 环境：Ubuntu 24.04 服务器 + RTX 3090 Ti 24GB（4 卡，实验用 GPU 0）+ vllm 0.4.1（本仓库 vllm_blend）+ torch 2.2.1/cu121 + xformers 0.0.25 + transformers 4.41.2
> 模型：**Mistral-7B-Instruct-v0.2 官方 bf16 非量化版**（~15GB 权重）
> 脚本：`example/blend.py`（模型加载段已按《服务器复现指南》选项 A 修改），`inputs/1.json ~ 10.json`，temperature=0，每次生成 10 token
> 前作：DAF 1.0 配置（AWQ 4bit + RTX 4060）见 `docs/daf_experiment_10samples.md`、实现见 `docs/daf1.0.md`
> 日志：`run_cb_bf16.log`（DAF=0）、`run_daf_bf16.log`（DAF=1）；AWQ 版日志 `run_cb.log`、`run_daf.log`

---

## 一、改进了什么

相比 DAF 1.0 的实验配置，本次（记作 1.1）做了以下升级，**算法代码零改动**：

| 项目 | 1.0（前作） | 1.1（本次） | 动机 |
|---|---|---|---|
| 模型精度 | AWQ 4bit 量化（~4GB） | **bf16 非量化（~15GB）** | 消除权重量化引入的额外误差源；与 CacheBlend 论文及 DAF 论文的实验配置（bf16）一致。量化版是 8GB 显存开发机的妥协 |
| 硬件 | RTX 4060 8GB ×1（WSL2 桌面机） | RTX 3090 Ti 24GB ×4 无桌面服务器（用 1 块） | 装得下 bf16 模型；无桌面程序抢 GPU，TTFT 测量更干净 |
| `gpu_memory_utilization` | 0.9 | 0.85 | 按复现指南选项 A 的建议值 |
| 复现方式 | 开发机结果转录 | **全新机器从零复现**（含编译 vllm_blend） | AWQ 版在新环境重跑，10 个答案与开发机基线逐字一致，证明结果可复现 |

顺带修复了新服务器上的三个环境问题（不影响算法，记录备查）：

1. `conda install cuda-toolkit=12.1` 实际解析出 CUDA 13.3 组件 → 改用 `nvidia/label/cuda-12.1.1` 显式钉住 nvcc/cudart/cccl 12.1，否则 vllm 0.4.1 编译不过
2. `vllm-nccl-cu12` 需从 GitHub 下载 libnccl（服务器连不通）→ 经 ghfast 镜像下载 278MB 放到 `~/.config/vllm/nccl/cu12/` 后安装通过
3. conda-forge 的 `ld` 与 Ubuntu 24.04 系统 glibc 冲突导致 cmake 检测 CUDA 编译器失败 → 卸载 conda 的 gcc/g++/binutils，用系统 g++ 11.5 编译

## 二、实验结果

### 2.1 质量对比（cached 答案 vs 同轮全量重算答案，四组全列）

| 样本 | AWQ 全量（标准） | AWQ CB | AWQ DAF | bf16 全量（标准） | bf16 CB | bf16 DAF |
|---|---|---|---|---|---|---|
| 1 | Carnegie Mellon University | ✅ | ✅ | Carnegie Mellon University. | ✅ | ✅ |
| 2 | The Soviet Story was made first. | ❌ The Soviet Story. (born 200 | ❌ Hell Up In Harlem (1973 | The Soviet Story (older) | ❌ was made later | ❌ Hell Up In Harlem (1973 |
| 3 | Cipriano Castro was born first. | ✅ | ✅ | 同 | ✅ | ✅ |
| 4 | Jugband Blues came out first. | ✅ | ✅（多引号） | 同 | ✅ | ✅（多引号） |
| 5 | Yakshya Malla's birthplace | ❌ Bulacan | ❌ Ducal house of Bourbon. | Bulacan, Philippines. | ✅ | ❌ Duke of Bourbon |
| 6 | Arthur Acheson, 1st Earl of | ✅ | ✅ | 同 | ✅ | ✅ |
| 7 | At home in Goring-on-Thames | ✅ Goring-on-Thames. | ❌ Died in bed. | 同 | ✅ Goring-on-Thames. | ❌ Died in bed. (Note:… |
| 8 | Yes, both are French. | ❌ German | ✅ **French** | Yes, both are French. | ❌ German | ❌ German |
| 9 | Wooden Crosses (1932) | ✅ | ✅ | 同 | ✅ | ✅ |
| 10 | Above Rubies came out earlier. | ✅ | ❌ The Magic Aster | Above Rubies came out earlier. | ✅ | ✅ |
| **一致率** | — | **8/10** | **6/10** | — | **8/10** | **6/10** |

（AWQ 两列与开发机基线逐字相同，见 `docs/daf_experiment_10samples.md`；bf16 全量列中样本 2、5、7、10 的标准答案本身也与 AWQ 版不同，是模型精度变化导致的正常漂移。）

**观察**：

1. **量化误差真实存在**：样本 5 上 AWQ 版全量重算答错（Yakshya Malla 出生地答成 Bulacan），bf16 版全量重算答对——少一个误差源后基线本身变强了。
2. **一致率跨配置惊人地稳定**：两种模型精度下都是 CB 8/10、DAF 6/10。DAF 一致率低于 CB 的现象**不是量化噪声造成的**，更可能来自 DAF 自身的工程假设（V-only 打分、多层算术平均、重要 token 无掩码互看，见 `docs/daf1.0.md` 第四节）。
3. **样本 8 提示小样本结论的脆弱性**：四种配置里只有 AWQ+DAF 一种答对（French），bf16 下全量/CB/DAF 全部答错（German）——这更像是 Mistral-7B 对该知识点本身判断不稳，而非某种方法的质量优势。10 个样本、每个 10 token 的对比，单个样本就能翻转 10 个百分点，不宜过度解读。
4. bf16 版 DAF 的样本 7 错误输出带上了推理痕迹（"Died in bed. (Note: The given"），说明它开始基于上下文做解释，但结论仍错。

### 2.2 速度对比（TTFT，秒；越短越好）

| 配置 | CB | DAF | 全量重算 | CB 加速比 | DAF 加速比 | DAF/CB |
|---|---|---|---|---|---|---|
| AWQ 4bit（RTX 4060→基线） | 19.3 | 26.4 | 40.4/54.0 | ~2.3× | ~1.6× | 1.37×（慢） |
| AWQ 4bit（本机 3090 Ti） | 0.222 | 0.356 | 0.723 | 3.26× | 2.03× | 1.60×（慢） |
| **bf16（本机 3090 Ti）** | **0.189** | **0.331** | **0.718** | **3.81×** | **2.17×** | **1.75×（慢）** |

（加速比 = 全量 TTFT 均值 ÷ 该方法 TTFT 均值；均为单轮干净环境数据。）

**观察**：

1. bf16 权重比 AWQ 大近 4 倍，但 CB/DAF 的 TTFT 反而各快了约 15%/7%（0.222→0.189、0.356→0.331）——本批样本 2~4k token，计算量小，端到端时间里有相当比例的固定开销，模型大小的影响被淹没了；不宜据此推断长上下文下的规律。
2. DAF 比 CB 慢 60~75% 的短上下文劣势在所有配置下稳定复现，与论文"6k 时两者持平、30k+ 才反超"的论断定性一致：**本批样本不在 DAF 的主场**。
3. 融合相对全量重算的加速比（CB ~3.8×、DAF ~2.2×）比论文 6k 档位的 1.8× 偏高，原因是样本偏短且固定开销占比大，不能直接与论文数字对比。

### 2.3 功能验证

- 10 样本 × 两组（DAF=0/1）全部跑完，无 AssertionError、无 NaN
- 日志确认 `Using XFormers backend`（未装 flash-attn，DAF 改动所在后端生效，无静默失效）

## 三、与论文（DAF, arXiv:2607.21599v1）结果的差别

| 维度 | 论文实验 | 本实验（1.1） | 差别说明 |
|---|---|---|---|
| 模型 | Qwen2.5-7B，bf16 | Mistral-7B-Instruct-v0.2，bf16 | CacheBlend 原型（vllm 0.4.1）写死了 Mistral 指令模板 token，换家族需改 blend.py 的模板编号并验证架构兼容；同规模 7B，定性可比 |
| 硬件 | 单卡 A100 80GB | 单卡 RTX 3090 Ti 24GB | 同代架构（Ampere），对本实验结论无实质影响 |
| 重算比例 | CB 与 DAF **同为 r=0.3** | CB 用原型默认 0.16，DAF 用 0.3 | **配置不对等**：DAF 重算量近乎 CB 两倍。论文的公平对比下 DAF 仍有精度优势；我们的对比里这一优势被放大的重算量抵消了一部分 |
| 上下文长度 | 6k / 9k / 30k / 136k | 2~4k | 论文里 DAF 相对 CB 的速度优势区间在 30k+，本批样本完全够不着；论文的精度优势（RULER、LongBenchV2 长上下文）同样只在长样本上体现 |
| 质量指标 | EM（QA）/ Rouge-L（摘要），几十上百样本 | 与全量重算的逐字一致率，10 样本 × 10 token | 我们的指标只能做功能冒烟，无统计显著性；样本 8 显示单样本即可翻转结论 |
| 精度结论 | 长上下文 DAF≈全量 ≫ CB；短上下文 DAF 略优于 CB | **DAF 6/10 < CB 8/10**（两种模型精度下一致） | 未能复现论文的精度排序。原因按可能性排序：① 样本太短太少；② V-only 偏差打分（论文写 KV deviation）；③ 多层算术平均（论文未给公式）；④ 重要 token 双向无掩码互看（论文未明说因果与否）；⑤ 两方法 r 不对等 |
| 速度结论 | 6k 持平（1.8×），136k DAF 达 CB 2 倍、vLLM 5.6× | 2~4k：DAF 比 CB 慢 1.75×；两者均远快于全量 | 与论文趋势定性一致（DAF 优势随长度增长），但绝对差距方向相反，因为长度远未进入 DAF 优势区间 |
| 后端 | Flash-Attention 稠密内核 | xformers（未装 flash-attn） | 论文的核心工程红利（FA 内核）没吃到；DAF 两路注意力在 xformers 下是 Python 级拼装，固定开销偏大，短上下文下尤其吃亏 |

**一句话总结**：1.1 把模型升级到了与论文同精度的 bf16，确认了两个跨配置稳定的结论——"融合远快于全量重算"与"短上下文 DAF 慢于 CB"；但论文的主打结论（长上下文 DAF 又快又准）在本实验的 2~4k 短样本上既无速度优势也未体现精度优势，要验证它必须补长上下文（30k+）样本与 EM/F1 正式评测。

## 四、后续建议（按价值排序）

1. **长上下文实验**：构造/选取 30k+ token 样本（复现指南 2.3 节的 RULER 生成方案），`max_model_len` 拉到 32768 以上，验证 DAF 的 r² 复杂度优势是否如论文般兑现——这是论文与当前实验差距最大的一环
2. **对齐 r 做公平对比**：DAF 也用 0.16 跑一组（或 CB 用 0.3），把"配置不对等"这个混杂因子剥离
3. **正式质量评测**：用仓库自带的 `blend_musique.py` / `blend_wikimqa.py`，几十上百样本 + EM/F1，替换现在的 10 样本目视对比
4. **消融工程假设**（`docs/daf1.0.md` 第六节清单）：K+V 打分、因果掩码、recomp_ratio 扫描
5. **Flash-Attention 后端**（可选）：安装 flash-attn 并把 DAF 两路注意力迁到 FA 内核，吃掉论文的工程红利——注意本仓库 xformers 后端的全部改动需要移植
