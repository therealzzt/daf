# DAF 实现工作总结

> 本文档总结把 CacheBlend 原型改造为 DAF 算法的全过程，依据为《DAF改造计划_v2_校正版.md》。
> 验证环境：WSL2 Ubuntu-22.04 + RTX 4060 8GB + Mistral-7B-Instruct-v0.2-AWQ。

## 一、背景与目标

**任务**：把 CacheBlend 原型仓库改造成 DAF 算法。

**CacheBlend 是什么**：RAG（检索增强生成，即"先检索几篇文档、再让 AI 基于文档答题"）场景下，文档的 KV 缓存（见名词表）可以提前算好存着，但直接拼接会缺"文档之间互相看一眼"的信息。CacheBlend 的办法是：在线请求时只重算约 16% 偏差最大的 token，其余用旧缓存。

**DAF 在 CacheBlend 基础上改了什么**：

1. 选人信号从"只看 1 层"升级为"聚合第 2~4 层的偏差"
2. 选完人之后，重要 token 只和重要 token 互相注意（不再看全文），问题 token 才看全文——这把核心复杂度从 O(r·Ld²) 降到 O(r²·Ld²)

**设计原则**：

- 所有 DAF 逻辑套在 `daf` 开关里，`daf=False` 时原版 CacheBlend 一行不受影响（方便对照实验）
- 正确性优先：用 clone 防止缓存污染、加满 shape 断言，性能优化留给以后

---

## 二、改动前的代码理解（实施前提）

**`cache_fuse_metadata` 白板机制**：一个挂在 `LlamaModel` 上的字典，blend.py 通过一长串内部访问拿到它的**引用**（不是副本），往里写开关；模型 32 层逐层把它当参数传下去，任何层写了什么后面的层立刻可见。两个复位点：`check` 开关用完一次会被引擎自动关回 False（`model_runner.py:913`）；每次 check 开头会清空临时字段（`llama.py`）。

**status 状态码**（每层一个整数，决定这层怎么算）：

| status | 含义 |
|---|---|
| -1 | decode（逐词生成阶段） |
| 0 | 普通全量 prefill |
| 1 | 选人层（原 CacheBlend：第 1 层算偏差、选 16%、砍序列） |
| 2 | 重算层（原 CacheBlend：第 2~31 层只对选中 token 重算并写回旧缓存） |
| **3** | **DAF 新增：打分层（完整 prefill + 只累积偏差分数，不做选择）** |

---

## 三、逐文件改动明细（改动前/后对照）

### 文件 1：`vllm_blend/vllm/model_executor/models/llama.py`（3 处）

**改动 1：白板初始化加 5 个 DAF 字段**

改动前：

```python
self.cache_fuse_metadata = {"check_layers":[1],
                            "check": False,
                            "recomp_ratio":0.16,
                            ...
                            "imp_indices": None,
                            "collect": False}
```

改动后（追加）：

```python
                            # ===== 以下为 DAF 新增字段（daf=False 时全部不生效） =====
                            "daf": False,            # DAF 总开关
                            "sel_layers": [1, 2, 3], # 打分层：论文的"第2~4层"（0起计数）
                            "dev_scores": None,      # 多层累积的偏差分数
                            "doc_imp_indices": None, # 仅文档的重要token位置（不含问题段）
                            "num_imp_tokens": 0}     # 重要token数，深层靠它切两段
```

**原因**：原版的 `imp_indices` 把"重要文档 token + 问题 token"混在一个列表里；DAF 深层要区分两条注意力路径（重要互看 / 问题看全文），必须单存一份只含文档的索引。

**改动 2：每次 check 开头重置 DAF 字段**

改动前（只清旧字段）：

```python
self.cache_fuse_metadata["fake_q"] = None
self.cache_fuse_metadata["attn_bias"] = None
self.cache_fuse_metadata["imp_indices"] = None
...
```

改动后（追加三行）：

```python
# DAF：清掉上一次 check 残留的打分和选人结果
self.cache_fuse_metadata["dev_scores"] = None
self.cache_fuse_metadata["doc_imp_indices"] = None
self.cache_fuse_metadata["num_imp_tokens"] = 0
```

**原因**：不清的话，下一个样本的偏差分数会叠在上一个样本的分数上，选人就全错了。

**改动 3：层调度器加 daf 分支**

改动前：

```python
if self.cache_fuse_metadata["check"]:
    if i in self.cache_fuse_metadata["check_layers"]:
        temp_status = 1      # 第 1 层选人
    elif i > self.cache_fuse_metadata["check_layers"][0]:
        temp_status = 2      # 第 2 层起重算
```

改动后：

```python
if self.cache_fuse_metadata["check"]:
    if self.cache_fuse_metadata["daf"]:
        # 第 0 层: status=0 普通全算
        # 第 1~3 层: status=3 全算 + 累积偏差分数
        # 第 4 层: status=1 用累计分数选人，进入压缩模式
        # 第 5 层起: status=2 压缩模式，每层跑 DAF 两路注意力
        sel_layers = self.cache_fuse_metadata["sel_layers"]
        select_layer = max(sel_layers) + 1    # 选择层 = 最后一个打分层 + 1
        assert select_layer < len(self.layers)
        if i in sel_layers:
            temp_status = 3
        elif i == select_layer:
            temp_status = 1
        elif i > select_layer:
            temp_status = 2
        else:
            temp_status = 0
    else:
        # 原版 CacheBlend 排班原样保留
        ...
```

**原因**：论文要求"先聚合多层信号再选择"，所以打分层（1~3）和选择层（4）必须错开；`daf=False` 走原路，保证可对照。

### 文件 2：`vllm_blend/vllm/attention/backends/xformers.py`（5 处）

**改动 1：旧 KV 的可见性**

改动前：

```python
if status in [1, 2]:
    key_old = old_kv[0].view(...)
    value_old = old_kv[1].view(...)
```

改动后：

```python
if status in [1, 2, 3]:
    # status=3 只需要旧 V 算偏差，不需要旧 K（K 带 RoPE 旋转，对齐麻烦）
    value_old = old_kv[1].view(...)
if status in [1, 2]:
    key_old = old_kv[0].view(...)
```

**原因**：不改的话 status=3 里用 `value_old` 会直接报"变量未定义"。

**改动 2：新增 status=3 打分层（全新代码）**

```python
if status == 3:
    suffix_len = cache_fuse_metadata['suffix_len']
    doc_len = value.shape[0] - suffix_len
    # temp_diff[i] = 第 i 个文档 token 的 ||V新 - V旧||²
    # 问题段没有旧缓存、无条件保留，所以只对文档部分打分
    temp_diff = torch.sum((value[:doc_len]-value_old[:doc_len])**2, dim=[1,2])
    temp_diff = temp_diff.float()   # float32 累加，减少 fp16 舍入误差
    if cache_fuse_metadata["dev_scores"] is None:
        cache_fuse_metadata["dev_scores"] = torch.zeros_like(temp_diff)
    cache_fuse_metadata["dev_scores"] += temp_diff
    # 不 topk、不裁剪、不改 positions——本层照常走完整因果注意力
```

**原因**：DAF 核心改动之一——"打分"和"选择"分离。前三层各自累积证据，第 4 层才拍板。偏差大的 token = 它的笔记受文档拼接影响大 = 最需要重算。

**改动 3：status=1 选人块加 daf 分支**

改动前（原版 CacheBlend，本层临时算偏差、立刻选）：

```python
topk_num = int((total_len-last_len)*cache_fuse_metadata["recomp_ratio"])
temp_diff = torch.sum((value[:-last_len]-value_old[:-last_len])**2, dim=[1,2])
top_indices = torch.topk(temp_diff, k=topk_num).indices
top_indices, _ = torch.sort(top_indices)
top_indices = torch.cat([top_indices, 末尾问题段])
query = query[top_indices]
cache_fuse_metadata["imp_indices"] = top_indices
```

改动后（daf 分支用**累积分数**、只从文档部分选）：

```python
if cache_fuse_metadata["daf"]:
    scores = cache_fuse_metadata["dev_scores"] / len(sel_layers)  # 多层平均
    topk_num = int(doc_len * cache_fuse_metadata["recomp_ratio"])  # r=0.3
    topk_num = max(1, min(doc_len, topk_num))
    doc_imp_indices = torch.sort(torch.topk(scores, k=topk_num).indices).values
    suffix_indices = torch.arange(doc_len, total_len, ...)  # 问题段无条件保留
    cache_fuse_metadata["doc_imp_indices"] = doc_imp_indices
    cache_fuse_metadata["num_imp_tokens"] = doc_imp_indices.numel()
    cache_fuse_metadata["imp_indices"] = torch.cat([doc_imp_indices, suffix_indices])
else:
    # 原版逻辑原样保留
```

**原因**：① 分数来自多层累积而非单层（论文要求）；② 问题 token 不参与 top-k 排名、无条件保留；③ 额外存 `doc_imp_indices` 供深层区分两路。

**改动 4：KV 融合块加 daf 分支（status 1/2 共用）**

改动前（原版只有 status=2，且**原地修改**旧缓存）：

```python
if status in [2]:
    key_old[imp_indices] = key      # 直接改 old_kv
    value_old[imp_indices] = value
    key = key_old
    value = value_old
```

改动后：

```python
if status in [1, 2] and cache_fuse_metadata["daf"]:
    # clone 出副本再改：防止污染可跨请求复用的离线缓存
    key_hybrid = key_old.clone()
    value_hybrid = value_old.clone()
    if status == 1:    # 选择层：输入还是完整序列，按原始位置取新鲜值
        key_hybrid[doc_imp] = key[doc_imp]
        key_hybrid[doc_len:] = key[doc_len:]     # 问题段也用新鲜值
        query = query[imp_indices]               # Q 砍成压缩序列
    else:              # status=2：输入已是压缩序列 [重要 | 问题]，按段切
        key_hybrid[doc_imp] = key[:num_imp]
        key_hybrid[doc_len:] = key[num_imp:]
    key = key_hybrid      # 后续写 GPU KV cache 的就是这份混合版
    value = value_hybrid
elif status in [2]:
    # 原版 CacheBlend 原样保留
```

**原因**：混合 KV 的三段来源——未选中文档用旧缓存（省钱的关键）、选中文档和问题段用本层新鲜值。clone 是正确性取舍：原地改会让"问题 A 的缓存"污染"问题 B"。

**改动 5：两个新 helper + prefill 调用点分支（全新代码）**

调用点：

```python
# 改动前：一律走
out = self._run_memory_efficient_xformers_forward(query, key, value, ...)
# 改动后：
if status in [1, 2] and cache_fuse_metadata["daf"]:
    out = self._run_daf_attention(query, key, value, cache_fuse_metadata)
else:
    out = self._run_memory_efficient_xformers_forward(...)
```

`_run_daf_attention` 的数据流（DAF 论文三步）：

```python
q_imp = query[:num_imp]          # 压缩 Q 切成两段
q_suf = query[num_imp:]
# A. 重要 token 互相自注意力：无掩码（双向可见），复杂度 O((r·Ld)²)
out_imp = self._dense_attention(q_imp, key[doc_imp], value[doc_imp], attn_bias=None)
# B. 问题 token 对全文混合 KV 做因果注意力（不能偷看后面的问题 token）
out_suf = self._dense_attention(q_suf, key, value,
                                attn_bias=LowerTriangularFromBottomRightMask())
# C. State Fusion：按 [重要 | 问题] 顺序拼接，与 imp_indices 顺序一致
out = torch.cat([out_imp, out_suf], dim=0)
```

`_dense_attention` 负责脏活：GQA 头数对齐（K/V 的 8 个头逻辑复制成 32 个对齐 Q）、补 batch 维、调 xformers 算子、还原形状。并配了全套断言（形状、NaN/Inf 检查）。

**原因**：两路注意力是 DAF 的灵魂——重要 token 互看把二次复杂度从 r·Ld² 压到 r²·Ld²（r=0.3 时系数 0.3→0.09）；抽成 helper 是因为 status=1 和 status=2 用同一套逻辑，不抽就要复制两大段。

### 文件 3：`example/blend.py`（2 处）

**改动 1：层数不写死**

```python
# 前：num_layer = 32
# 后：num_layer = len(llm.llm_engine...model.model.layers)   # 动态数出来，换模型也能用
```

**改动 2：打开 DAF（阶段二 check 之前新增）**

```python
cache_fuse_metadata["daf"] = True                  # 改 False 即退回原版 CacheBlend
cache_fuse_metadata["sel_layers"] = [1, 2, 3]      # 论文的第 2~4 层
cache_fuse_metadata["recomp_ratio"] = 0.3          # 论文实验值，可调
cache_fuse_metadata["check_layers"] = [4]          # 同步旧字段，防配置打架
```

---

## 四、关键设计决策与工程假设

写实验记录时要区分：

**论文明确支持**：多层信号聚合（第 2~4 层）、重要 token 互相自注意力、问题在尾部看全文、两路输出拼接、复杂度 r²·Ld²、r=0.3 实验配置。

**我们的工程假设（论文没写死）**：

1. **V-only 偏差**：论文写 "KV deviation"，我们沿用原型的只比 V（K 带 RoPE 旋转，新旧对齐麻烦，留作升级）
2. **算术平均聚合**：论文只说 aggregate 没给公式；层数固定时求和与求平均对 top-k 排名等价
3. **第 4 层首次压缩**：论文没说打分层之后第几层必须开始选
4. **clone 混合 KV**：正确性原型的取舍，正式版应换成按请求分配的缓冲区

**RoPE 为什么原样复用**：旧 K 存的是旋转前的"生"值，每层用时按 `org_pos` 现场补旋转一次；status=3 不碰旧 K，每个样本 blend.py 重建 `old_kvs`，不存在重复旋转。

---

## 五、验证过程与结果

测试脚本 `wsl_setup/daf_smoke.py`（只跑 1 号样本，`DAF=0/1` 环境变量切换），WSL 内运行。

| 检查项 | DAF=0（原版回归） | DAF=1（新实现） |
|---|---|---|
| 跑通无报错 | ✅ | ✅（全部断言通过，无 NaN） |
| cached 生成 | Carnegie Mellon University | Carnegie Mellon University |
| full 全量对照 | Carnegie Mellon University | Carnegie Mellon University |
| 逐字一致 | ✅ | ✅ |
| 选人数量 | 1193 | 1134 重要 + 59 问题 = 1193 ✓ |

数字验证：总长度 3839，文档 3780 × 0.3 = 1134（= `num_imp_tokens`），加问题段 59 = 1193（= `imp_indices` 长度）——说明多层打分、仅文档选人、问题段保留全部按预期工作。

**DAF 这次 81s vs CacheBlend 66s，略慢，符合预期**：短上下文（3.8k）下 DAF 优势不显现（论文里 6k 时两者持平、30k+ 才拉开差距）；且我们用的 xformers 后端而非 Flash-Attention，还有 clone 开销。

---

## 六、当前状态与后续路线

**已完成**：计划 v2 的阶段 1~4（修基线、多层打分、status=1 两路 DAF、status=2 两路 DAF），即"完整 DAF"。

**怎么用**：

- 完整实验：`cd ~/CacheBlend && HF_ENDPOINT=https://hf-mirror.com python example/blend.py`
- 对照实验：blend.py 里 `daf = True → False`
- 快速冒烟：`bash wsl_setup/run_smoke.sh 0` 或 `1`

**后续**（按优先级）：commit 检查点 → 10 样本全量跑 → 长上下文 TTFT 对比 → 可选升级（K+V 偏差、去 clone、归一化聚合、Flash-Attention）。

---

## 七、名词表

- **token**：模型处理文字的最小单位，大致是"半个词到一个词"。代码里就是一个整数编号。
- **prefill（读题阶段）**：把整段输入一次性算一遍，慢。你等 AI 开口的"转圈时间"主要花在这。
- **decode（答题阶段）**：一个词一个词往外蹦，每步只处理 1 个新 token，快。
- **TTFT（Time To First Token）**：从提交请求到生成第一个字的等待时间，本项目的核心优化指标。
- **K / Q / V（Key/Query/Value）**：注意力机制的三个向量。Q="我想找什么"，K="我是什么的索引"，V="我的实际内容"。新 token 拿 Q 去和所有旧 token 的 K 匹配，按匹配度加权取出 V。
- **KV cache（KV 缓存）**：把每个 token 在每层算出的 K/V 存显存里，下次不用重算。CacheBlend/DAF 省的就是重算它的时间。
- **RoPE（Rotary Position Embedding，旋转位置编码）**：告诉模型 token 先后顺序的方法——按位置编号给 Q/K 做一次数学旋转。本项目旧 K 存"旋转前"的，用时按原位置补转。
- **GQA（Grouped-Query Attention，分组查询注意力）**：Mistral-7B 的 Q 有 32 个头、K/V 只有 8 个，4 个 Q 头共用 1 组 K/V，省显存。代码里要把 K/V 逻辑复制 4 倍才能和 Q 对齐计算。
- **因果掩码（causal mask）**：生成模型的规矩——每个位置只能看自己和之前的 token，不能偷看未来。DAF 的问题路径用它；重要 token 路径**不用**（互相全可见）。
- **LowerTriangularFromBottomRightMask**：xformers 提供的一种因果掩码，适用于"Q 是完整序列尾部连续一段"的场景——正好是"问题在文档后面"的形状。
- **hybrid KV（混合 KV）**：DAF/CacheBlend 的核心产物——一份缓存里，大部分位置是离线旧值，少数重要位置是本次重算的新值。
- **压缩序列（compact hidden states）**：选人之后，不再让全部 3839 个 token 逐层计算，只留 1193 个（重要文档+问题）往下走，这是省时间的直接来源。
- **state fusion（状态融合）**：把"重要 token 那一路"和"问题那一路"的注意力输出按 [重要 | 问题] 顺序拼成一份，作为下一层的输入。
- **top-k**：从 N 个分数里挑出最大的 k 个。这里用来选"偏差最大的 30% 文档 token"。
- **suffix（后缀）/ suffix_len**：输入末尾的"问题+结束标记"段及其长度（本样本 59 个 token），无条件保留、不参与打分。
- **residual（残差）**：Transformer 里绕过注意力层的那条"直通通道"，序列砍短时它也要跟着砍（llama.py:265）。
- **status**：每层一个的工作模式码（-1/0/1/2/3），由 llama.py 排班、xformers.py 执行。
- **slot_mapping / paged cache**：vLLM 管理显存 KV 的机制——显存切成固定大小的"块"，slot_mapping 记录每个 token 的 KV 该写进哪个格子。
- **xformers**：Meta 的注意力计算库，本项目用它做实际注意力运算（因为 FlashAttention 没装）。
- **AWQ**：4bit 权重量化方法，把模型从约 15GB 压到约 4GB，8GB 显卡才装得下。
- **enforce_eager**：关闭 CUDA Graph（一种把计算图录制下来加速的技术）——它要多占几百 MB 显存，显存紧张所以关。
- **clone**：张量深拷贝。DAF 里用它复制旧 KV 再改，避免污染可复用的离线缓存。
- **dev_scores**：DAF 新增字段，每个文档 token 一个分数，记录它在第 1~3 层累积的 V 偏差。
