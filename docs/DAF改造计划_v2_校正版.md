# DAF 改造计划 v2（基于 CacheBlend 原型仓库，按 DAF 论文校正）

> 目标：先在 CacheBlend 的 `xformers` 原型上正确实现 DAF 的算法语义，再做性能优化。
>
> 重要说明：DAF 论文只有 2 页，给出了核心算法和复杂度，但没有公开到逐行代码级别的实现细节。因此本文把内容分成两类：
>
> - **论文明确支持**：可以直接按论文实现。
> - **工程实现假设**：论文没有写死，需要我们自己选择，并在实验记录中注明。

---

## 一、先修正当前计划里的几个关键问题

### 1. 文件路径有一个拼写错误

应为：

```text
vllm_blend/vllm/attention/backends/xformers.py
```

不是：

```text
vllm_blendllm/attention/backends/xformers.py
```

另外统一使用文件名 `xformers.py`，不要写成 `xform.py`。

---

### 2. `example/blend.py` 里现有的 `suffix_len` 计算必须先修

当前 CacheBlend 原型里有：

```python
last_len = len([q_ids + s_end])
```

这个表达式的结果几乎永远是 `1`，因为外层 `[...]` 创建的是一个只含一个元素的 Python 列表。

应改为：

```python
last_len = len(q_ids + s_end)
```

这里的 `suffix_len` 表示“尾部问题段的 token 数量”。  
如果它错误地等于 1，后面的“文档 token”和“问题 token”分界都会错，DAF 的选 token、问题注意力和形状计算都会一起出错。

**建议把这个修复单独做一次 git commit，然后重新跑 CacheBlend 基线。**

---

### 3. 不要把“CacheBlend 只看第 2 层”写成 CacheBlend 算法本身的事实

当前 GitHub 原型默认：

```python
"check_layers": [1]
```

所以这个**原型代码**确实是在 0-indexed 的第 1 层，也就是人类习惯说的“第 2 层”，做一次选择，然后深层沿用所选 token。

但是 CacheBlend 原论文还描述过逐层的 gradual filtering（逐层筛选）机制。

因此建议改写成：

> “当前 CacheBlend GitHub 原型默认只在第 2 个 Transformer 层执行一次选择。DAF 论文则提出聚合多层（通常第 2～4 层）的 KV-deviation 信号。这里的改造针对当前原型代码，而不是声称所有 CacheBlend 实现都只使用单层。”

---

### 4. DAF 论文写的是 **KV deviation**，当前原型代码实际只算 **V deviation**

当前 `xformers.py` 使用：

```python
temp_diff = torch.sum(
    (value[:-last_len] - value_old[:-last_len]) ** 2,
    dim=[1, 2],
)
```

这只比较了 Value，也就是 **V**。

而 DAF 论文文字写的是多层 **KV deviation**。

因此有两个实现路线：

#### 路线 A：先保持 CacheBlend 原型的 V-only 评分

优点：

- 改动小；
- 容易先跑通；
- 不需要额外处理 K 的 RoPE 对齐。

缺点：

- 不能声称这是论文精确的 “KV deviation”。

建议第一版采用 A，并在实验记录中写：

> “Important-token selection uses the V-deviation implementation inherited from the CacheBlend prototype.”

#### 路线 B：之后升级为 K+V deviation

例如概念上：

```python
k_diff = ((key_doc - key_old_doc) ** 2).sum(dim=[1, 2])
v_diff = ((value_doc - value_old_doc) ** 2).sum(dim=[1, 2])
temp_diff = k_diff + v_diff
```

但是 K 带有 RoPE（Rotary Position Embedding，旋转位置编码），新旧 K 必须处于相同的位置编码状态才能比较，所以不要第一版就贸然改。

---

### 5. `r = 0.3` 是论文实验配置，不是 DAF 永远固定的算法常数

论文实验把 important-token fraction 设为：

```text
r = 0.3
```

但论文的 Limitations 同时明确说，`r` 需要按任务调整，用来平衡准确率和速度。

所以代码注释建议改成：

```python
cache_fuse_metadata["recomp_ratio"] = 0.3
# 论文实验使用 0.3；它不是不可修改的算法常数
```

---

### 6. “深层继续 CacheBlend”只能作为调试版本，不能作为最终 DAF

DAF 的关键复杂度是：

```text
CacheBlend: O(r * Ld^2 + Ld * Lq)

DAF:        O(r^2 * Ld^2 + Lq * (Ld + Lq))
```

其中：

- `Ld`：所有文档 token 数；
- `Lq`：问题 token 数；
- `r`：重要 token 比例。

如果深层仍让重要 token 对完整文档做注意力，那么深层还是 `r * Ld^2`，DAF 最核心的 `r^2 * Ld^2` 优势就没有真正实现。

所以：

- **混合版**：可以作为排错阶段；
- **最终版**：`status=2` 也必须执行 DAF 的“两路注意力 + 状态融合”。

---

## 二、第 0 步：建立正确基线

建议分成两个 checkpoint。

### Checkpoint A：保存当前能运行的原始版本

```bash
git add -A
git commit -m "checkpoint original CacheBlend prototype"
```

### 修复 `suffix_len`

```python
last_len = len(q_ids + s_end)
```

并把：

```python
num_layer = 32
```

尽量改成动态获取：

```python
llm_layers = (
    llm.llm_engine.model_executor.driver_worker
    .model_runner.model.model.layers
)
num_layer = len(llm_layers)
```

这样代码不会被固定在 32 层模型上。

### Checkpoint B：保存“修正后的 CacheBlend 基线”

确认 `daf=False` 时可以正常运行后：

```bash
git add -A
git commit -m "fix CacheBlend baseline before DAF"
```

之后所有 DAF 回归测试，都应该和 **Checkpoint B** 比，而不是和带 `suffix_len` 错误的版本比。

---

# 三、第 1 步：扩展 `cache_fuse_metadata`

位置：

```text
vllm_blend/vllm/model_executor/models/llama.py
```

在 `LlamaModel.__init__()` 的：

```python
self.cache_fuse_metadata = {...}
```

中加入：

```python
"daf": False,

# DAF 论文说通常聚合第 2～4 层信号。
# Python 从 0 开始编号，所以对应 1、2、3。
"sel_layers": [1, 2, 3],

# 多层 deviation 累积分数。
"dev_scores": None,

# 只包含文档中的重要 token 的“原始完整序列位置”。
"doc_imp_indices": None,

# 下游兼容索引：
# [文档重要 token 的原始位置, 问题 suffix 的原始位置]
"imp_indices": None,

# 方便深层把 compact hidden states 切成
# [important docs | question suffix] 两部分。
"num_imp_tokens": 0,
```

每个新请求开始时至少重置：

```python
self.cache_fuse_metadata["dev_scores"] = None
self.cache_fuse_metadata["doc_imp_indices"] = None
self.cache_fuse_metadata["imp_indices"] = None
self.cache_fuse_metadata["num_imp_tokens"] = 0
```

## 为什么要新增 `doc_imp_indices`

原 CacheBlend 的 `imp_indices` 同时包含：

```text
重要文档 token + 问题 token
```

但是 DAF 深层需要明确区分两条注意力路径：

```text
重要文档 token -> 只和重要文档 token 互相注意
问题 token     -> 对完整文档 + 问题做注意力
```

所以只保存一个混合索引不够。

---

# 四、第 2 步：修改 layer scheduler（层调度）

这里的 `status` 可以理解成“当前 Transformer 层应该采用哪一种计算模式”。

建议定义：

```text
status = 0 : 普通完整 prefill
status = 3 : 完整 prefill + 收集本层 deviation
status = 1 : 完成多层聚合、选择 important tokens，并进入 DAF compact 模式
status = 2 : 后续 DAF compact 层
status = -1: decode
```

## 推荐调度

如果：

```python
sel_layers = [1, 2, 3]
```

那么第一版可以采用：

| 0-indexed 层 | status | 行为 |
|---|---:|---|
| 0 | 0 | 普通完整 prefill |
| 1, 2, 3 | 3 | 完整 prefill，并累积 deviation |
| 4 | 1 | 根据累计分数选 important tokens，执行 DAF |
| > 4 | 2 | 继续执行 DAF |

伪代码：

```python
if self.cache_fuse_metadata["check"]:
    if self.cache_fuse_metadata["daf"]:
        sel_layers = self.cache_fuse_metadata["sel_layers"]
        select_layer = max(sel_layers) + 1

        if i in sel_layers:
            temp_status = 3
        elif i == select_layer:
            temp_status = 1
        elif i > select_layer:
            temp_status = 2
        else:
            temp_status = 0
    else:
        # 完整保留原 CacheBlend 行为
        ...
```

## 这里必须注明：这是“工程实现假设”

论文明确说：

> 聚合多个层的信号，通常使用第 2～4 层。

但论文没有逐行说明：

> “第 2～4 层全部 full attention，然后必须在第 5 层才第一次 compact。”

因此 `[1,2,3] -> 第 4 个索引层开始选择` 是一个合理的实现方案，但不是论文明确写死的细节。

---

# 五、第 3 步：加入 `status=3` —— 多层 deviation 收集

位置：

```text
vllm_blend/vllm/attention/backends/xformers.py
```

## 先处理一个容易漏掉的问题

当前代码只在：

```python
if status in [1, 2]:
    key_old = ...
    value_old = ...
```

时创建 `value_old`。

如果新写：

```python
if status == 3:
    temp_diff = value - value_old
```

但没有修改前面的条件，程序会直接报：

```text
value_old is not defined
```

因此 V-only 第一版至少要让 status 3 可以读取旧 V，例如：

```python
if status in [1, 2, 3]:
    value_old = old_kv[1].view(
        -1, self.num_kv_heads, self.head_size
    )

if status in [1, 2]:
    key_old = old_kv[0].view(
        -1, self.num_kv_heads, self.head_size
    )
```

如果以后实现 K+V deviation，再让 status 3 同时获得正确 RoPE 对齐后的 `key_old`。

---

## status=3 的 V-only 基线实现

```python
if status == 3:
    suffix_len = cache_fuse_metadata["suffix_len"]
    doc_len = value.shape[0] - suffix_len

    temp_diff = torch.sum(
        (value[:doc_len] - value_old[:doc_len]) ** 2,
        dim=[1, 2],
    )

    # 推荐用 float32 累加，减少半精度累加误差。
    temp_diff = temp_diff.float()

    if cache_fuse_metadata["dev_scores"] is None:
        cache_fuse_metadata["dev_scores"] = torch.zeros_like(temp_diff)

    cache_fuse_metadata["dev_scores"] += temp_diff
```

然后 `status=3` 仍然走普通完整 causal attention。

这里不要：

- top-k；
- 裁剪 hidden states；
- 改 `positions`；
- 设置 `imp_indices`。

## 为什么

status=3 只负责“收集证据”。

“causal attention” 指生成模型的因果注意力：当前位置只能读取自己和前面的 token，不能读取未来 token。

---

# 六、第 4 步：`status=1` 完成选择并执行真正 DAF

这是最核心的一步。

---

## 4.1 多层分数聚合

第一版可以：

```python
scores = cache_fuse_metadata["dev_scores"]
scores = scores / len(cache_fuse_metadata["sel_layers"])
```

注意：

如果每层都只是直接相加，并且参与层数固定，那么“求和”和“求平均”对 top-k 排名其实完全一样。

更重要的问题是：

> 不同层的 deviation 数值尺度可能不同。

论文只说“aggregate signals across multiple layers”，没有公开具体归一化公式。

所以建议：

### 第一版

直接平均，先保证逻辑跑通。

### 后续实验版

尝试每层先归一化再相加，例如：

```python
normalized = temp_diff / (temp_diff.mean() + 1e-6)
```

或者使用 rank aggregation（按每层排名聚合）。

**这两种都属于工程实验，不要写成论文原方法。**

---

## 4.2 只从文档部分选择 important tokens

```python
suffix_len = cache_fuse_metadata["suffix_len"]
total_len = value.shape[0]
doc_len = total_len - suffix_len

ratio = cache_fuse_metadata["recomp_ratio"]

topk_num = int(doc_len * ratio)
topk_num = max(1, min(doc_len, topk_num))

doc_imp_indices = torch.topk(
    scores,
    k=topk_num,
).indices

doc_imp_indices = torch.sort(doc_imp_indices).values
```

问题 token 不参与 top-k。

问题 suffix 的完整原始位置为：

```python
suffix_indices = torch.arange(
    doc_len,
    total_len,
    device=doc_imp_indices.device,
    dtype=torch.long,
)
```

然后保存：

```python
cache_fuse_metadata["doc_imp_indices"] = doc_imp_indices
cache_fuse_metadata["num_imp_tokens"] = doc_imp_indices.numel()

imp_indices = torch.cat(
    [doc_imp_indices, suffix_indices],
    dim=0,
)

cache_fuse_metadata["imp_indices"] = imp_indices
```

`imp_indices` 继续保留，是为了兼容 `llama.py` 现有逻辑：

```python
residual = residual[imp_indices]
positions = positions[imp_indices]
```

---

# 七、第 5 步：把 DAF 注意力写成一个可复用 helper

不要把 `status=1` 和 `status=2` 各复制一大段代码。

建议抽出一个概念函数，例如：

```python
def _run_daf_attention(
    query,
    key,
    value,
    key_old,
    value_old,
    cache_fuse_metadata,
):
    ...
```

更进一步，可以再抽一个低层 dense-attention helper：

```python
def _dense_attention(self, q, k, v, attn_bias):
    ...
```

这个 helper 的职责是：

1. 正确处理 MHA/GQA；
2. 添加 batch 维；
3. 调 `xops.memory_efficient_attention_forward()`；
4. 恢复输出 shape。

---

## 什么是 GQA

GQA = Grouped-Query Attention，分组查询注意力。

例如当前 CacheBlend 示例用的 Mistral-7B：

```text
Q heads  = 32
KV heads = 8
```

Q 的 head 数和 K/V 不一样。

当前 `xformers.py` 已经有一段专门处理 GQA 的 reshape/expand 逻辑。

**正确做法是复用这一段逻辑，而不是在 DAF 外层再随意 expand 一次。**

---

# 八、第 6 步：DAF 三个阶段的正确数据流

DAF 对每个 compact 层都做：

```text
A. important-token self-attention
B. question-document self-attention
C. state fusion
```

---

## A. Important-token self-attention

在第一次选择层 `status=1` 中，当前 Q/K/V 仍然对应完整输入。

取出：

```python
doc_imp = cache_fuse_metadata["doc_imp_indices"]

q_imp = query[doc_imp]
k_imp = key[doc_imp]
v_imp = value[doc_imp]
```

然后：

```python
out_imp = dense_attention(
    q_imp,
    k_imp,
    v_imp,
    attn_bias=None,
)
```

这里 `attn_bias=None` 表示不使用 causal mask，即这些重要 token 可以互相读取。

这符合论文的：

```text
compute self-attention mutually among them
```

它的主要二次复杂度约为：

```text
(r * Ld)^2 = r^2 * Ld^2
```

当 `r = 0.3` 时：

```text
r^2 = 0.09
```

也就是二次项的系数从约 `0.3` 下降到 `0.09`。

---

## B. 先更新当前层的 hybrid KV，再让问题读取完整上下文

这里需要纠正原计划的一句话：

> 不是“important token 开完会以后，才产生新的 K/V”。

在 Transformer 的当前层中：

```text
hidden_states
    ↓
QKV projection
    ↓
q_imp, k_imp, v_imp
    ↓
attention
    ↓
out_imp
```

当前层用于 KV cache 的 `k_imp/v_imp` 在 **QKV projection 阶段就已经产生**。

`out_imp` 会成为本层输出 hidden states，并在**下一层**再产生新的 Q/K/V。

因此当前层应该做的是：

```python
key_old[doc_imp] = k_imp
value_old[doc_imp] = v_imp
```

然后构造“混合 KV”：

```text
未选文档 token -> 离线旧 KV
重要文档 token -> 当前请求重新计算的 KV
问题 token     -> 当前请求的新鲜 KV
```

---

## C. Question-document self-attention

问题是输入的尾部 suffix。

第一次选择层：

```python
q_suf = query[doc_len:]
k_suf = key[doc_len:]
v_suf = value[doc_len:]
```

文档部分使用 hybrid cache：

```python
k_doc_hybrid = key_old[:doc_len]
v_doc_hybrid = value_old[:doc_len]
```

拼接：

```python
k_all = torch.cat([k_doc_hybrid, k_suf], dim=0)
v_all = torch.cat([v_doc_hybrid, v_suf], dim=0)
```

然后：

```python
out_suf = dense_attention(
    q_suf,
    k_all,
    v_all,
    attn_bias=LowerTriangularFromBottomRightMask(),
)
```

这里 bottom-right causal mask 很适合 DAF 的 question path，因为：

```text
Q = 最尾部连续的问题 token
K/V = 所有文档 + 问题 token
```

所以每个问题 token：

- 可以读取全部文档；
- 可以读取自己前面的 question token；
- **不能偷看后面的 question token**。

原计划里“问题内部的字偷看后面的字”应改成“不能偷看后面的字”。

---

## D. State Fusion

最后：

```python
out = torch.cat(
    [out_imp, out_suf],
    dim=0,
)
```

这时输出顺序为：

```text
[重要文档 token, 问题 token]
```

同时：

```python
imp_indices = torch.cat(
    [doc_imp_indices, suffix_indices]
)
```

使用完全相同的顺序。

因此 `llama.py` 中现有的：

```python
residual = residual[imp_indices]
positions = positions[imp_indices]
```

可以继续复用。

这就是 DAF 论文所说的 State Fusion：把两条注意力路径的输出拼成下一层的 compact hidden states。

---

# 九、第 7 步：`status=2` 也执行 DAF，而不是退回 CacheBlend

进入下一层后，输入已经不是完整序列，而是：

```text
[重要文档 token | 问题 token]
```

设：

```python
num_imp = cache_fuse_metadata["num_imp_tokens"]
doc_imp = cache_fuse_metadata["doc_imp_indices"]
suffix_len = cache_fuse_metadata["suffix_len"]
```

则 compact QKV 可以切成：

```python
q_imp = query[:num_imp]
k_imp = key[:num_imp]
v_imp = value[:num_imp]

q_suf = query[num_imp:]
k_suf = key[num_imp:]
v_suf = value[num_imp:]
```

---

## 9.1 当前层更新重要文档 KV

注意：

`doc_imp_indices` 是它们在“完整原始文档序列”中的位置。

因此：

```python
key_old[doc_imp] = k_imp
value_old[doc_imp] = v_imp
```

---

## 9.2 important-token self-attention

```python
out_imp = dense_attention(
    q_imp,
    k_imp,
    v_imp,
    attn_bias=None,
)
```

---

## 9.3 question-document self-attention

```python
doc_len = key_old.shape[0] - suffix_len

k_doc_hybrid = key_old[:doc_len]
v_doc_hybrid = value_old[:doc_len]

k_all = torch.cat(
    [k_doc_hybrid, k_suf],
    dim=0,
)

v_all = torch.cat(
    [v_doc_hybrid, v_suf],
    dim=0,
)

out_suf = dense_attention(
    q_suf,
    k_all,
    v_all,
    attn_bias=LowerTriangularFromBottomRightMask(),
)
```

---

## 9.4 再次融合

```python
out = torch.cat(
    [out_imp, out_suf],
    dim=0,
)
```

每一层都重复这个结构。

这才真正保持：

```text
important path: r^2 * Ld^2
question path : Lq * (Ld + Lq)
```

---

# 十、第 8 步：RoPE 不要直接写“完全不用碰”，而要做明确检查

RoPE = Rotary Position Embedding，旋转位置编码。

CacheBlend 原型的 `hack_kv` 是在：

```python
q, k = self.rotary_emb(...)
```

之前保存 K 的。

在线使用旧 K 时，原代码会再根据 `org_pos` 给旧 K 加上正确的 RoPE。

第一版如果 deviation 只比较 V：

- status=3 评分不需要旧 K；
- status=1/2 做 question attention 时仍需要正确旋转旧 K。

因此现有 RoPE 机制**大概率可以复用**，但建议不要写成“天然保证，不需要检查”。

至少加入以下断言/检查：

```text
1. key_old 在进入 DAF attention 前是否已经完成 RoPE。
2. fresh k_imp/k_suf 是否也处于相同坐标系。
3. 同一个 old_kv 是否可能在同一请求中被重复旋转。
4. 跨请求复用缓存时，是否会因为原地修改产生二次旋转。
```

---

# 十一、第 9 步：不要污染共享的离线 KV cache

这是原计划没有强调的一个重要风险。

DAF 会做：

```python
key_old[doc_imp] = k_imp
value_old[doc_imp] = v_imp
```

这些新的 K/V 是**当前问题请求相关**的。

如果 `old_kv` 真的是多个请求共享的离线缓存，那么直接原地修改会导致：

```text
问题 A 改过的缓存
    ↓
问题 B 又拿来用
    ↓
缓存被污染
```

当前 `example/blend.py` 每个 sample 会重新构建 `old_kvs`，所以简单实验里不一定暴露。

但如果以后做真正服务，应把“离线 KV”和“当前请求的 hybrid KV”分开。

第一版为了正确性可以：

```python
key_hybrid = key_old.clone()
value_hybrid = value_old.clone()
```

然后只修改 hybrid 副本。

注意：全量 clone 会增加显存拷贝和 TTFT，所以它适合 correctness prototype，不适合作为最终性能实现。正式版本应该使用 request-local buffer 或更高效的缓存管理方式。

---

# 十二、第 10 步：修改 `example/blend.py`

至少做下面几件事。

## 1. 修复 suffix length

```python
last_len = len(q_ids + s_end)
```

---

## 2. 不要硬编码层数

```python
llm_layers = (
    llm.llm_engine.model_executor.driver_worker
    .model_runner.model.model.layers
)
num_layer = len(llm_layers)
```

---

## 3. 开启 DAF

```python
cache_fuse_metadata["daf"] = True

cache_fuse_metadata["sel_layers"] = [1, 2, 3]

cache_fuse_metadata["recomp_ratio"] = 0.3
# 论文实验配置，不是永远固定值
```

如果 scheduler 已经使用：

```python
select_layer = max(sel_layers) + 1
```

就没有必要同时再维护一个可能与 `sel_layers` 冲突的：

```python
check_layers = [4]
```

如果为了兼容旧代码仍保留 `check_layers`，应在初始化时自动同步：

```python
cache_fuse_metadata["check_layers"] = [
    max(cache_fuse_metadata["sel_layers"]) + 1
]
```

避免两套配置互相矛盾。

---

## 4. collect 阶段第一版可以继续沿用，但要理解它不是论文的最简结构

DAF 论文图中“Offline KV Cache”对应的是文档缓存，问题是在在线阶段处理的。

当前 CacheBlend 原型为了代码方便，把最后的问题 chunk 也放进 `old_kvs` 结构。

第一版可以保留这个结构，以减少改动，但在 DAF question attention 中：

```python
k_doc_hybrid = key_old[:doc_len]
```

只使用旧缓存中的文档部分，并用新鲜 `k_suf/v_suf` 作为问题 KV。

这样语义更接近论文。

---

# 十三、建议的开发顺序

不要一次把所有东西一起改。

## 阶段 1：先修基线

只改：

```text
suffix_len
动态 num_layer
```

确认 CacheBlend 还能工作。

---

## 阶段 2：只加入多层评分，不改变最终注意力

加入：

```text
daf
sel_layers
dev_scores
status=3
```

但选择层仍暂时走原 CacheBlend。

验证：

```text
dev_scores.shape == [doc_len]
每个 sel_layer 都确实累加一次
问题 suffix 没进入 deviation
```

---

## 阶段 3：只改 status=1 为 DAF

第一次进入 compact 状态时实现：

```text
important self-attention
question-document attention
state fusion
```

先确认 shape 和输出正常。

---

## 阶段 4：把同一 DAF helper 接到 status=2

这一步完成后才算“完整 DAF”。

---

## 阶段 5：性能优化

正确性稳定后再处理：

```text
减少 clone
减少 view/reshape
避免 Python list
预分配索引
Flash-Attention backend
CUDA profiling
```

---

# 十四、强烈建议加入的 shape 断言

刚开始调试时不要只 `print(shape)`，最好直接 `assert`。

例如：

```python
assert suffix_len > 0
assert doc_len > 0

assert scores.shape[0] == doc_len

assert doc_imp_indices.ndim == 1
assert doc_imp_indices.numel() == topk_num

assert torch.all(doc_imp_indices < doc_len)

assert q_imp.shape[0] == num_imp
assert q_suf.shape[0] == suffix_len

assert k_all.shape[0] == doc_len + suffix_len
assert v_all.shape[0] == doc_len + suffix_len

assert out_imp.shape[0] == num_imp
assert out_suf.shape[0] == suffix_len

assert out.shape[0] == num_imp + suffix_len
```

还应检查：

```python
assert not torch.isnan(out).any()
assert not torch.isinf(out).any()
```

等程序稳定后再移除或用 debug flag 包住。

---

# 十五、验证方案 v2

| 顺序 | 验证内容 | 推荐通过标准 |
|---:|---|---|
| 1 | 修复后的 `daf=False` CacheBlend | greedy decoding 下生成 token 与修复后基线一致 |
| 2 | `status=3` 单独测试 | 每层 score 长度等于 `doc_len`，且只累计文档 token |
| 3 | `status=1` shape 测试 | `[important, question]` 的 Q/K/V/output 形状全部满足断言 |
| 4 | `status=2` 深层测试 | 每层保持 compact 长度，不重新膨胀成完整 Q |
| 5 | Full / CacheBlend / DAF 输出比较 | DAF 不出现明显语义崩坏；正式数据集用 EM、Rouge-L 等指标 |
| 6 | 短上下文速度 | 不要求 DAF 必然严格快于 CacheBlend |
| 7 | 长上下文速度 | 重点观察上下文增长后 DAF 相对 CacheBlend 的优势是否扩大 |
| 8 | 多次运行 | 做 warm-up，并报告中位数或 P50，不要只看一次 TTFT |

---

## 为什么“DAF TTFT 必须 <= CacheBlend”不适合作为单次功能测试

论文表中：

- 6k：DAF 与 CacheBlend TTFT 相同；
- 9k：只快一点；
- 30k：差距明显；
- 136k：差距很大。

因此 DAF 的优势主要应该在长上下文中体现。

另外你的第一版是在 `xformers.py` 上实现，而论文强调的是标准 dense attention 可以很好使用 Flash-Attention。

因此：

> “xFormers 原型没有复现论文同样的绝对速度”并不自动代表算法实现错误。

---

# 十六、论文复现和“算法原型”要分开

当前 CacheBlend 示例使用：

```text
Mistral-7B-Instruct-v0.2
```

DAF 论文实验使用：

```text
Qwen2.5-7B
单张 NVIDIA A100 80GB
greedy decoding
r = 0.3
```

因此建议把目标分两级。

## 目标 A：功能复现

```text
Mistral-7B + 当前 CacheBlend xformers 原型
```

验证：

```text
算法数据流正确
质量不崩
复杂度趋势合理
```

## 目标 B：论文实验复现

再尽量对齐：

```text
Qwen2.5-7B
A100 80GB
相同数据集
相同长度
greedy decoding
r = 0.3
Flash-Attention 路径
```

否则只能说：

> “实现了 DAF-inspired / DAF-style 原型”

不宜直接说“完整复现论文性能”。

---

# 十七、论文明确支持 vs 我们的工程假设

| 项目 | 论文是否明确 |
|---|---|
| 聚合多层 selection signal，通常第 2～4 层 | **是** |
| important tokens 之间互相 self-attention | **是** |
| question 位于尾部并读取完整上下文 | **是** |
| important-token KV 在 question attention 前已更新 | **是** |
| 两路输出做 state fusion / concatenate | **是** |
| 复杂度从 `r Ld²` 降为 `r² Ld²` 级别 | **是** |
| 实验使用 `r=0.3` | **是** |
| `r=0.3` 永远固定 | **否** |
| 多层 deviation 一定用 raw mean | **否，论文未写具体公式** |
| 一定只比较 V deviation | **否，论文写的是 KV deviation** |
| 第 2～4 层一定全部 full attention | **论文未给代码级细节** |
| 一定在第 5 层第一次选择 | **论文未写死** |
| `xformers.py` 就能复现论文 Flash-Attention 性能 | **否** |
| 深层继续 CacheBlend 仍然算完整 DAF | **否，复杂度不匹配** |

---

# 十八、最终推荐的最小正确架构

```text
                 ┌──────────────────────────────┐
                 │  Layer 0: full prefill       │
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
                 │ Layers 1~3: full prefill     │
                 │ + accumulate deviation       │
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
                 │ Select important doc tokens  │
                 └──────────────┬───────────────┘
                                │
                ┌───────────────▼────────────────┐
                │ For each remaining layer:      │
                │                                │
                │  important QKV ─► self-attn    │
                │          │                     │
                │          ├─► update hybrid KV  │
                │          │                     │
                │ question QKV ─► full-context   │
                │                 causal attn     │
                │          │                     │
                │  concatenate outputs           │
                └───────────────┬────────────────┘
                                │
                                ▼
                       next compact layer
```

这里“compact”指：

> 不再让所有文档 token 的 hidden state 继续逐层计算，只保留重要文档 token 和问题 token。

---

# 十九、最容易出错的 8 个点

1. **`suffix_len` 计算错误**：会直接破坏所有文档/问题边界。
2. **status=3 没拿到 `value_old`**：会直接报变量未定义。
3. **把 suffix 也参加 top-k**：问题 token 应无条件保留，而不是参加文档重要性排序。
4. **只保存 `imp_indices`，不保存 document-only indices**：深层无法区分两条 DAF 路径。
5. **误以为 `out_imp` 才是新 KV**：当前层 KV 来自 QKV projection，不来自 attention output。
6. **status=2 又让 important Q 读取完整文档**：这样会退回 CacheBlend 的 `r * Ld²`。
7. **直接修改共享离线 KV**：多请求时会发生缓存污染。
8. **在 xFormers 上测到速度不够快就判定 DAF 错误**：论文的性能卖点还依赖 dense kernel / Flash-Attention 的硬件利用率。

---

# 二十、一句话实施建议

第一版不要追求“又快又完全复现”。

按下面顺序最稳：

```text
先修 suffix_len
→ 保证 CacheBlend 基线正确
→ status=3 多层打分
→ status=1 两路 DAF
→ status=2 两路 DAF
→ 做 shape/质量验证
→ 最后才优化 Flash-Attention 和缓存拷贝
```

这样每一步只引入一种新错误来源，出了问题更容易定位。
