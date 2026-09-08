# DAF 改造计划（基于 CacheBlend 原型仓库）

## 第 0步：开工前的准备（不改代码，先做保障）

- **git 提交或备份当前状态**。`vllm_blend` 和 `blend.py` 现在是能跑通的版本，改坏了要能回滚。建议：`git add -A && git commit -m "checkpoint before DAF"`。
- **明确改造范围**：只动 3 个文件。`vllm_blend/vllm/model_executor/models/llama.py`（调度）、`vllm_blendllm/attention/backends/xformers.py`（算法核心）、`example/blend.py`（驱动脚本）。
- **原则：DAF 做成"可选开关"，不破坏 CacheBlend**。所有新逻辑都包在 `if cache_f_metadata["daf"]` 里，开关关掉时行为和现在一模一样——这样你可以在同一套代码里随时对比两种算法，这也是写论文/报告时必须的实验对照。

------

## 第 1 步：改造 `llama.py` 的控制字典（改动最小，10 分钟）

**位置**：约 llama.py:300,`LlamaModel.__init__` 里的 `self.cache_fuse_metadata = {...}`。



**加三个键**:

```
"daf": False,          # DAF 总开关
"sel_layers": [1, 2, 3],  # 参与投票的层论文的"第2~4层"，代码从0数所以是1,2,3）
"dev_scores": None,    # 偏差记分板：攒多层偏差用
```

**为什么**:



- `daf` 开关——见 0 步的原则，保证可随时切回 CacheBlend 做对照。
- `sel_layers`——这是 DAF 论文和 CacheBlend 的**第一个本质区别**。CacheBlend 只问第 2 层"哪些 token 偏差大"；单层判断有随机性，长文本下会挑错人。DAF 让第 2~4 层各自打分、取平均再选人，相当于"三个评委投票"，选得更稳。
- `dev_scores`——要投票就得有地方存票这是一个长度等于"文档 token 数"的张量，每过一层就往里累加一次偏差，攒到选择层再用。



同时在 check 初始化块（约 llama.py:332-340，就是那串 `self.cache_fuse_metadata[] = None` 的地方）加一行 `self.cache_fuse_metadata["dev_scores"] = None`。**为什么**：每次新请求开始时必须清空上一轮的记分板，否则不同样本的偏差会混在一起。

------

## 第 2 步：改造 `llama.py` 的层循环排班表（30 分钟）

**位置**：约 llama.py:348-356，`for i in range(len(self.layers))` 循环里的 status 分配。



**现在的逻辑**:check 模式下，第 1 层是"选择层"(status=1)，之后的层全是 status=2。



**改成 DAF 模式下的新排班**(`daf=True` 时生效）:

| **istatus干什么为什么** |                 |                                                  |                                                              |
| ----------------------- | --------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| 0                       | 0               | 正常全量 prefill                                 | 第 1 层还没有任何缓存对照，只能全算（CacheBlend 也这样）     |
| 1, 2, 3                 | **3（新状态）** | 正常全量 prefill + 把本层偏差累加进 `dev_scores` | 这就是"三个评委打分"的过程。要打分就得先算出新鲜值，所以这 层必须全量跑——这是 DAF 比 CacheBlend 多付的固定成本 |
| 4                       | 1               | 聚合分数 → 选人 → 执行 DAF 三板斧                | 选择点推迟到第 5 ，因为前 3 层的票要攒齐                     |
| 5~31                    | 2               | 只有被选中的 token 继续往后走                    | 和 CacheBlend 一样：没选中的 token 从此"冻结"，省掉它们的后续计算 |

**为什么这么排**:CacheBlend 在第 2 层就选人（快但草率）;DAF 故意多花 3 层的全量计算换选人质量。论文实验表明这个交换是赚的——选人准了，后面 层的修复效果就好，长文本下精度明显更高。

------

## 第 3 步：给 `xformers.py` 加 status=3 分支（30 分钟）

**位置**:`XFormersImpl.forward`，在 `if status in [1]:` 附近加分支。



**干什么**：照抄现有算偏差的代码（xformers.py:209-211 附近，`temp_diff = torch.sum((value[:-last_len] - value_old[:-last_len])**2, dim=[1,2])`)，但**不做 topk、不剪枝**，只做两件事：



1. `dev_scores = dev_scores + temp_diff`（累加投票）
2. 注意力走正常全因果路径（和 status=0 一样）



**为什么**:status=3 的层是"评委"，职责是打分而不是。偏差度量的定义沿用 CacheBlend 的 **V 偏差**（新鲜算的 V 和缓存的 V 的平方差），因为 V 直接决定注意力的输出内容，偏差大的 token 就是"受跨文档缺失影响最大" token——也就是最该修的人。这个度量是 CacheBlend 论文验证过的，DAF 论文也沿用，没必要发明新的。

## 第 4 步：重写 `xformers.py` 的 status=1 分支（核心改造，半天）

**位置**：现有 `if status in [1]:` 分支（xform.py:204-221 附近）。CacheBlend 原来的逻辑是"选人 → 被选的人对**全部**上下文做一次注意力"。DAF 要把它拆成三次操作。

**现在的 CacheBlend 流程**（对照用）:

```
算出每个 token 的 V 偏差 → topk 选 16% → 被选的 token + 问题 token
→ 对整段上下文做一次带因果掩码的注意力 → 完
```

**改成 DAF 流程**(daf=True 时走新路）:

```
# —— 第 0 小步：投票汇总 + 选人 ——
score = dev_scores / len(sel_layers          # 三个评委的票取平均
top_indices = topk(score, k=int(0.3 * 文档长度))  # 选出偏差最大的 30%
top_indices = top_indices.sort()              # 排序！保持原有先后顺序
suffix_indices = 末尾 suffix_len 个位置          # 问题段无条件入选

# —— 第一板斧：重要 token 开小会（互看）——
q_imp, k_imp, v_imp = qkv[top_indices]     # 只取被选中 token 的 Q/K/V
out_imp = attention(q_imp, k_imp, v_imp, attn_bias=None)
#         ↑ 无掩码 = 双向互看 = 真正的"互相开会"
#         计算量只有 (0.3L)² = 0.09L²，这是 DAF 省钱的

# —— 把开完会的新笔记写回缓存柜子 ——
old_kv.K[top_indices] = k_imp
old_kv.V[top_indices] = v_imp
#   ↑ 对应论文原文："the KV states for important tokens have
#     already been updated during the QKV computation phase"

# —— 第二板斧：问题读全局 ——
K_all = cat(old_kv.K, 新鲜问题k)               # 混合缓存：重要 token 是新的，其余是
V_all = cat(old_kv.V, 新鲜问题v)
out_suf = attention(q_suf, K_all, V_all,
                    attn_bias=LowerTriangularFromBottomRightMask())
#         ↑ 因果掩码：问题内部的字偷看后面的字（这是生成模型铁律）

# —— 第三板斧：拼回去 ——
out = cat(out_imp, out_suf)
```

**为什么这么改，逐条说**:

- **为什么小会不带掩码（双向）**：论文说的是 "self-attention mutually among them"（互相做注意力）。开小会的目的就是让关键信息跨文档流动，张三的要点和李四的要点地位平等，不存在"谁在前谁在后"的因果限制，所以双向。
- **为什么先写回缓存算问题**：这样第二板斧就是一次**标准的、无分支的**稠密注意力——不用在 kernel 里判断"哪些位置是新的哪些是旧的"。论文特意强调这个设计（"eliminates conditional checks, avoiding thread divergence"),GPU 上分支判断很伤性能。
- **为什么拼接顺序是 [排序后的重要token, 问题token]**：这是**故意的兼容设计**——CacheBlend 现有的 `imp_indices` 恰好就是这个顺序，下游所有代码（llama.py 里切 residual 的 `residual[imp_indices]`、切 positions 的那行）都按这个约定写的，我们一行都不用动。
- **为什么 RoPE 相关代码不用碰**：现有的 `fake_q`/`org_pos`机制负责给缓存的 K 补上位置编码，它在注意力调用之前就生效，两板斧都是在它之后发生的事，天然继承。

------

## 第 5 步：决定深层（status=2）怎么办（论文没写，要自己选**两个选项**:

- **选项 A（省事版）**：第 5~32 层保持 CacheBlend 老逻辑——被选中的 token 每层对全部缓存做因果注意力。改动量：零。
- **选项 B（忠实版，推荐）**：第 5~32 层也按 DAF 三板斧来——每层重要 token 先互看、写回、问题再读全局。改动量：把第 4 步写的逻辑抽成一个函数，status=1 和 status=2 都调它。

为什么推荐 B**：论文的复杂度优势（r² 而不是 r）只有每层都按 DAF 走才能完整兑现。如果用 A，深层每层仍是"30% 的人读 100% 的材料"，长上下文下的加速比会明显。而且 B 的边际成本很低——就是函数复用。

**建议先 A 后 B**：第一次实现时用 A，跑通、验证精度；确认无误后再升级到 B，对比 TTFT 差异。这样出问题时容易定位——永远是"一次只改一件事"。

------

## 第 6 步：改 `example/blend.py` 驱动脚本（10 分钟）

只加配置，不改流程：

```
cache_fuse_metadata["daf"] = True            # 打开 DAF 模式
cache_fuse_metadata["recomp_ratio"] = 0.3    # 选人比例30%（论文固定值）
cache_fuse_metadata["check_layers"] = [4]    # 第 5 层（0-indexed 4）做选择
```

**为什么 collect 阶段一行不改**:DAF 和 CacheBlend 的"离线建缓存"骨架完全相同——都是逐块喂文档、偷 `hack_kv`、拼进 `old_kvs` 柜子。两种算法的差别只在"在线推理时怎么修复"，所以阶段一原样保留。

------

## 第 7 步：验证（按顺序做，一步不过不往下走）

| 顺序 | 验证内容                             | 通过标准                                             |
| :--- | :----------------------------------- | :--------------------------------------------------- |
|      | `daf=False` 跑一遍                   | 结果和改造前完全一致（证明没碰坏 CacheBlend)         |
| 2    | `daf=True` 跑 `inputs/1.json`        | 输出与全量 prefill 语义一致（答非所问）              |
| 3    | 三模式同场对比（全量/CacheBlend/DAF) | DAF 的 TTFT ≤ CacheBlend，质量 ≥ CacheBlend          |
| 4    | 拼更多文档块撑长上下文               | DAF vs CacheBlend 的 TTFT 差距随长度拉大（核心趋势） |

**为什么第 1 步最重要**：它是回归测试。改造引入的 bug 十有八九会先在这里暴露（比如开关没隔离好），先排除它再谈新功能。

------

## 风险清单（提前知道哪里容易翻车）

1. **张量形状对不齐**：两路注意力拆开后，`out_imp` 和 `out_suf` 的拼接顺序、 residual 的切片下标必须严格一致。这是最容易出错的地方调试时用 `print(shape)` 大法逐层核对。
2. **GQA 头数问题**:Mistral-7B 是 GQA(32 个 Q 头但只有 8 个 KV 头），调用 xformers 时 q 和 k/v 的头数不同，要照抄现有代码的处理方式，别自己 expand。
3. **记分板没清零**：新请求复用了旧请求的 `dev_scores`，结果莫名其妙地错。第 1 步的清零千万别漏。
4. **论文没写明的假设**：选人只做一次（深层沿用）、"小会"双向、投票取平均——这些是我们的合理假设，实验记录里要注明。