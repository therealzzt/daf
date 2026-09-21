# ChatML 消融报告：裸 prompt vs 官方对话模板对 RULER 30k 档精度的影响

> 日期：2026-09-21
> 目的：daf1.4 发现 RULER 30k 档三方法绝对精度系统性低于论文（qa_2 全量基线
> SM 0.458 vs 论文 vLLM 0.765），三个嫌疑之一是"prompt 形态与官方评测不一致"。
> 本消融在 500 样本放量前量化该因子
> 样本：`experiments/inputs_qwen_ruler30k/` 现有 48 样本（qa_1/qa_2 各 24），未重新生成
> 评分：`--raw_parse`（SM = 归一化后互为子串 ≈ RULER 官方 string_match_part）

## 一、现状核实结论：上轮不是裸 prompt，是"半官方 ChatML"

读 blend.py qwen 路径（`example/blend.py:122` 起）+ 重建上轮实际输入后发现：
上轮 RULER 跑批时模型看到的 prompt **已经套了 ChatML 信封**——

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n
{RULER 完整 prompt：指令 + Document 1..N + 指令 + Question: {query} Answer:}
<|im_end|>\n<|im_start|>assistant\n
```

与 RULER 官方评测形态的唯一差别在 **system 内容**：

- 上轮：`You are a helpful assistant.`（blend.py qwen 预设写死）
- 官方：RULER 的 client（`ruler_src/scripts/pred/client_wrappers.py:288-291`）
  只发 user 单条消息、**不发 system**；prompt = `input + answer_prefix`
  （`call_api.py:305`）。服务端套 Qwen2.5 默认 chat template，system 为
  `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`

即：信封结构（system+user+assistant 头、` Answer:` 留在 user 尾部）上轮就已
与官方一致，差异仅 system 一句话的措辞。**daf1.4.md 与
em_report_qwen_ruler30k.md 第四节中"未套 ChatML 对话模板"的表述不准确，
以本报告为准**（两处已加更正注记）。

## 二、开关实现

`example/blend.py:122-141`：qwen 预设新增 `BLEND_CHATML` 环境变量——

- 默认（0/未设）：保持历史行为（system = "You are a helpful assistant."），
  与 B-1/B-2/B-3 已跑数据完全一致
- `BLEND_CHATML=1`：用 `tokenizer.apply_chat_template` 提取官方信封
  （以 `_PLACEHOLDER_` 占位符拆分 prefix/suffix 后分别 encode），system 为
  Qwen2.5 官方默认；已验证拼接结果与官方整串逐字符一致。
  BPE 坑照旧规避：全程 token id 直传，拼接在 id 层面做

## 三、对照结果（24 样本/任务，各 1 轮；裸 = 上轮 3 轮中位）

### SM（主指标，均值）

| 任务 | 方法 | 上轮（自定义 system） | BLEND_CHATML=1（官方 system） | Δ |
|---|---|---|---|---|
| qa_1 | vLLM 全量 | 0.625 | 0.667 | +0.042（1 题） |
| qa_1 | CacheBlend | 0.458 | 0.417 | -0.042 |
| qa_1 | DAF | 0.375 | 0.375 | 0 |
| qa_2 | vLLM 全量 | 0.458 | 0.458 | 0 |
| qa_2 | CacheBlend | 0.250 | 0.208 | -0.042 |
| qa_2 | DAF | 0.250 | 0.250 | 0 |

### EM（记录用）

| 任务 | 方法 | 上轮 | ChatML |
|---|---|---|---|
| qa_1 | 全量 / CB / DAF | 0.250 / 0.125 / 0.167 | 0.250 / 0.125 / 0.167 |
| qa_2 | 全量 / CB / DAF | 0.333 / 0.083 / 0.083 | 0.292 / 0.042 / 0.083 |

### TTFT 中位（顺带记录，ChatML 只多 10 个 token，无实质影响）

| 任务 | CB 上轮→ChatML | DAF 上轮→ChatML |
|---|---|---|
| qa_1 | 3.177 → 3.172s | 2.962 → 2.986s |
| qa_2 | 3.118 → 3.117s | 2.942 → 2.939s |

## 四、结论

**prompt 形态不是精度差距的主因。** 官方 ChatML（含官方默认 system）下全量基线
qa_2 SM 0.458 → 0.458 纹丝不动，qa_1 0.625 → 0.667（+1 题，24 样本噪声范围内）；
所有方法的 |Δ| ≤ 1 题。与论文 qa_2 vLLM 0.765 的差距（-0.31）完全无法用模板
解释。剩余嫌疑按可能性排序：

1. **样本量**：24 vs 500。24 样本 ±1 题 = ±0.042；且 RULER 按 seed 抽干扰文档，
   小子集的难度分布可能与全集有偏
2. **xformers vs Flash-Attention 数值差**（理论上不影响全量基线的贪心解码结果，
   但 bf16 累加顺序差异在长序列上有微小概率翻转 argmax）
3. 论文实验的其他未公开细节（模型 revision、解码 stop 词等）

**500 样本放量建议**：用 `BLEND_CHATML=1`（官方信封）。理由不是精度收益（实测
无），而是与 RULER 官方评测形态严格一致、消除审稿口径上的质疑；代价为零
（TTFT 不变），且本轮已实测该路径稳定（4 组合 × 24 样本零 OOM、输出正常）。
若要与已跑的 24 样本三档结果严格可比，则保持默认即可——两者数字上无显著差异。

## 五、数据索引

- ChatML 日志：`experiments/logs/qwen_ruler30k/run_{cb,daf}_chatml_qwen_ruler_qa{1,2}_r1.log`
  （各 24 样本完整，零 OOM）
- 裸 prompt 基准：`experiments/logs/qwen_ruler30k/run_{cb,daf}_qwen_ruler_qa{1,2}_r{1,2,3}.log`
- 逐样本明细：本报告由 `score_em_long_6k10k.py --raw_parse` 评分，
  ChatML 明细表见附录（下）
- 已知口径差异：本消融 1 轮 vs 上轮 3 轮中位；贪心解码下质量指标逐轮确定，
  仅 TTFT 有轮次波动，不影响结论
