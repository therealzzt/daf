# =============================================================================
# 生成 Qwen2.5 实验样本（目标 B 第二档：LongBench 官方 samsum，≈9k token，Rouge-L 指标）
#
# 数据来源：experiments/longbench_src/data/samsum.jsonl（THUDM/LongBench 官方
#           data.zip 解压；字段 input/context/answers/length）
#
# LongBench samsum 的数据结构（实测 + 官方 pred.py / dataset2prompt.json 核对）：
#   context = 约 53 个完整的 "Dialogue: ...\nSummary: <参考摘要>" few-shot 示例
#   input   = "Dialogue: <待摘要对话>\nSummary: "（answers 即该对话的标准摘要）
#   官方 prompt 模板："Summarize the dialogue into a few short sentences.
#                     The following are some examples.\n\n{context}\n\n{input}"
#   官方生成上限 max_gen = 64（跑批时 BLEND_MAX_TOKENS=64）
#
# 挑选规则：用 Qwen/Qwen2.5-7B-Instruct 分词器统计「全部文档块 + 问题」的总 token 数，
#           只保留总长落在 [8000, 10000] 区间的样本（论文 samsum 均长 ≈9k），
#           按长度均匀抽样，最多 24 个
# 切块策略：按 "Dialogue: " 边界把 context 切成 few-shot 示例块（每块一个完整
#           对话+参考摘要）；官方模板的指令行拼在第 0 块开头（官方格式里指令
#           就在 context 之前）；query = input 原文（以 "Summary: " 结尾，
#           直接触发生成，不再加额外指令）
# 输出：experiments/inputs_qwen_samsum/（须从仓库根目录运行），文件名格式
#      samsum_{原jsonl行号:03d}_len{总token数}.json，另附 _selection_report.md
#
# 生成的 json 字段与 make_inputs_qwen_2wikimqa.py 一致（chunk_num / "0".."n" /
# query / gold / source_dataset / source_index / total_tokens）
# =============================================================================

import json
import os
import re

from transformers import AutoTokenizer

# ---- 参数（命名即含义） ----
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 与实验所用模型同款分词器
MIN_TOTAL_TOKENS = 8000    # 样本总 token 下限（含问题）
MAX_TOTAL_TOKENS = 10000   # 样本总 token 上限（含问题）
MAX_SAMPLES = 24           # 最多挑几个（按长度均匀抽样）
SOURCE_FILE = "experiments/longbench_src/data/samsum.jsonl"
OUTPUT_DIR = "experiments/inputs_qwen_samsum"

# LongBench 官方 samsum prompt 模板的指令行（官方格式里位于 context 之前，
# 这里拼在第 0 个文档块的开头，保持官方版式）
INSTRUCTION = ("Summarize the dialogue into a few short sentences. "
               "The following are some examples.\n\n")

# few-shot 示例块边界："Dialogue: "（context 由 ~53 个 Dialogue+Summary 段拼接而成）
SEGMENT_BOUNDARY = re.compile(r"(?=Dialogue: )")


def split_segments(context):
    """把 LongBench samsum 的 context 按 Dialogue 边界切成示例块。
    每块 = 一个完整的 'Dialogue: ...\\nSummary: <参考摘要>\\n' 段，保持原文。"""
    parts = SEGMENT_BOUNDARY.split(context)
    return [p for p in parts if p.strip()]


def count_sample_tokens(doc_texts, query_text, tokenizer):
    """总 token 数 = 各文档块 + 问题。Qwen tokenizer 不加 BOS，无需剥除"""
    total = sum(len(tokenizer.encode(t)) for t in doc_texts)
    total += len(tokenizer.encode(query_text))
    return total


def pick_evenly(candidates, k):
    """按长度均匀抽 k 个：避免挑出来的全挤在同一长度段"""
    if len(candidates) <= k:
        return candidates
    ordered = sorted(candidates, key=lambda c: c["total_tokens"])
    step = len(ordered) / k
    return [ordered[int(i * step)] for i in range(k)]


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_lines = [
        "# LongBench samsum（Qwen2.5，≈9k 档）样本挑选报告",
        "",
        f"- 分词器：`{TOKENIZER_NAME}`",
        f"- 数据来源：`{SOURCE_FILE}`（LongBench 官方 jsonl）",
        f"- 挑选区间：总 token 数 [{MIN_TOTAL_TOKENS}, {MAX_TOTAL_TOKENS}]（含问题）",
        f"- 最多取 {MAX_SAMPLES} 个（按长度均匀抽样）",
        "- 结构：context = ~53 个 Dialogue+Summary few-shot 示例（按 `Dialogue: ` 切块）；"
        "query = input 原文（`Dialogue: <待摘要对话>\\nSummary: `）；"
        "官方指令行拼在第 0 块开头",
        "- 评分主指标：Rouge-L（LongBench 官方口径，rouge_score rougeL fmeasure）",
        "",
        "## 入选样本",
        "",
        "| 输出文件 | 原行号 | 文档块数 | 总token数 | 标准摘要（截断） |",
        "|---|---|---|---|---|",
    ]

    candidates = []
    skipped_short, skipped_long, skipped_bad = 0, 0, 0
    with open(SOURCE_FILE) as f:
        for idx, line in enumerate(f):
            ex = json.loads(line)
            doc_texts = split_segments(ex["context"])
            if not doc_texts or not ex.get("input") or not ex.get("answers"):
                skipped_bad += 1
                continue
            # 官方模板：指令行在 context 之前 → 拼到第 0 块开头
            doc_texts[0] = INSTRUCTION + doc_texts[0]
            query = ex["input"]
            total = count_sample_tokens(doc_texts, query, tokenizer)
            if total < MIN_TOTAL_TOKENS:
                skipped_short += 1
                continue
            if total > MAX_TOTAL_TOKENS:
                skipped_long += 1
                continue
            candidates.append({
                "idx": idx, "doc_texts": doc_texts, "query": query,
                "gold": [str(a) for a in ex["answers"]],
                "total_tokens": total,
            })

    picked = pick_evenly(candidates, MAX_SAMPLES)
    for c in picked:
        out_name = f"samsum_{c['idx']:03d}_len{c['total_tokens']}.json"
        out = {
            "chunk_num": len(c["doc_texts"]),
            "query": c["query"],
            "gold": c["gold"],
            "source_dataset": SOURCE_FILE,
            "source_index": c["idx"],
            "total_tokens": c["total_tokens"],
        }
        for i, t in enumerate(c["doc_texts"]):
            out[str(i)] = t
        with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
            json.dump(out, f, ensure_ascii=False)
        report_lines.append(
            f"| {out_name} | {c['idx']} | {len(c['doc_texts'])} "
            f"| {c['total_tokens']} | {c['gold'][0][:60]} |"
        )

    report_lines += [
        "",
        f"候选 {len(candidates)} 个（{len(picked)} 入选 / 低于下限 {skipped_short}"
        f" / 超上限 {skipped_long} / 缺字段 {skipped_bad}）",
        "",
    ]
    if candidates:
        lens = sorted(c["total_tokens"] for c in candidates)
        report_lines.append(
            f"候选长度分布：min={lens[0]}, 中位={lens[len(lens)//2]}, max={lens[-1]}"
        )
        picked_lens = sorted(c["total_tokens"] for c in picked)
        report_lines.append(
            f"入选长度分布：min={picked_lens[0]}, "
            f"中位={picked_lens[len(picked_lens)//2]}, max={picked_lens[-1]}, "
            f"均值={sum(picked_lens)/len(picked_lens):.0f}"
        )

    with open(os.path.join(OUTPUT_DIR, "_selection_report.md"), "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"完成：{len(picked)} 个样本写入 {OUTPUT_DIR}/，明细见 _selection_report.md")


if __name__ == "__main__":
    main()
