# =============================================================================
# 生成 Qwen2.5 实验样本（目标 B 第一档：LongBench 官方 2wikimqa，≈6k token，EM 指标）
#
# 数据来源：experiments/longbench_src/data/2wikimqa.jsonl（THUDM/LongBench 官方
#           data.zip 解压；字段 input/context/answers/length）
# 挑选规则：用 Qwen/Qwen2.5-7B-Instruct 分词器统计「全部文档块 + 问题」的总 token 数，
#           只保留总长落在 [5000, 7000] 区间的样本（论文 2wikimqa 均长 ≈6k），
#           按长度均匀抽样，最多 24 个
# 切块策略：LongBench 的 context 是 "Passage N:\n标题\n正文" 的拼接，按 Passage
#           边界切分——每篇 Passage 一个文档块（去掉 "Passage N:" 标记行，
#           保留 标题\n正文\n\n，拼法对齐 musique_s 的 title\n\ntext\n\n）
# 输出：experiments/inputs_qwen_2wikimqa/（须从仓库根目录运行），文件名格式
#      2wikimqa_{原jsonl行号:03d}_len{总token数}.json，另附 _selection_report.md
#
# 生成的 json 字段（与 example/blend.py 兼容，gold 字段名对齐
# experiments/scripts/score_em_long_6k10k.py 的读取逻辑）：
#   chunk_num / "0".."chunk_num-1" / query / gold / source_dataset /
#   source_index / total_tokens
# =============================================================================

import json
import os
import re
import sys

from transformers import AutoTokenizer

# ---- 参数（命名即含义） ----
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 与实验所用模型同款分词器
MIN_TOTAL_TOKENS = 5000    # 样本总 token 下限（含问题）
MAX_TOTAL_TOKENS = 7000    # 样本总 token 上限（含问题）
MAX_SAMPLES = 24           # 最多挑几个（按长度均匀抽样）
SOURCE_FILE = "experiments/longbench_src/data/2wikimqa.jsonl"
OUTPUT_DIR = "experiments/inputs_qwen_2wikimqa"

# 问题前缀/后缀：与 example/blend_musique.py 的 query_prompt 保持一致，
# 让模型"直接作答、5词以内"，EM 判分才可行
QUERY_INSTRUCTION = (
    "\n\nAnswer the question directly based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words. "
    "\nQuestion:"
)

# LongBench context 的 Passage 边界标记："Passage 12:\n"
PASSAGE_MARKER = re.compile(r"^Passage \d+:\n", re.M)


def split_passages(context):
    """把 LongBench 的 context 按 Passage 边界切成文档块列表。
    每块 = 去掉 'Passage N:' 标记行后的 标题\\n正文，末尾补 \\n\\n
    （与 musique_s 样本的块格式一致）。"""
    parts = PASSAGE_MARKER.split(context)
    chunks = [p.strip() + "\n\n" for p in parts if p.strip()]
    return chunks


def count_sample_tokens(doc_texts, query_text, tokenizer):
    """总 token 数 = 各文档块 + 问题。Qwen tokenizer 不加 BOS，
    不需要像 Mistral 那样 [1:] 剥除（与 blend.py qwen 预设的口径一致）"""
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
        "# LongBench 2wikimqa（Qwen2.5，≈6k 档）样本挑选报告",
        "",
        f"- 分词器：`{TOKENIZER_NAME}`",
        f"- 数据来源：`{SOURCE_FILE}`（LongBench 官方 jsonl）",
        f"- 挑选区间：总 token 数 [{MIN_TOTAL_TOKENS}, {MAX_TOTAL_TOKENS}]（含问题）",
        f"- 最多取 {MAX_SAMPLES} 个（按长度均匀抽样）",
        "- 切块：按 `Passage N:` 边界，每篇 Passage 一个文档块；"
        "问题拼法与 `example/blend_musique.py` 的指令一致",
        "",
        "## 入选样本",
        "",
        "| 输出文件 | 原行号 | 文档块数 | 总token数 | 问题 | 标准答案 |",
        "|---|---|---|---|---|---|",
    ]

    candidates = []
    skipped_short, skipped_long, skipped_bad = 0, 0, 0
    with open(SOURCE_FILE) as f:
        for idx, line in enumerate(f):
            ex = json.loads(line)
            doc_texts = split_passages(ex["context"])
            if not doc_texts or not ex.get("input") or not ex.get("answers"):
                skipped_bad += 1
                continue
            query = QUERY_INSTRUCTION + ex["input"] + "\nAnswer:"
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
                "total_tokens": total, "question": ex["input"],
            })

    picked = pick_evenly(candidates, MAX_SAMPLES)
    for c in picked:
        out_name = f"2wikimqa_{c['idx']:03d}_len{c['total_tokens']}.json"
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
        ans_preview = "; ".join(c["gold"])[:60]
        report_lines.append(
            f"| {out_name} | {c['idx']} | {len(c['doc_texts'])} "
            f"| {c['total_tokens']} | {c['question'][:50]} | {ans_preview} |"
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
