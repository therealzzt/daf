# =============================================================================
# 生成长上下文实验样本（第一档：6k–10k，来源：仓库自带官方数据集文件）
#
# 数据来源：inputs/musique_s.json（多跳QA）、inputs/wikimqa_s.json（维基QA）、
#           inputs/samsum.json（对话摘要）——CacheBlend 仓库自带的官方样本
# 挑选规则：用 Mistral-7B-Instruct-v0.2 分词器统计「全部文档块 + 问题」的总 token 数，
#           只保留总长落在 [6000, 10000] 区间的样本
# 输出：experiments/inputs_long_6k10k_official/ 目录（须从仓库根目录运行），每个样本一个 json，文件名格式
#      {数据集}_{原文件内序号:03d}_len{总token数}.json，一眼看出出处和长度
#      另附 _selection_report.md 记录挑选明细（选了哪些、跳过哪些、长度分布）
#
# 生成的 json 字段（与 example/blend.py 的 inputs/*.json 格式完全兼容）：
#   chunk_num            文档块数
#   "0".."chunk_num-1"   每块文档文本（title\n\ntext，沿用 blend_musique.py 的拼法）
#   query                问题（带"5词以内直接作答"指令，沿用 blend_musique.py 的问法）
#   gold                 标准答案列表（wikimqa 的嵌套列表已拍平）
#   source_dataset       来源数据集文件名（溯源用）
#   source_index         在来源文件中的原始下标（溯源用）
#   total_tokens         总 token 数（挑选依据，方便核对）
# =============================================================================

import json
import os
import re
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "example"))
from utils import normalize_question  # noqa: E402

# ---- 参数（命名即含义） ----
TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # 与实验所用模型同款分词器
MIN_TOTAL_TOKENS = 6000    # 样本总 token 下限（含问题）
MAX_TOTAL_TOKENS = 10000   # 样本总 token 上限（含问题）
MAX_SAMPLES_PER_DATASET = 8   # 每个数据集最多挑几个（保证三数据集都有代表、控制总时长）
SOURCE_FILES = [
    ("inputs/musique_s.json", "musique_s"),
    ("inputs/wikimqa_s.json", "wikimqa_s"),
    ("inputs/samsum.json", "samsum"),
]
OUTPUT_DIR = "experiments/inputs_long_6k10k_official"

# 问题前缀/后缀：与 example/blend_musique.py 的 query_prompt 保持一致，
# 让模型"直接作答、5词以内"，EM 判分才可行
QUERY_INSTRUCTION = (
    "\n\nAnswer the question directly based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words. "
    "\nQuestion:"
)


def flatten_answers(answers):
    """wikimqa 的 answers 是 [['Ozalj']] 这种嵌套列表，拍平成字符串列表"""
    flat = []
    for a in answers:
        if isinstance(a, list):
            flat.extend(str(x) for x in a)
        else:
            flat.append(str(a))
    return flat


def count_sample_tokens(doc_texts, query_text, tokenizer):
    """总 token 数 = 各文档块 + 问题（去 BOS，与 blend.py 的统计口径一致）"""
    total = sum(len(tokenizer.encode(t)[1:]) for t in doc_texts)
    total += len(tokenizer.encode(query_text)[1:])
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
        "# 6k–10k 长上下文样本挑选报告",
        "",
        f"- 分词器：`{TOKENIZER_NAME}`",
        f"- 挑选区间：总 token 数 [{MIN_TOTAL_TOKENS}, {MAX_TOTAL_TOKENS}]（含问题）",
        f"- 每个数据集最多取 {MAX_SAMPLES_PER_DATASET} 个（按长度均匀抽样）",
        f"- 文档块拼法：`title\\n\\ntext`；问题拼法：与 `example/blend_musique.py` 的指令一致",
        "",
        "## 入选样本",
        "",
        "| 输出文件 | 来源 | 原下标 | 文档块数 | 总token数 | 问题 | 标准答案 |",
        "|---|---|---|---|---|---|---|",
    ]

    grand_total = 0
    for path, stem in SOURCE_FILES:
        data = json.load(open(path))
        candidates = []
        skipped_short, skipped_long, skipped_bad = 0, 0, 0
        for idx, ex in enumerate(data):
            ctxs = ex.get("ctxs") or []
            doc_texts = [f"{c.get('title','')}\n\n{c.get('text','')}\n\n" for c in ctxs]
            doc_texts = [t for t in doc_texts if t.strip()]
            if not doc_texts or not ex.get("question") or not ex.get("answers"):
                skipped_bad += 1
                continue
            query = QUERY_INSTRUCTION + normalize_question(ex["question"]) + "\nAnswer:"
            total = count_sample_tokens(doc_texts, query, tokenizer)
            if total < MIN_TOTAL_TOKENS:
                skipped_short += 1
                continue
            if total > MAX_TOTAL_TOKENS:
                skipped_long += 1
                continue
            candidates.append({
                "idx": idx, "doc_texts": doc_texts, "query": query,
                "gold": flatten_answers(ex["answers"]),
                "total_tokens": total, "question": ex["question"],
            })

        picked = pick_evenly(candidates, MAX_SAMPLES_PER_DATASET)
        for c in picked:
            out_name = f"{stem}_{c['idx']:03d}_len{c['total_tokens']}.json"
            out = {
                "chunk_num": len(c["doc_texts"]),
                "query": c["query"],
                "gold": c["gold"],
                "source_dataset": path,
                "source_index": c["idx"],
                "total_tokens": c["total_tokens"],
            }
            for i, t in enumerate(c["doc_texts"]):
                out[str(i)] = t
            with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
                json.dump(out, f, ensure_ascii=False)
            ans_preview = "; ".join(c["gold"])[:60]
            report_lines.append(
                f"| {out_name} | {os.path.basename(path)} | {c['idx']} "
                f"| {len(c['doc_texts'])} | {c['total_tokens']} "
                f"| {c['question'][:50]} | {ans_preview} |"
            )
        grand_total += len(picked)
        report_lines += [
            "",
            f"### {os.path.basename(path)}：候选 {len(candidates)} 个"
            f"（{len(picked)} 入选 / 低于下限 {skipped_short} / 超上限 {skipped_long}"
            f" / 缺字段 {skipped_bad}）",
            "",
        ]
        # 长度分布概览
        if candidates:
            lens = sorted(c["total_tokens"] for c in candidates)
            report_lines.append(
                f"候选长度分布：min={lens[0]}, 中位={lens[len(lens)//2]}, max={lens[-1]}"
            )
            report_lines.append("")

    report_lines.insert(4, f"- 入选总数：**{grand_total}** 个")
    with open(os.path.join(OUTPUT_DIR, "_selection_report.md"), "w") as f:
        f.write("\n".join(report_lines))
    print(f"完成：{grand_total} 个样本写入 {OUTPUT_DIR}/，明细见 _selection_report.md")


if __name__ == "__main__":
    main()
