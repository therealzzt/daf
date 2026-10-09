# =============================================================================
# 生成长上下文实验样本（第二档：30k，RULER 风格"埋答案"构造）
#
# 构造方法（与 RULER qa1/qa2 同思路，素材全部来自仓库自带数据集，零外网）：
#   1. 选种子题：wikimqa_s/musique_s 里带标准答案的 QA 样本（排除 6k–10k 档
#      已用的 24 个，保证两档独立）
#   2. 种子题自己的 10 篇文档 = 「答案文档块」（含答题所需信息）
#   3. 从其他样本抽文档当「干扰段落」，凑到总长约 30k token
#   4. 把答案文档块埋在干扰段落池的 20% / 50% / 80% 深度处（4 题 × 3 深度 = 12 个样本）
#      —— 模拟 RULER 检验"从长上下文任意位置检索答案"的能力
#
# 输出：experiments/inputs_long_30k_buriedqa/（须从仓库根目录运行），
#      文件名 {种子数据集}_{种子下标}_at{深度}_len{token数}.json
#      字段与 blend.py 格式兼容（chunk_num / "0".. / query / gold / 溯源字段）
# =============================================================================

import json
import os
import random
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "example"))
from utils import normalize_question  # noqa: E402

TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MIN_TOTAL_TOKENS = 28000
MAX_TOTAL_TOKENS = 31500
SEEDS_PER_DATASET = 2          # wikimqa 2 题 + musique 2 题 = 4 题
DEPTHS = [0.2, 0.5, 0.8]       # 答案块埋入深度（按干扰段 token 占比）
EXCLUDE_LOG = "experiments/inputs_long_6k10k_official"   # 6k–10k 档已用样本（按 source_dataset+index 排除）
SOURCE_FILES = [("inputs/wikimqa_s.json", "wikimqa_s"),
                ("inputs/musique_s.json", "musique_s")]
OUTPUT_DIR = "experiments/inputs_long_30k_buriedqa"
QUERY_INSTRUCTION = (
    "\n\nAnswer the question directly based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words. "
    "\nQuestion:"
)
random.seed(42)   # 固定随机种子，重复生成结果一致


def flatten_answers(answers):
    flat = []
    for a in answers:
        if isinstance(a, list):
            flat.extend(str(x) for x in a)
        else:
            flat.append(str(a))
    return flat


def tok_len(text, tokenizer):
    return len(tokenizer.encode(text)[1:])


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 6k–10k 档用过的样本，排除掉
    used = set()
    for fn in os.listdir(EXCLUDE_LOG):
        if fn.endswith(".json") and not fn.startswith("_"):
            ex = json.load(open(os.path.join(EXCLUDE_LOG, fn)))
            used.add((os.path.basename(ex["source_dataset"]), ex["source_index"]))

    # 收集全部文档段（text）和候选种子
    distractor_pool = []          # [(来源, 下标, 段号, text)]
    seeds = []
    for path, stem in SOURCE_FILES:
        data = json.load(open(path))
        for idx, ex in enumerate(data):
            ctxs = [c.get("text", "") for c in (ex.get("ctxs") or [])]
            ctxs = [t for t in ctxs if t.strip()]
            if not ctxs:
                continue
            for ci, t in enumerate(ctxs):
                distractor_pool.append((stem, idx, ci, t))
            if ((stem, idx) not in used and ex.get("question") and ex.get("answers")
                    and len(ctxs) >= 5):
                seeds.append({"stem": stem, "path": path, "idx": idx, "ex": ex})

    # 每数据集按原顺序取前 N 个种子（稳定可复现）
    seeds_by_ds = {}
    for s in seeds:
        seeds_by_ds.setdefault(s["stem"], []).append(s)
    picked_seeds = []
    for stem, lst in seeds_by_ds.items():
        picked_seeds.extend(lst[:SEEDS_PER_DATASET])

    report = [
        "# 30k 长上下文样本构造报告（RULER 风格埋答案）",
        "",
        f"- 构造：种子题的 10 篇文档为答案块，干扰段来自其他样本的文档池，"
        f"答案块埋于干扰池的 {DEPTHS} 深度处",
        f"- 长度目标：[{MIN_TOTAL_TOKENS}, {MAX_TOTAL_TOKENS}] token（含问题）",
        f"- 种子：{len(picked_seeds)} 题 × {len(DEPTHS)} 深度 = "
        f"{len(picked_seeds) * len(DEPTHS)} 个样本；已排除 6k–10k 档用过的样本",
        f"- 随机种子 42（重复生成结果一致）",
        "",
        "| 输出文件 | 种子 | 埋深 | 块数 | 总token | 问题 | 标准答案 |",
        "|---|---|---|---|---|---|---|",
    ]

    total_written = 0
    for s in picked_seeds:
        ex = s["ex"]
        rel_docs = [f"{c.get('title','')}\n\n{c.get('text','')}\n\n"
                    for c in ex["ctxs"] if c.get("text", "").strip()]
        query = QUERY_INSTRUCTION + normalize_question(ex["question"]) + "\nAnswer:"
        rel_tokens = sum(tok_len(t, tokenizer) for t in rel_docs)

        for depth in DEPTHS:
            # 目标总长约 30k：干扰段 token = (目标 - 答案块 - 问题) 按深度切前后两段
            target = random.randint(MIN_TOTAL_TOKENS, MAX_TOTAL_TOKENS)
            budget_distract = target - rel_tokens - tok_len(query, tokenizer)
            # 干扰池不放种子题自己的段（避免答案信息泄漏到干扰段里）
            pool = [d for d in distractor_pool
                    if not (d[0] == s["stem"] and d[1] == s["idx"])]
            random.shuffle(pool)

            def take(upto):
                """从干扰池顺序取段，累计 token 不超过 upto"""
                docs, acc = [], 0
                for _, _, _, t in pool:
                    lt = tok_len(t, tokenizer)
                    if acc + lt > upto:
                        continue
                    docs.append(t)
                    acc += lt
                return docs, acc

            front, ftok = take(int(budget_distract * depth))
            back, btok = take(budget_distract - ftok)
            doc_prompts = [f"{t}\n\n" for t in front] + rel_docs + [f"{t}\n\n" for t in back]

            total = (ftok + btok + rel_tokens
                     + tok_len(query, tokenizer))
            depth_tag = f"{int(depth*100):03d}"
            name = f"{s['stem']}_{s['idx']:03d}_at{depth_tag}_len{total}.json"
            out = {
                "chunk_num": len(doc_prompts),
                "query": query,
                "gold": flatten_answers(ex["answers"]),
                "source_dataset": s["path"],
                "source_index": s["idx"],
                "buried_depth": depth,
                "total_tokens": total,
            }
            for i, t in enumerate(doc_prompts):
                out[str(i)] = t
            with open(os.path.join(OUTPUT_DIR, name), "w") as f:
                json.dump(out, f, ensure_ascii=False)
            total_written += 1
            ans = "; ".join(out["gold"])[:60]
            report.append(
                f"| {name} | {os.path.basename(s['path'])}#{s['idx']} | {depth} "
                f"| {len(doc_prompts)} | {total} | {ex['question'][:50]} | {ans} |"
            )

    report.insert(4, f"- 生成总数：**{total_written}**")
    with open(os.path.join(OUTPUT_DIR, "_selection_report.md"), "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"完成：{total_written} 个样本写入 {OUTPUT_DIR}/，明细见 _selection_report.md")


if __name__ == "__main__":
    main()
