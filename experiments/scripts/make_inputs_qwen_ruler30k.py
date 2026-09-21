# =============================================================================
# 生成 Qwen2.5 实验样本（目标 B 第三档：RULER 官方 qa_1/qa_2，32768 长度档）
#
# 数据来源：experiments/ruler_src/data_gen/{qa_1,qa_2}/validation.jsonl
#           （RULER 官方 scripts/data/synthetic/qa.py 生成：qa_1=squad、
#            qa_2=hotpotqa，--max_seq_length 32768 --tokens_to_generate 32
#            --num_samples 24 --random_seed 42，模板为官方 qa 模板）
#
# RULER qa 样本结构（实测生成结果）：
#   input = 指令行 + "\n\nThe following are given documents.\n\n"
#           + context（"Document 1:\n...\n\nDocument N:\n..."）
#           + "\n\n" + 指令行 + "\n\nQuestion: {query}"
#   answer_prefix = " Answer:"（官方 qa.py 已从 input 尾部剥出，评测时拼回）
#   outputs = gold 答案列表
#   官方生成上限 tokens_to_generate = 32（跑批时 BLEND_MAX_TOKENS=32）
#
# 切块策略：context 按 "Document N:" 边界切块（每块一个完整文档，保持原文）；
#           指令前缀（含 "The following are given documents."）拼在第 0 块开头；
#           query = 尾部指令 + "Question: {query}" + answer_prefix（" Answer:"），
#           直接触发生成，不再加额外内容
# 输出：experiments/inputs_qwen_ruler30k/（须从仓库根目录运行），文件名格式
#      qa_1_{index:03d}_len{总token数}.json，另附 _selection_report.md
#
# 生成的 json 字段与 make_inputs_qwen_samsum.py 一致（chunk_num / "0".."n" /
# query / gold / source_dataset / source_index / total_tokens）；
# source_dataset 末尾带 qa_1 / qa_2 以便评分脚本按任务聚合
# =============================================================================

import argparse
import json
import os
import re

from transformers import AutoTokenizer

# ---- 参数（命名即含义） ----
# 命令行参数（均有默认值，不传参时行为与 daf1.4 的 24 样本轮完全一致）：
#   --source-dir  RULER qa.py 生成的 jsonl 所在目录（其下 {qa_1,qa_2}/validation.jsonl）
#   --output-dir  blend 样本输出目录
#   --num-samples 仅写进报告头的说明文字（实际条数以 jsonl 行数为准）
# 500 样本放量用法（daf1.5）：
#   python experiments/scripts/make_inputs_qwen_ruler30k.py \
#       --source-dir experiments/ruler_src/data_gen_500 \
#       --output-dir experiments/inputs_qwen_ruler30k_500 --num-samples 500
_parser = argparse.ArgumentParser()
_parser.add_argument("--source-dir", default="experiments/ruler_src/data_gen")
_parser.add_argument("--output-dir", default="experiments/inputs_qwen_ruler30k")
_parser.add_argument("--num-samples", type=int, default=24)
_args = _parser.parse_args()

TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 与实验所用模型同款分词器
SOURCE_DIR = _args.source_dir
TASKS = ["qa_1", "qa_2"]                     # qa_1=squad, qa_2=hotpotqa
OUTPUT_DIR = _args.output_dir

# RULER 官方 qa 模板（constants.py TASKS['qa']），指令行在 context 前出现一次、
# 在 Question 前再出现一次
INSTRUCTION = ("Answer the question based on the given documents. "
               "Only give me the answer and do not output any other words.")
PROMPT_PREFIX = INSTRUCTION + "\n\nThe following are given documents.\n\n"
QUESTION_MARKER = "\n\n" + INSTRUCTION + "\n\nQuestion: "

# 文档块边界："Document N:"（RULER qa.py DOCUMENT_PROMPT = "Document {i}:\n{document}"）
SEGMENT_BOUNDARY = re.compile(r"(?=Document \d+:)")


def split_sample(input_text, answer_prefix):
    """把 RULER qa 的 input 拆成 (文档块列表, query)。
    文档块保持官方版式；指令前缀拼回第 0 块开头；query 尾部接 answer_prefix。"""
    assert input_text.startswith(PROMPT_PREFIX), "input 不以官方指令前缀开头"
    body = input_text[len(PROMPT_PREFIX):]
    idx = body.rfind(QUESTION_MARKER)
    assert idx > 0, "找不到 Question 标记"
    context, question = body[:idx], body[idx:]
    doc_texts = [p for p in SEGMENT_BOUNDARY.split(context) if p.strip()]
    assert len(doc_texts) > 0, "未切出任何文档块"
    doc_texts[0] = PROMPT_PREFIX + doc_texts[0]
    query = question + answer_prefix  # "Question: {query} Answer:"
    return doc_texts, query


def count_sample_tokens(doc_texts, query_text, tokenizer):
    """总 token 数 = 各文档块 + 问题。Qwen tokenizer 不加 BOS，无需剥除"""
    total = sum(len(tokenizer.encode(t)) for t in doc_texts)
    total += len(tokenizer.encode(query_text))
    return total


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_lines = [
        "# RULER qa_1/qa_2（Qwen2.5，32768 档）样本转换报告",
        "",
        f"- 分词器：`{TOKENIZER_NAME}`",
        f"- 数据来源：`{SOURCE_DIR}/{{qa_1,qa_2}}/validation.jsonl`"
        f"（RULER 官方 qa.py 生成，--max_seq_length 32768，seed 42，"
        f"各 {_args.num_samples} 条）",
        "- qa_1 = squad（单文档问答，干扰文档为随机 squad 段落）",
        "- qa_2 = hotpotqa（多跳问答，干扰文档为 hotpotqa distractor）",
        "- 结构：context 按 `Document N:` 切块；指令前缀拼在第 0 块开头；"
        "query = 尾部指令 + `Question: {query} Answer:`",
        "- 评分主指标：SM（RULER string_match，答案列表任一命中即对），EM 记录",
        "",
        "## 入选样本",
        "",
        "| 输出文件 | 任务 | 原行号 | 文档块数 | 总token数 | gold（截断） |",
        "|---|---|---|---|---|---|",
    ]

    n_out = 0
    for task in TASKS:
        src = os.path.join(SOURCE_DIR, task, "validation.jsonl")
        with open(src) as f:
            for idx, line in enumerate(f):
                ex = json.loads(line)
                doc_texts, query = split_sample(ex["input"], ex["answer_prefix"])
                total = count_sample_tokens(doc_texts, query, tokenizer)
                gold = [str(a) for a in ex["outputs"]]
                out_name = f"{task}_{idx:03d}_len{total}.json"
                out = {
                    "chunk_num": len(doc_texts),
                    "query": query,
                    "gold": gold,
                    "source_dataset": f"{src} ({task})",
                    "source_index": idx,
                    "total_tokens": total,
                }
                for i, t in enumerate(doc_texts):
                    out[str(i)] = t
                with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
                    json.dump(out, f, ensure_ascii=False)
                report_lines.append(
                    f"| {out_name} | {task} | {idx} | {len(doc_texts)} "
                    f"| {total} | {gold[0][:60]} |"
                )
                n_out += 1

        # 每任务长度分布
        lens = sorted(
            int(fn.rsplit("_len", 1)[1].removesuffix(".json"))
            for fn in os.listdir(OUTPUT_DIR)
            if fn.startswith(f"{task}_") and fn.endswith(".json")
        )
        report_lines += [
            "",
            f"{task} 长度分布：min={lens[0]}, 中位={lens[len(lens)//2]}, "
            f"max={lens[-1]}, 均值={sum(lens)/len(lens):.0f}",
            "",
        ]

    with open(os.path.join(OUTPUT_DIR, "_selection_report.md"), "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"完成：{n_out} 个样本写入 {OUTPUT_DIR}/，明细见 _selection_report.md")


if __name__ == "__main__":
    main()
