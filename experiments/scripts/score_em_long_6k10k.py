# =============================================================================
# 长上下文实验（6k–10k 档）EM/F1 评分脚本
#
# 输入（须从仓库根目录运行）：
#   --dir   样本目录（默认 experiments/inputs_long_6k10k_official/，json 里带 gold 标准答案）
#   --logs  一个或多个 blend.py 运行日志（按 "===== 样本文件名 =====" 切分样本结果）
#   --out   输出报告路径（默认 experiments/reports/em_report_long_6k10k.md）
#
# 评分规则（沿用 SQuAD 惯例，复用 example/utils.py 的归一化函数）：
#   EM  = 归一化后与任一标准答案完全一致 → 1，否则 0（多答案取最大值）
#   F1  = 按 token 集合算 precision/recall 的调和平均（多答案取最大值）
#   归一化 = 小写 + 去冠词(a/an/the) + 去标点 + 空白规整，Yes/No 开头归一
# 报告内容：每样本×每轮明细 + 按方法聚合（EM/F1/TTFT 均值）+ 跨轮中位数
# =============================================================================

import argparse
import glob
import json
import os
import re
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "example"))
from utils import compute_f1, normalize_answer, parse_generation  # noqa: E402

TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
SAMPLE_HEADER = re.compile(r"^===== (\S+\.json) =====$")
CACHED_RE = re.compile(r"^Cached generation: (.*)$")
NORMAL_RE = re.compile(r"^Normal generation: (.*)$")
TTFT_CACHE_RE = re.compile(r"^TTFT with cache: ([\d.eE+-]+)")
TTFT_FULL_RE = re.compile(r"^TTFT with full prefill: ([\d.eE+-]+)")


def exact_match(pred, golds):
    """归一化后与任一 gold 完全一致即 1（多答案取最大）。
    与 LongBench 官方口径一致：小写+去冠词+去标点+空白规整
    （example/utils.py 的 normalize_answer 即此实现）"""
    pred_n = normalize_answer(parse_generation(pred))
    return max(int(pred_n == normalize_answer(g)) for g in golds)


def substring_match(pred, golds):
    """子串匹配（RULER 官方判分风格）：归一化后 gold 与 pred 互为子串即 1。
    宽容 EM：给"Charles university in Prague"（gold: Charles University）、
    "Goring-on-Thames"（gold: Goring-on-Thames, Oxfordshire）这类正确但带补充的
    回答记分。多答案取最大。"""
    pred_n = normalize_answer(parse_generation(pred))
    if not pred_n:
        return 0
    for g in golds:
        g_n = normalize_answer(g)
        if g_n and (pred_n in g_n or g_n in pred_n):
            return 1
    return 0


def parse_log(path):
    """把一份运行日志解析成 {样本文件名: {cached, normal, ttft_cache, ttft_full}}"""
    results = {}
    cur = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = SAMPLE_HEADER.match(line)
            if m:
                cur = m.group(1)
                results[cur] = {}
                continue
            if cur is None:
                continue
            for regex, key in [(CACHED_RE, "cached"), (NORMAL_RE, "normal"),
                               (TTFT_CACHE_RE, "ttft_cache"),
                               (TTFT_FULL_RE, "ttft_full")]:
                m = regex.match(line)
                if m:
                    results[cur][key] = float(m.group(1)) if key.startswith("ttft") else m.group(1)
    return results


def median(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return float("nan")
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/inputs_long_6k10k_official")
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/reports/em_report_long_6k10k.md")
    ap.add_argument("--tokenizer", default=TOKENIZER_NAME,
                    help="F1 分词用的 tokenizer（默认 Mistral；qwen 实验传 "
                         "Qwen/Qwen2.5-7B-Instruct）")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # F1 分词的 BOS 处理：utils.compute_f1 写死 [1:] 剥首 token（Mistral 有 BOS）。
    # Qwen 等不加 BOS 的 tokenizer 不能剥，否则 F1 少算一个 token。
    _probe = tokenizer.encode("a")
    _probe_nospecial = tokenizer.encode("a", add_special_tokens=False)
    _strip_first = len(_probe) > len(_probe_nospecial)

    def compute_f1_tok(pred, gold):
        import collections
        a_pred = parse_generation(pred)
        gold_toks = tokenizer.encode(normalize_answer(gold))
        pred_toks = tokenizer.encode(normalize_answer(a_pred))
        if _strip_first:
            gold_toks, pred_toks = gold_toks[1:], pred_toks[1:]
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        return (2 * precision * recall) / (precision + recall)

    golds = {}
    for path in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        ex = json.load(open(path))
        name = os.path.basename(path)
        golds[name] = {"gold": ex["gold"], "tokens": ex.get("total_tokens"),
                       "source": os.path.basename(ex.get("source_dataset", "?"))}

    runs = {os.path.basename(p): parse_log(p) for p in args.logs}

    lines = [
        "# 6k–10k 长上下文实验评分报告",
        "",
        f"- 样本目录：`{args.dir}`（{len(golds)} 个样本）",
        f"- 评分：EM（精确匹配，多答案取最大）+ token-F1（多答案取最大）",
        f"- 归一化：小写/去冠词/去标点/Yes-No 规整（`example/utils.py`）",
        f"- 输入日志：{', '.join('`'+os.path.basename(p)+'`' for p in args.logs)}",
        "",
        "## 一、逐样本明细（质量指标为多轮均值，TTFT 为多轮中位数）",
        "",
        "| 样本 | 来源 | 长度 | 方法 | EM | SM | F1 | TTFT缓存(s) | TTFT全量(s) | 缓存答案 | 全量答案 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    # 方法识别：日志名含 daf → DAF，含 cb → CacheBlend
    def method_of(log_name):
        return "DAF" if "daf" in log_name.lower() else "CacheBlend"

    # 汇总表: {方法: {指标: [每轮值]}}
    agg = {}
    per_sample = {name: {} for name in golds}

    for log_name, res in runs.items():
        meth = method_of(log_name)
        agg.setdefault(meth, {"em": [], "sm": [], "f1": [], "ttft_c": [], "ttft_f": []})
        ems, sms, f1s = [], [], []
        for name in sorted(golds):
            r = res.get(name)
            if not r or "cached" not in r:
                continue
            em = exact_match(r["cached"], golds[name]["gold"])
            sm = substring_match(r["cached"], golds[name]["gold"])
            f1 = max(compute_f1_tok(r["cached"], g) for g in golds[name]["gold"])
            ems.append(em)
            sms.append(sm)
            f1s.append(f1)
            per_sample[name].setdefault(meth, {"em": [], "sm": [], "f1": [], "ttft_c": [], "ttft_f": []})
            per_sample[name][meth]["em"].append(em)
            per_sample[name][meth]["sm"].append(sm)
            per_sample[name][meth]["f1"].append(f1)
            per_sample[name][meth]["ttft_c"].append(r.get("ttft_cache"))
            per_sample[name][meth]["ttft_f"].append(r.get("ttft_full"))
        agg[meth]["em"].append(sum(ems) / len(ems) if ems else float("nan"))
        agg[meth]["sm"].append(sum(sms) / len(sms) if sms else float("nan"))
        agg[meth]["f1"].append(sum(f1s) / len(f1s) if f1s else float("nan"))
        round_ttc = [res[n]["ttft_cache"] for n in res if "ttft_cache" in res[n]]
        round_ttf = [res[n]["ttft_full"] for n in res if "ttft_full" in res[n]]
        agg[meth]["ttft_c"].append(sum(round_ttc) / len(round_ttc) if round_ttc else float("nan"))
        agg[meth]["ttft_f"].append(sum(round_ttf) / len(round_ttf) if round_ttf else float("nan"))

    # 明细行：每样本×方法，EM/SM/F1 为各轮均值，TTFT 为中位数
    for name in sorted(golds):
        for meth in ("CacheBlend", "DAF"):
            d = per_sample[name].get(meth)
            if not d:
                continue
            em = sum(d["em"]) / len(d["em"])
            sm = sum(d["sm"]) / len(d["sm"])
            f1 = sum(d["f1"]) / len(d["f1"])
            ttc = median([x for x in d["ttft_c"] if x is not None])
            ttf = median([x for x in d["ttft_f"] if x is not None])
            # 取该方法第一轮日志里的答案文本展示
            ans_c, ans_f = "", ""
            for log_name, res in runs.items():
                if method_of(log_name) == meth and name in res and "cached" in res[name]:
                    ans_c = res[name]["cached"][:40]
                    ans_f = res[name].get("normal", "")[:40]
                    break
            lines.append(
                f"| {name} | {golds[name]['source']} | {golds[name]['tokens']} | {meth} "
                f"| {em:.2f} | {sm:.2f} | {f1:.2f} | {ttc:.3f} | {ttf:.3f} | {ans_c} | {ans_f} |"
            )

    # vLLM 全量基线：每份日志的 Normal generation 行就是一次全量 prefill 跑批，
    # 与缓存方法同模型同 prompt，是第三方法的精度/TTFT 来源
    full_agg = {"em": [], "sm": [], "f1": [], "ttft": []}
    for log_name, res in runs.items():
        ems, sms, f1s, ttfs = [], [], [], []
        for name in sorted(golds):
            r = res.get(name)
            if not r or "normal" not in r:
                continue
            ems.append(exact_match(r["normal"], golds[name]["gold"]))
            sms.append(substring_match(r["normal"], golds[name]["gold"]))
            f1s.append(max(compute_f1_tok(r["normal"], g) for g in golds[name]["gold"]))
            if "ttft_full" in r:
                ttfs.append(r["ttft_full"])
        if ems:
            full_agg["em"].append(sum(ems) / len(ems))
            full_agg["sm"].append(sum(sms) / len(sms))
            full_agg["f1"].append(sum(f1s) / len(f1s))
            full_agg["ttft"].append(sum(ttfs) / len(ttfs) if ttfs else float("nan"))

    lines += ["", "## 二、按方法聚合（每轮一行，再给出跨轮中位数）", "",
              "| 方法 | 轮次 | EM | SM | F1 | TTFT缓存均值(s) | TTFT全量均值(s) | 缓存加速比 |",
              "|---|---|---|---|---|---|---|---|"]
    for meth in ("CacheBlend", "DAF"):
        a = agg.get(meth)
        if not a:
            continue
        n_rounds = len(a["em"])
        for i in range(n_rounds):
            speed = a["ttft_f"][i] / a["ttft_c"][i] if a["ttft_c"][i] else float("nan")
            lines.append(
                f"| {meth} | r{i+1} | {a['em'][i]:.3f} | {a['sm'][i]:.3f} | {a['f1'][i]:.3f} "
                f"| {a['ttft_c'][i]:.3f} | {a['ttft_f'][i]:.3f} | {speed:.2f}× |")
        em_med, sm_med, f1_med = median(a["em"]), median(a["sm"]), median(a["f1"])
        tc_med, tf_med = median(a["ttft_c"]), median(a["ttft_f"])
        lines.append(
            f"| **{meth}** | **中位** | **{em_med:.3f}** | **{sm_med:.3f}** | **{f1_med:.3f}** "
            f"| **{tc_med:.3f}** | **{tf_med:.3f}** | **{tf_med/tc_med:.2f}×** |")
    # vLLM 全量基线行：无缓存，TTFT 就是全量 prefill 时间，加速比恒 1
    if full_agg["em"]:
        for i in range(len(full_agg["em"])):
            lines.append(
                f"| vLLM全量 | r{i+1} | {full_agg['em'][i]:.3f} | {full_agg['sm'][i]:.3f} "
                f"| {full_agg['f1'][i]:.3f} | - | {full_agg['ttft'][i]:.3f} | 1.00× |")
        lines.append(
            f"| **vLLM全量** | **中位** | **{median(full_agg['em']):.3f}** "
            f"| **{median(full_agg['sm']):.3f}** | **{median(full_agg['f1']):.3f}** "
            f"| - | **{median(full_agg['ttft']):.3f}** | **1.00×** |")

    # 按数据集×方法聚合（per_sample 里已有多轮数据，取中位数）
    lines += ["", "## 三、按数据集×方法聚合（跨轮中位数；QA 数据集看 SM/F1，samsum 摘要看 F1）", "",
              "| 数据集 | 样本数 | 方法 | EM中位 | SM中位 | F1中位 | TTFT缓存中位(s) |",
              "|---|---|---|---|---|---|---|"]
    sources = sorted({golds[n]["source"] for n in golds})
    for src in sources:
        names = [n for n in sorted(golds) if golds[n]["source"] == src]
        for meth in ("CacheBlend", "DAF"):
            ems, sms, f1s, ttcs = [], [], [], []
            for n in names:
                d = per_sample[n].get(meth)
                if not d:
                    continue
                ems.append(sum(d["em"]) / len(d["em"]))
                sms.append(sum(d["sm"]) / len(d["sm"]))
                f1s.append(sum(d["f1"]) / len(d["f1"]))
                ttcs.extend(x for x in d["ttft_c"] if x is not None)
            if not ems:
                continue
            lines.append(
                f"| {src} | {len(names)} | {meth} | {median(ems):.3f} | {median(sms):.3f} "
                f"| {median(f1s):.3f} | {median(ttcs):.3f} |")
    # 控制台也打印分组摘要
    print("--- 按数据集 ---")
    for line in lines[-7:]:
        print(line)

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"报告已写入 {args.out}")
    # 控制台同时打印聚合摘要，方便不打开文件
    for meth in ("CacheBlend", "DAF"):
        a = agg.get(meth)
        if a:
            print(f"{meth}: EM中位={median(a['em']):.3f} SM中位={median(a['sm']):.3f} "
                  f"F1中位={median(a['f1']):.3f} "
                  f"TTFT缓存中位={median(a['ttft_c']):.3f}s "
                  f"TTFT全量中位={median(a['ttft_f']):.3f}s "
                  f"加速比={median(a['ttft_f'])/median(a['ttft_c']):.2f}×")
    if full_agg["em"]:
        print(f"vLLM全量: EM中位={median(full_agg['em']):.3f} "
              f"SM中位={median(full_agg['sm']):.3f} F1中位={median(full_agg['f1']):.3f} "
              f"TTFT中位={median(full_agg['ttft']):.3f}s")


if __name__ == "__main__":
    main()
