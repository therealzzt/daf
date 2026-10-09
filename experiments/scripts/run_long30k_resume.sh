#!/bin/bash
# =============================================================================
# 长上下文实验（30k 档，RULER 风格埋答案样本）幂等续跑脚本
# 位置：experiments/scripts/（2026-09-16 从仓库根目录整理迁入）
# 用法：
#   nohup bash experiments/scripts/run_long30k_resume.sh \
#       > experiments/logs/long30k/resume.out 2>&1 &
# 特性：
#   - 每轮 12 个样本，日志里 "===== " 出现 12 次即视为该轮完整跑完
#   - 已完整的轮次自动跳过，不完整的整轮重跑（贪心解码，重跑结果一致）
#   - 30k 档必须用 AWQ 模型（BLEND_MODEL_PRESET=awq；bf16 的 KV 池装不下 32k 上下文）
#   - nohup 启动后可安全断开 SSH，脚本会跑到全部完成
# 完成后评分（在仓库根目录执行）：
#   python experiments/scripts/score_em_long_6k10k.py \
#       --dir experiments/inputs_long_30k_buriedqa \
#       --logs experiments/logs/long30k/run_cb_long30k_r*.log \
#              experiments/logs/long30k/run_daf_long30k_r*.log \
#       --out experiments/reports/em_report_long_30k.md
# =============================================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cacheblend
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # 仓库根目录：按本脚本位置（experiments/scripts/ 的上两级）自动定位，媒体盘挂载点变化也不受影响
export HF_ENDPOINT=https://hf-mirror.com

EXPECT_SAMPLES=12
LOG_DIR=experiments/logs/long30k

round_complete() {
    local log=$1
    [ -f "$log" ] || return 1
    local n
    n=$(grep -c '^===== ' "$log" 2>/dev/null || echo 0)
    [ "$n" -ge "$EXPECT_SAMPLES" ]
}

for round in 1 2 3; do
    log=$LOG_DIR/run_cb_long30k_r${round}.log
    if round_complete "$log"; then
        echo "[跳过] $log 已完整（${EXPECT_SAMPLES} 样本）"
    else
        echo "[开跑] DAF=0 第 ${round} 轮 → $log"
        BLEND_INPUT_DIR=experiments/inputs_long_30k_buriedqa \
            BLEND_MODEL_PRESET=awq DAF=0 \
            python example/blend.py > "$log" 2>&1
        echo "[完成] $log exit=$?"
    fi
done

for round in 1 2 3; do
    log=$LOG_DIR/run_daf_long30k_r${round}.log
    if round_complete "$log"; then
        echo "[跳过] $log 已完整（${EXPECT_SAMPLES} 样本）"
    else
        echo "[开跑] DAF=1 第 ${round} 轮 → $log"
        BLEND_INPUT_DIR=experiments/inputs_long_30k_buriedqa \
            BLEND_MODEL_PRESET=awq DAF=1 \
            python example/blend.py > "$log" 2>&1
        echo "[完成] $log exit=$?"
    fi
done

echo "全部轮次结束"
