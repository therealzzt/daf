#!/bin/bash
# =============================================================================
# 长上下文实验（6k–10k 档）幂等续跑脚本
# 位置：experiments/scripts/（2026-09-16 从仓库根目录整理迁入）
# 用法（在任意目录均可，脚本自行 cd 到仓库根）：
#   nohup bash experiments/scripts/run_long6k10k_resume.sh \
#       > experiments/logs/long6k10k/resume.out 2>&1 &
# 特性：
#   - 每轮 24 个样本，日志里 "===== " 出现 24 次即视为该轮完整跑完
#   - 已完整的轮次自动跳过，不完整的轮次整轮重跑（贪心解码，重跑结果一致）
#   - nohup 启动后可安全断开 SSH，脚本会跑到全部完成
# 完成后评分（在仓库根目录执行）：
#   python experiments/scripts/score_em_long_6k10k.py \
#       --logs experiments/logs/long6k10k/run_cb_long6k10k_r*.log \
#              experiments/logs/long6k10k/run_daf_long6k10k_r*.log
# =============================================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cacheblend
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # 仓库根目录：按本脚本位置（experiments/scripts/ 的上两级）自动定位，媒体盘挂载点变化也不受影响
export HF_ENDPOINT=https://hf-mirror.com

EXPECT_SAMPLES=24
LOG_DIR=experiments/logs/long6k10k

round_complete() {
    local log=$1
    [ -f "$log" ] || return 1
    local n
    n=$(grep -c '^===== ' "$log" 2>/dev/null || echo 0)
    [ "$n" -ge "$EXPECT_SAMPLES" ]
}

for round in 1 2 3; do
    log=$LOG_DIR/run_cb_long6k10k_r${round}.log
    if round_complete "$log"; then
        echo "[跳过] $log 已完整（24 样本）"
    else
        echo "[开跑] DAF=0 第 ${round} 轮 → $log"
        BLEND_INPUT_DIR=experiments/inputs_long_6k10k_official DAF=0 \
            python example/blend.py > "$log" 2>&1
        echo "[完成] $log exit=$?"
    fi
done

for round in 1 2 3; do
    log=$LOG_DIR/run_daf_long6k10k_r${round}.log
    if round_complete "$log"; then
        echo "[跳过] $log 已完整（24 样本）"
    else
        echo "[开跑] DAF=1 第 ${round} 轮 → $log"
        BLEND_INPUT_DIR=experiments/inputs_long_6k10k_official DAF=1 \
            python example/blend.py > "$log" 2>&1
        echo "[完成] $log exit=$?"
    fi
done

echo "全部轮次结束"
