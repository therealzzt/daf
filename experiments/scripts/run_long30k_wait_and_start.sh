#!/bin/bash
# =============================================================================
# 30k 档实验看门狗：等 GPU 0 空出足够显存后自动启动实验
# 位置：experiments/scripts/（2026-09-16 从仓库根目录整理迁入）
# 用法：
#   nohup bash experiments/scripts/run_long30k_wait_and_start.sh \
#       > experiments/logs/long30k/wait.out 2>&1 &
#
# 背景：2026-09-15 发现服务器 4 卡被 white 用户的 daf_reference.py 占满
# （每卡 21~24GB），我们的实验引擎起不来。本脚本每分钟检查一次 GPU 0 空闲
# 显存，够 23GB 时启动 run_long30k_resume.sh（幂等续跑，跑完自动停）。
# 最长等待 24 小时，超时自动放弃并在日志里说明。
# =============================================================================
NEED_FREE_MIB=23000   # 峰值需求 ~22.4GB（vllm 18 + old_kvs 3.9 + 零碎），留 0.6GB 安全余量
MAX_WAIT_MIN=1440   # 24 小时
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # 仓库根目录：按本脚本位置（experiments/scripts/ 的上两级）自动定位，媒体盘挂载点变化也不受影响
LOG="$REPO_ROOT/experiments/logs/long30k/wait.out"

free_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0
}

for ((i=0; i<MAX_WAIT_MIN; i++)); do
    free=$(free_mib)
    if [ "$free" -ge "$NEED_FREE_MIB" ]; then
        echo "[$(date '+%F %T')] GPU 0 空闲 ${free} MiB ≥ ${NEED_FREE_MIB}，启动 30k 实验" >> "$LOG"
        exec bash "$REPO_ROOT/experiments/scripts/run_long30k_resume.sh" >> "$LOG" 2>&1
    fi
    sleep 60
done
echo "[$(date '+%F %T')] 等待超时（${MAX_WAIT_MIN} 分钟），GPU 仍未空出，放弃自动启动" >> "$LOG"
