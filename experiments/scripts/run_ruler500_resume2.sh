#!/bin/bash
# =============================================================================
# RULER 30k 档 500 样本放量 续跑脚本 v2（daf1.5）
# 位置：experiments/scripts/
#
# 与 v1（run_ruler500_resume.sh）的差异（v2 取代 v1，新跑/补跑都用本脚本；
# v1 保留仅因有两个 2026-09-21 16:25 启动的实例正在增量读它，改文件会破坏运行中
# 的 bash；那两个实例跑完后如有残缺片，用本脚本补跑即可）：
#   1. CB 臂 util 0.90 → 0.89（可用 CB_UTIL 环境变量覆盖）：
#      v1 的 CB/0.90 配置在 50 样本长进程里碎片化累积，实测第 9~29 个样本处
#      224MB 分配 OOM（24 样本短进程没事）。util 只决定 KV 池大小
#      （0.89 池 ~3.7 万 token > max_model_len 32768），不影响任何数值结果；
#      与 DAF 臂同为 0.89 反而更对称
#   2. 单片失败后多轮补跑：最多 3 趟扫全部片，残缺片（"^===== " 计数不足）
#      自动重跑；v1 单趟扫过就完，残缺片要靠人工再启动一次
#
# 用法（与 v1 相同）：
#   nohup bash experiments/scripts/run_ruler500_resume2.sh <GPU> <qa_1|qa_2> <0|1> \
#       > experiments/logs/qwen_ruler30k_500/resume_gpu<N>.out 2>&1 &
# 配置、日志命名、评分命令均与 v1 相同（见 v1 文件头注释）。
# =============================================================================
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cacheblend
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # 仓库根目录：按本脚本位置（experiments/scripts/ 的上两级）自动定位，媒体盘挂载点变化也不受影响
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU=$1          # CUDA_VISIBLE_DEVICES
TASK=$2         # qa_1 | qa_2
DAF_FLAG=$3     # 0 | 1

TASK_SHORT=${TASK/_/}            # qa_1 -> qa1（日志文件名用）
CHUNK=50                         # 每片样本数
TOTAL=500
NCHUNK=$(( (TOTAL + CHUNK - 1) / CHUNK ))
MAX_PASS=3                       # 最多补跑趟数
LOG_DIR=experiments/logs/qwen_ruler30k_500
# 样本大目录里 qa_1/qa_2 混在一起，按任务建软链暂存目录（幂等）
TASK_DIR=/tmp/blend_ruler500_${TASK}

mkdir -p "$LOG_DIR" "$TASK_DIR"
for f in experiments/inputs_qwen_ruler30k_500/${TASK}_*.json; do
    ln -sf "$HOME/daf/$f" "$TASK_DIR/$(basename "$f")"
done
N_LINK=$(ls "$TASK_DIR" | wc -l)
if [ "$N_LINK" -ne "$TOTAL" ]; then
    echo "[错误] $TASK_DIR 只有 $N_LINK 个样本（期望 $TOTAL），先跑 make_inputs_qwen_ruler30k.py"
    exit 1
fi

# util：DAF 臂 0.89（历史配置）；CB 臂 0.89（v2 起，见文件头差异说明 1）
if [ "$DAF_FLAG" = "1" ]; then
    METHOD=daf
    UTIL=${DAF_UTIL:-0.89}
else
    METHOD=cb
    UTIL=${CB_UTIL:-0.89}
fi

chunk_complete() {
    local log=$1 expect=$2
    [ -f "$log" ] || return 1
    local n
    n=$(grep -c '^===== ' "$log" 2>/dev/null || echo 0)
    [ "$n" -ge "$expect" ]
}

for ((pass=1; pass<=MAX_PASS; pass++)); do
    all_done=1
    for ((c=0; c<NCHUNK; c++)); do
        lo=$((c * CHUNK))
        hi=$((lo + CHUNK))
        [ $hi -gt $TOTAL ] && hi=$TOTAL
        expect=$((hi - lo))
        log=$LOG_DIR/run_${METHOD}_chatml_qwen_ruler500_${TASK_SHORT}_c$(printf %02d $c).log
        if chunk_complete "$log" "$expect"; then
            echo "[跳过] $log 已完整（$expect 样本）"
        else
            all_done=0
            echo "[开跑] 第${pass}趟 GPU$GPU $TASK DAF=$DAF_FLAG 第 $c 片（样本 $lo:$hi）→ $log"
            CUDA_VISIBLE_DEVICES=$GPU \
            BLEND_INPUT_DIR=$TASK_DIR \
            BLEND_SAMPLE_RANGE="$lo:$hi" \
            BLEND_MODEL_PRESET=qwen \
            BLEND_GPU_MEM_UTIL=$UTIL \
            BLEND_MAX_MODEL_LEN=32768 \
            BLEND_MAX_TOKENS=32 \
            BLEND_CHATML=1 \
            DAF=$DAF_FLAG \
                python example/blend.py > "$log" 2>&1
            echo "[完成] $log exit=$?（完成度以 '===== ' 计数为准）"
        fi
    done
    if [ "$all_done" = "1" ]; then
        echo "GPU$GPU $TASK DAF=$DAF_FLAG 全部 $NCHUNK 片完整（第 $pass 趟确认）"
        exit 0
    fi
    echo "[第${pass}趟结束] 仍有残缺片，进入下一趟补跑"
done

echo "[警告] GPU$GPU $TASK DAF=$DAF_FLAG 跑满 $MAX_PASS 趟仍有残缺片，需人工排查"
exit 1
