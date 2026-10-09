#!/bin/bash
# =============================================================================
# RULER 30k 档 500 样本【FLASH_ATTN 后端】续跑脚本（daf1.6：kernel 红利测量）
# 位置：experiments/scripts/
#
# 由 run_ruler500_resume2.sh 复制而来，与 xformers 轮的唯一差异：
#   1. VLLM_ATTENTION_BACKEND=FLASH_ATTN（换 flash kernel，其余配置逐项对齐）
#   2. 日志目录 experiments/logs/qwen_ruler30k_500_flash/，文件名加 flash_ 前缀
#   3. CB/DAF 两臂 util 统一 0.89（xformers 轮 CB 臂最初用过 0.90，后改 0.89；
#      util 只决定 KV 池大小不影响数值——报告里注明这个不对称即可）
#
# 目的：同样本同配置只换 kernel，测 DAF/CB 时长比从 xformers 轮的 0.946
# 能降多少（论文在 A100+FlashAttention 上是 0.77）。
#
# 用法（4 组合各占 1 张卡）：
#   mkdir -p experiments/logs/qwen_ruler30k_500_flash
#   nohup bash experiments/scripts/run_ruler500_flash_resume.sh 0 qa_1 0 \
#       > experiments/logs/qwen_ruler30k_500_flash/resume_gpu0.out 2>&1 &
#   （其余三组合：1 qa_1 1 / 2 qa_2 0 / 3 qa_2 1）
#
# 完成后评分（RULER 必须 --raw_parse；文件名含 flash_daf 的判为 DAF）：
#   python experiments/scripts/score_em_long_6k10k.py \
#       --dir experiments/inputs_qwen_ruler30k_500 \
#       --tokenizer Qwen/Qwen2.5-7B-Instruct --raw_parse \
#       --logs experiments/logs/qwen_ruler30k_500_flash/run_*_qa1_*.log \
#       --out /tmp/scored500_flash_qa1.md   # qa_2 同理换 qa2
#
# 显存：flash 后端内存行为与 xformers 不同，若 0.89 OOM 用 CB_UTIL/DAF_UTIL
# 环境变量逐步降 0.88/0.87 并记录实际值（util 不影响数值结论）。
# =============================================================================
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cacheblend
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # 仓库根目录：按本脚本位置（experiments/scripts/ 的上两级）自动定位，媒体盘挂载点变化也不受影响
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASH_ATTN    # 本脚本与 xformers 轮的唯一后端差异

GPU=$1          # CUDA_VISIBLE_DEVICES
TASK=$2         # qa_1 | qa_2
DAF_FLAG=$3     # 0 | 1

TASK_SHORT=${TASK/_/}            # qa_1 -> qa1（日志文件名用）
CHUNK=50                         # 每片样本数
TOTAL=500
NCHUNK=$(( (TOTAL + CHUNK - 1) / CHUNK ))
MAX_PASS=3                       # 最多补跑趟数
LOG_DIR=experiments/logs/qwen_ruler30k_500_flash
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

# util：两臂统一 0.89（见文件头差异说明 3）；OOM 时用 CB_UTIL/DAF_UTIL 降档
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
        log=$LOG_DIR/run_flash_${METHOD}_chatml_qwen_ruler500_${TASK_SHORT}_c$(printf %02d $c).log
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
