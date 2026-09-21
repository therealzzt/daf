#!/bin/bash
# =============================================================================
# RULER 30k 档 500 样本放量（daf1.5 最终阶段）幂等续跑脚本
# 位置：experiments/scripts/
#
# 用法（4 个组合各占 1 张卡，并行启动 4 个实例）：
#   mkdir -p experiments/logs/qwen_ruler30k_500
#   nohup bash experiments/scripts/run_ruler500_resume.sh 0 qa_1 0 \
#       > experiments/logs/qwen_ruler30k_500/resume_gpu0.out 2>&1 &
#   nohup bash experiments/scripts/run_ruler500_resume.sh 1 qa_1 1 \
#       > experiments/logs/qwen_ruler30k_500/resume_gpu1.out 2>&1 &
#   nohup bash experiments/scripts/run_ruler500_resume.sh 2 qa_2 0 \
#       > experiments/logs/qwen_ruler30k_500/resume_gpu2.out 2>&1 &
#   nohup bash experiments/scripts/run_ruler500_resume.sh 3 qa_2 1 \
#       > experiments/logs/qwen_ruler30k_500/resume_gpu3.out 2>&1 &
#   参数：$1=GPU 编号  $2=任务（qa_1|qa_2）  $3=DAF 开关（0=CacheBlend 对照）
#
# 跑批协议（与 24 样本轮的差异及理由）：
#   - 只跑 1 轮全量 500 样本：贪心解码（temperature=0）下质量指标逐轮确定，
#     重复轮次不增加信息；TTFT 改用【跨 500 样本的中位数】，
#     比 24 样本 × 3 轮的逐样本中位数更稳健
#   - 500 样本切成 10 片（每片 50 个），每片一个独立进程/日志：
#     单进程 4+ 小时中途崩溃的代价从"整轮重跑"降到"重跑一片（~25 分钟）"；
#     分片由 blend.py 的 BLEND_SAMPLE_RANGE="start:end" 实现
#   - 已完整的片（日志里 "^===== " 出现 50 次）自动跳过，不完整的整片重跑
#
# 配置（与 K+V/ChatML 消融轮一致，仅 kv_dev 不开）：
#   BLEND_MODEL_PRESET=qwen BLEND_MAX_MODEL_LEN=32768 BLEND_CHATML=1
#   BLEND_MAX_TOKENS=32，util：CB=0.90 / DAF=0.89（util 只影响 KV 池大小），
#   后端默认 XFORMERS，V-only 打分（BLEND_KV_DEV 不传 = 关）
#
# 完成后评分（在仓库根目录执行；RULER 必须 --raw_parse）：
#   python experiments/scripts/score_em_long_6k10k.py \
#       --dir experiments/inputs_qwen_ruler30k_500 \
#       --tokenizer Qwen/Qwen2.5-7B-Instruct --raw_parse \
#       --logs experiments/logs/qwen_ruler30k_500/run_*_qa1_*.log \
#       --out /tmp/scored_qa1.md   # qa_2 同理换 qa2
# =============================================================================
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cacheblend
cd /home/therealzzt/daf
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU=$1          # CUDA_VISIBLE_DEVICES
TASK=$2         # qa_1 | qa_2
DAF_FLAG=$3     # 0 | 1

TASK_SHORT=${TASK/_/}            # qa_1 -> qa1（日志文件名用）
CHUNK=50                         # 每片样本数
TOTAL=500
NCHUNK=$(( (TOTAL + CHUNK - 1) / CHUNK ))
LOG_DIR=experiments/logs/qwen_ruler30k_500
# 样本大目录里 qa_1/qa_2 混在一起，按任务建软链暂存目录（幂等）
TASK_DIR=/tmp/blend_ruler500_${TASK}

mkdir -p "$LOG_DIR" "$TASK_DIR"
# 软链已存在则跳过（ln -sf 覆盖同名旧链，指向不变时无副作用）
for f in experiments/inputs_qwen_ruler30k_500/${TASK}_*.json; do
    ln -sf "$HOME/daf/$f" "$TASK_DIR/$(basename "$f")"
done
N_LINK=$(ls "$TASK_DIR" | wc -l)
if [ "$N_LINK" -ne "$TOTAL" ]; then
    echo "[错误] $TASK_DIR 只有 $N_LINK 个样本（期望 $TOTAL），先跑 make_inputs_qwen_ruler30k.py"
    exit 1
fi

# CB 对照臂 util 0.90，DAF 臂 0.89（KV 池差 ~3k token，均 >32768 上限）
if [ "$DAF_FLAG" = "1" ]; then
    METHOD=daf
    UTIL=0.89
else
    METHOD=cb
    UTIL=0.90
fi

chunk_complete() {
    local log=$1 expect=$2
    [ -f "$log" ] || return 1
    local n
    n=$(grep -c '^===== ' "$log" 2>/dev/null || echo 0)
    [ "$n" -ge "$expect" ]
}

for ((c=0; c<NCHUNK; c++)); do
    lo=$((c * CHUNK))
    hi=$((lo + CHUNK))
    [ $hi -gt $TOTAL ] && hi=$TOTAL
    expect=$((hi - lo))
    log=$LOG_DIR/run_${METHOD}_chatml_qwen_ruler500_${TASK_SHORT}_c$(printf %02d $c).log
    if chunk_complete "$log" "$expect"; then
        echo "[跳过] $log 已完整（$expect 样本）"
    else
        echo "[开跑] GPU$GPU $TASK DAF=$DAF_FLAG 第 $c 片（样本 $lo:$hi）→ $log"
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

echo "GPU$GPU $TASK DAF=$DAF_FLAG 全部 $NCHUNK 片结束"
