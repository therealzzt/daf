#!/bin/bash
# =============================================================================
# 路线 A 排雷实验 续跑脚本：原生新版 vLLM 跑 RULER 30k 全量基线（两臂）
# 位置：experiments/scripts/
#
# 用法：
#   nohup bash experiments/scripts/run_newvllm_baseline_resume.sh <GPU> <qa_1|qa_2> <32|128> \
#       > experiments/logs/newvllm_baseline/resume_<...>.out 2>&1 &
#   参数：$1=GPU  $2=任务  $3=max_tokens（32=臂1 隔离引擎因素；128=臂2 隔离截断因素）
#
# 特性：
#   - 500 样本切 5 片 × 100（NEWVLLM_RANGE），按 "^===== " 计数幂等续跑
#   - 用 vllm_new 环境的 python（绝不动 cacheblend 环境）
#   - NEWVLLM_GPU_UTIL 可调（被占卡上默认 0.90 会撞显存，用 0.70 试试）
# 评分（RULER 必须 --raw_parse；日志名不含 daf 会被标成 CacheBlend，报告里改标）：
#   python experiments/scripts/score_em_long_6k10k.py \
#       --dir experiments/inputs_qwen_ruler30k_500 \
#       --tokenizer Qwen/Qwen2.5-7B-Instruct --raw_parse \
#       --logs experiments/logs/newvllm_baseline/run_newvllm_mt32_qa1_c*.log \
#       --out /tmp/newvllm_mt32_qa1.md
# =============================================================================
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_new          # 注意：不是 cacheblend
cd /home/therealzzt/daf
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1          # 模型已在本地 HF 缓存，禁止联网
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# flashinfer 0.6.18 在本机会触发 JIT 编译：系统 nvcc 是 CUDA 12.0 太老，
# pip 版 CUDA 13.4 又与 flashinfer 自带 CCCL 头文件版本冲突，两条路都编不过。
# 试点验证：关掉 flashinfer 采样器（退回 PyTorch 原生）即可绕开 JIT，
# 精度路径不受影响（贪心解码）。
# 注意：vllm 0.30 已不认识 VLLM_ATTENTION_BACKEND（日志会报 Unknown env var），
# attention 走默认后端即可，别再加这个变量。
export VLLM_USE_FLASHINFER_SAMPLER=0

GPU=$1          # CUDA_VISIBLE_DEVICES
TASK=$2         # qa_1 | qa_2
MT=$3           # 32 | 128

TASK_SHORT=${TASK/_/}
CHUNK=100
TOTAL=500
NCHUNK=$(( (TOTAL + CHUNK - 1) / CHUNK ))
LOG_DIR=experiments/logs/newvllm_baseline
mkdir -p "$LOG_DIR"

chunk_complete() {
    local log=$1 expect=$2
    [ -f "$log" ] || return 1
    local n
    n=$(grep -c '^===== ' "$log" 2>/dev/null || echo 0)
    [ "$n" -ge "$expect" ]
}

# 清理孤儿 EngineCore（PPID=1 说明它的 python 父进程已死，
# 却继续占着显存——os._exit 强退的副作用；正常在跑的引擎父进程是活的，不会误杀）
clean_orphan_engines() {
    local p ppid
    for p in $(pgrep -f "VLLM::EngineCore" 2>/dev/null); do
        ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
        if [ "$ppid" = "1" ]; then
            echo "[清理] 孤儿 EngineCore pid=$p（占显存不放）"
            kill "$p" 2>/dev/null
        fi
    done
}

all_chunks_complete() {
    local c lo hi expect log
    for ((c=0; c<NCHUNK; c++)); do
        lo=$((c * CHUNK)); hi=$((lo + CHUNK)); [ $hi -gt $TOTAL ] && hi=$TOTAL
        log=$LOG_DIR/run_newvllm_mt${MT}_${TASK_SHORT}_c$(printf %02d $c).log
        chunk_complete "$log" $((hi - lo)) || return 1
    done
    return 0
}

# 外层重试：某片可能因其他进程临时占显存启动失败（vllm 0.30 会检查
# 空闲显存是否够 util×总量，不够直接拒启动）。分片幂等，重跑只补缺的。
MAX_PASS=5
for ((pass=1; pass<=MAX_PASS; pass++)); do
    if all_chunks_complete; then
        break
    fi
    [ $pass -gt 1 ] && { echo "[重试] 第 $pass 轮（有分片未完成）"; sleep 30; }
    for ((c=0; c<NCHUNK; c++)); do
        lo=$((c * CHUNK))
        hi=$((lo + CHUNK))
        [ $hi -gt $TOTAL ] && hi=$TOTAL
        expect=$((hi - lo))
        log=$LOG_DIR/run_newvllm_mt${MT}_${TASK_SHORT}_c$(printf %02d $c).log
        if chunk_complete "$log" "$expect"; then
            echo "[跳过] $log 已完整（$expect 样本）"
        else
            clean_orphan_engines
            echo "[开跑] GPU$GPU $TASK mt$MT 第 $c 片（样本 $lo:$hi）→ $log"
            CUDA_VISIBLE_DEVICES=$GPU \
            NEWVLLM_TASK=$TASK \
            NEWVLLM_MAX_TOKENS=$MT \
            NEWVLLM_RANGE="$lo:$hi" \
                python experiments/scripts/run_newvllm_ruler_baseline.py > "$log" 2>&1
            echo "[完成] $log exit=$?（完成度以 '===== ' 计数为准）"
            clean_orphan_engines
        fi
    done
done

if all_chunks_complete; then
    echo "GPU$GPU $TASK mt$MT 全部 $NCHUNK 片完成"
else
    echo "GPU$GPU $TASK mt$MT 重试 $MAX_PASS 轮后仍有分片未完成，需人工介入"
fi
