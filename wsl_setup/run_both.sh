#!/bin/bash
# 依次跑原版 CacheBlend（DAF=0）和 DAF（DAF=1）的完整 10 样本实验
# 日志写入 Windows 侧 wsl_setup/logs/，方便事后分析
set -e
cd /home/zzt/CacheBlend
export HF_ENDPOINT=https://hf-mirror.com
PY=/home/zzt/miniconda3/envs/cacheblend/bin/python
LOGS=/mnt/d/Project/CacheBlend/wsl_setup/logs
mkdir -p "$LOGS"

echo "=== [1/2] DAF=0 (CacheBlend) start: $(date) ==="
DAF=0 $PY example/blend.py > "$LOGS/run_cb.log" 2>&1
echo "=== DAF=0 done: $(date) ==="

echo "=== [2/2] DAF=1 (DAF) start: $(date) ==="
DAF=1 $PY example/blend.py > "$LOGS/run_daf.log" 2>&1
echo "=== DAF=1 done: $(date) ==="

echo "ALL_DONE"
