#!/bin/bash
# 运行 DAF 冒烟测试：$1 = 0（原版 CacheBlend 回归）或 1（DAF）
set -e
cp /mnt/d/Project/CacheBlend/wsl_setup/daf_smoke.py /home/zzt/CacheBlend/daf_smoke.py
cd /home/zzt/CacheBlend
export HF_ENDPOINT=https://hf-mirror.com
DAF=$1 /home/zzt/miniconda3/envs/cacheblend/bin/python daf_smoke.py
