#!/bin/bash
set -e
P=$HOME/miniconda3/envs/cacheblend
source $HOME/miniconda3/etc/profile.d/conda.sh
conda install -n cacheblend -c nvidia/label/cuda-12.1.0 \
  "cuda-nvcc=12.1.66" "cuda-cudart-dev=12.1.55" "cuda-libraries-dev=12.1.0" -y 2>&1 | tail -3
echo "== nvcc version =="
$P/bin/nvcc --version | tail -1
mkdir -p $P/include $P/lib
ln -sfn $P/targets/x86_64-linux/include/* $P/include/ 2>/dev/null || true
ln -sfn $P/targets/x86_64-linux/lib/* $P/lib/ 2>/dev/null || true
ls -l $P/include/cuda.h | head -1
ls $P/lib/libcudart.so*
