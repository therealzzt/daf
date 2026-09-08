#!/bin/bash
PYENV=$HOME/miniconda3/envs/cacheblend
echo "== nvcc =="; $PYENV/bin/nvcc --version | tail -1
echo "== prefix include =="; ls $PYENV/include/cuda.h 2>&1
echo "== targets layout =="; ls $PYENV/targets/x86_64-linux/include/cuda.h $PYENV/targets/x86_64-linux/lib/libcudart.so 2>&1
echo "== lib =="; ls $PYENV/lib/libcudart* 2>&1 | head -3
echo "== /usr/local/cuda =="; ls -d /usr/local/cuda* 2>&1
