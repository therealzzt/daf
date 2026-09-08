#!/bin/bash
set -e
P=$HOME/miniconda3/envs/cacheblend
source $HOME/miniconda3/etc/profile.d/conda.sh
conda install -n cacheblend -c nvidia/label/cuda-12.1.0 "cuda-cudart-static=12.1.55" "cuda-driver-dev=12.1.55" -y 2>&1 | tail -3
ls -l $P/lib/libcudart_static.a $P/lib/libcudadevrt.a $P/lib/stubs/libcuda.so
