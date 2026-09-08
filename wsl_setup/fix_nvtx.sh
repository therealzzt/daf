#!/bin/bash
set -e
P=$HOME/miniconda3/envs/cacheblend
source $HOME/miniconda3/etc/profile.d/conda.sh
conda install -n cacheblend -c nvidia/label/cuda-12.1.0 "cuda-nvtx-dev=12.1.55" -y 2>&1 | tail -2
ls -l $P/lib/libnvToolsExt.so $P/include/nvToolsExt.h 2>&1
