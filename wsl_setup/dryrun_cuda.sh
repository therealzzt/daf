#!/bin/bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda install -n cacheblend -c nvidia/label/cuda-12.1.0 "cuda-nvcc=12.1.66" "cuda-cudart-dev=12.1.55" --dry-run 2>&1 | tail -30
