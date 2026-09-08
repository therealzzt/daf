#!/bin/bash
set -e
PYENV=$HOME/miniconda3/envs/cacheblend/bin
$PYENV/python --version
$PYENV/nvcc --version | tail -1
$PYENV/pip install torch==2.2.1 xformers==0.0.25 -i https://pypi.tuna.tsinghua.edu.cn/simple
$PYENV/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
