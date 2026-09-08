#!/bin/bash
set -e
PYENV=$HOME/miniconda3/envs/cacheblend/bin
$PYENV/pip install "numpy<2" cmake ninja packaging setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
export PATH=/usr/bin:$PYENV:$PATH
export CUDA_HOME=$HOME/miniconda3/envs/cacheblend
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12
export MAX_JOBS=8
# 只编译 Ada Lovelace (RTX 4060, sm_89) 的内核，节省时间
export TORCH_CUDA_ARCH_LIST="8.9"
cd $HOME/CacheBlend/vllm_blend
$PYENV/pip install -e . --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
$PYENV/pip install -r $HOME/CacheBlend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
$PYENV/python -c 'import vllm; print("vllm", vllm.__version__)'
