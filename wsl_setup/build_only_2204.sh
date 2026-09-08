#!/bin/bash
set -e
PYENV=$HOME/miniconda3/envs/cacheblend/bin
export PATH=/usr/bin:$PYENV:$PATH
export CUDA_HOME=$HOME/miniconda3/envs/cacheblend
export MAX_JOBS=3
export TORCH_CUDA_ARCH_LIST="8.9"
export TMPDIR=$HOME/.tmp
mkdir -p $TMPDIR
cd $HOME/CacheBlend/vllm_blend
echo "== 安装运行时依赖 =="
$PYENV/pip install -r requirements-cuda.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "== 编译安装 vllm_blend =="
$PYENV/pip install -e . --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "== 验证 =="
cd $HOME
$PYENV/python -c 'import vllm; print("vllm", vllm.__version__)'
echo "BUILD_DONE"
