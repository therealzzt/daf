#!/bin/bash
PYENV=$HOME/miniconda3/envs/cacheblend/bin
export PATH=/usr/bin:$PYENV:$PATH
export CUDA_HOME=$HOME/miniconda3/envs/cacheblend
export TORCH_CUDA_ARCH_LIST="8.9"
cd $HOME/CacheBlend/vllm_blend
rm -rf /tmp/vbuild
cmake . -B /tmp/vbuild -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE=$PYENV/python3.10 2>&1 | tail -40
