#!/bin/bash
set -e
# 1. Miniconda
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel defaults 2>/dev/null || true
conda create -n cacheblend python=3.10 -y
# 2. CUDA 12.1 组件（精确版本，避开 conda 元包不降级组件的坑）
conda install -n cacheblend -c nvidia/label/cuda-12.1.0 "cuda-nvcc=12.1.66" "cuda-cudart-dev=12.1.55" "cuda-libraries-dev=12.1.0" -y
P=$HOME/miniconda3/envs/cacheblend
$P/bin/nvcc --version | tail -1
mkdir -p $P/include $P/lib
ln -sfn $P/targets/x86_64-linux/include/* $P/include/ 2>/dev/null || true
ln -sfn $P/targets/x86_64-linux/lib/* $P/lib/ 2>/dev/null || true
# 3. torch 2.2.1 + xformers 0.0.25（PyPI 的 torch 2.2.1 即 cu121 构建）
$P/bin/pip install "numpy<2" cmake ninja packaging setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
$P/bin/pip install torch==2.2.1 xformers==0.0.25 -i https://pypi.tuna.tsinghua.edu.cn/simple
$P/bin/python -c 'import torch; print("torch", torch.__version__, torch.version.cuda, "available", torch.cuda.is_available())'
# 4. 项目拷贝 + 编译 vllm_blend（排除大体积 rootfs）
mkdir -p $HOME/CacheBlend
cd /mnt/d/Project/CacheBlend
tar cf - --exclude=wsl_setup/ubuntu2204.rootfs.tar.gz . | tar xf - -C $HOME/CacheBlend
cd $HOME/CacheBlend/vllm_blend
export PATH=/usr/bin:$P/bin:$PATH
export CUDA_HOME=$P
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="8.9"
$P/bin/pip install -e . --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
$P/bin/pip install -r $HOME/CacheBlend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
$P/bin/python -c 'import vllm; print("vllm", vllm.__version__)'
echo "ALL_DONE"
