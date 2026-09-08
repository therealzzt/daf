#!/bin/bash
set -e
P=$HOME/miniconda3/envs/cacheblend/bin
$P/pip install "transformers==4.41.2" "tokenizers<0.20" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -2
cd $HOME
$P/python -c 'import vllm, transformers, torch; print("vllm", vllm.__version__, "| transformers", transformers.__version__, "| torch", torch.__version__)'
