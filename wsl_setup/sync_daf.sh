#!/bin/bash
# 把 Windows 侧改好的 3 个文件同步到 WSL 工作副本，并做 Python 语法检查
set -e
SRC=/mnt/d/Project/CacheBlend
DST=/home/zzt/CacheBlend

cp "$SRC/vllm_blend/vllm/model_executor/models/llama.py" "$DST/vllm_blend/vllm/model_executor/models/llama.py"
cp "$SRC/vllm_blend/vllm/attention/backends/xformers.py" "$DST/vllm_blend/vllm/attention/backends/xformers.py"
cp "$SRC/example/blend.py" "$DST/example/blend.py"
echo "sync done"

PY=/home/zzt/miniconda3/envs/cacheblend/bin/python
$PY -m py_compile "$DST/vllm_blend/vllm/model_executor/models/llama.py"
$PY -m py_compile "$DST/vllm_blend/vllm/attention/backends/xformers.py"
$PY -m py_compile "$DST/example/blend.py"
echo "py_compile OK"
