#!/bin/bash
P=$HOME/miniconda3/envs/cacheblend/bin
echo "== vllm editable 安装状态 =="
$P/pip show vllm 2>&1 | head -4
echo "== 编译产物 .so =="
ls $HOME/CacheBlend/vllm_blend/vllm/*.so 2>&1
echo "== 关键依赖 =="
$P/pip list 2>/dev/null | grep -iE "^(transformers|xformers|torch|vllm|outlines|fastapi|ray) " 
