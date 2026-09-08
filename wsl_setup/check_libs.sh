#!/bin/bash
P=$HOME/miniconda3/envs/cacheblend
echo "== lib 下静态库 =="
ls -l $P/lib/libcudart_static.a $P/lib/libcudadevrt.a 2>&1
echo "== targets 下静态库 =="
ls $P/targets/x86_64-linux/lib/libcudart_static.a $P/targets/x86_64-linux/lib/libcudadevrt.a 2>&1
echo "== stubs =="
ls -ld $P/lib/stubs $P/targets/x86_64-linux/lib/stubs 2>&1
echo "== lib 下 cudart 相关 =="
ls -l $P/lib/ | grep -i cudart | head -10
