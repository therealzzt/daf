#!/bin/bash
set -e
P=$HOME/miniconda3/envs/cacheblend
NVTX=$P/lib/python3.10/site-packages/nvidia/nvtx
ln -sf $NVTX/lib/libnvToolsExt.so.1 $P/lib/libnvToolsExt.so.1
ln -sf $P/lib/libnvToolsExt.so.1 $P/lib/libnvToolsExt.so
ln -sf $NVTX/include/nvToolsExt.h $NVTX/include/nvToolsExtCuda.h $NVTX/include/nvToolsExtCudaRt.h $NVTX/include/nvToolsExtSync.h $NVTX/include/nvToolsExtOpenCL.h $P/include/ 2>/dev/null || true
ln -sfn $NVTX/include/nvtx3 $P/include/nvtx3
ls -l $P/lib/libnvToolsExt.so $P/include/nvToolsExt.h
