#!/bin/bash
P=$HOME/miniconda3/envs/cacheblend
find $P -name "nvToolsExt*" 2>/dev/null | head -10
find $P -name "libnvToolsExt*" 2>/dev/null | head -10
source $HOME/miniconda3/etc/profile.d/conda.sh
conda list -n cacheblend | grep -i nvtx
