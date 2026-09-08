#!/bin/bash
set -e
# 换清华源（jammy）
sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list
# 创建用户 zzt，免密 sudo
useradd -m -s /bin/bash zzt
echo 'zzt ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/zzt
printf '[user]\ndefault=zzt\n' > /etc/wsl.conf
# 装编译工具链（22.04 默认 gcc-11，CUDA 12.1 官方支持）
apt-get update -qq
apt-get install -y -qq build-essential git wget 2>&1 | tail -2
gcc --version | head -1
