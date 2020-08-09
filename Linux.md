---
layout: page
title: Linux
---

1. It is best to change the mirror site to Tsinghua or Aliyun. A hell a lot faster. For instance pip3 install --user --upgrade tensorflow  -i https://pypi.tuna.tsinghua.edu.cn/simple
2. watch -n0.1 nvidia-smi can monitor the gpu usage
3. watch -n-0.1 cpufreq-info
4. sudo cupfreq-set -g powersave, performance,conservative
5. change mirror server: /etc/apt/sources.list
Change Mirror to Tsinghua:
URL: https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/

# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
