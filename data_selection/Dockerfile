# 1. 基础镜像：选择带开发工具的 CUDA 镜像
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 设置环境变量，避免交互式安装时的卡顿
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 2. 安装系统级依赖（解决 DCLM 的编译问题）
RUN apt-get update && apt-get install -y \
    git wget cmake build-essential \
    g++-9 gcc-9 \
    python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置 g++-9 优先级（你的 Patch 依赖这个）
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90

# 3. 安装 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# 4. 复制项目文件（包括你的 Patches）
COPY . /app

# 5. 执行补丁并创建环境（自动化你的 README 步骤）
RUN cd /app/third_party/Cherry_LLM && git apply ../../assets/patches/cherry_llm.patch || true
RUN cd /app/third_party/DCLM && git apply ../../assets/patches/dclm.patch || true

# 6. (可选) 预创建核心 Conda 环境（为了加速复现，可以只创建最常用的）
RUN conda create -n bench python=3.11 -y && \
    /opt/conda/envs/bench/bin/pip install -e /app/third_party/LlamaFactory