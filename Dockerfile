# ============================================================
# Dockerfile: Ubuntu 22.04 + CUDA 12.8 + Python 3.10 + PyTorch 2.6.0
# ============================================================

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# ---- 安装系统依赖 ----
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-distutils \
    build-essential git wget curl vim ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- 设置 python 默认指向 python3 ----
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# ---- 安装 PyTorch / TorchAO / Torchtune ----
RUN pip install torch==2.6.0 torchao==0.8.0 \
    --index-url https://download.pytorch.org/whl/cu118 && \
    pip install torchtune==0.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# ---- 复制项目文件并安装 requirements ----
WORKDIR /workspace
COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ---- 默认启动命令 ----
CMD ["python", "--version"]
