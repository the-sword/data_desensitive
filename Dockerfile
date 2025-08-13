# 多阶段构建 - 基础环境层
FROM docker.xuanyuan.me/python:3.10-slim as base

# 设置工作目录
WORKDIR /app

# 安装系统依赖（这层会被缓存，除非基础镜像更新）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    ffmpeg \
    libfontconfig1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python依赖层
FROM base as dependencies

# 先只复制依赖文件（优化缓存 - 只有requirements.txt变化时才重建）
COPY requirements.txt .

# 安装 Python 依赖到用户目录（加速安装）
RUN pip install --user --no-cache-dir -r requirements.txt

# 最终应用层
FROM base as final

# 从依赖层复制已安装的包
COPY --from=dependencies /root/.local /root/.local

# 将用户安装的包添加到PATH
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码（最后复制，避免代码变化导致重新安装依赖）
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY models/ ./models/
COPY app.py config.py batch_process.py ./

# 创建必要的目录
RUN mkdir -p uploads output temp logs static templates

# 设置环境变量
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/api/health || exit 1

# 启动命令
CMD ["python", "app.py"]
