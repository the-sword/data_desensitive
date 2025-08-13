# 多阶段构建 - 基础环境层
FROM docker.m.daocloud.io/python:3.9-slim AS base

# 设置工作目录
WORKDIR /app

# 安装系统依赖（这层会被缓存，除非基础镜像更新）
RUN apt-get update && apt-get install -y --no-install-recommends \
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
FROM base AS dependencies

# 先只复制依赖文件（优化缓存 - 只有requirements.txt变化时才重建）
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 最终应用层
FROM base AS final

# 从依赖层复制已安装的包
COPY --from=dependencies /usr/local /usr/local

# 复制应用代码（最后复制，避免代码变化导致重新安装依赖）
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY app.py config.py batch_process.py ./

# DBNet will be mounted at runtime via docker-compose
RUN mkdir -p DBNet

# 创建必要的目录并设置权限
RUN mkdir -p uploads output temp logs static templates \
    && chown -R nobody:nogroup /app \
    && chmod -R 755 /app

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
# Switch to non-root user for better security
USER nobody

CMD ["python", "app.py"]
