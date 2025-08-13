#!/bin/bash

# 数据脱敏系统部署脚本

set -e

echo "🚀 开始部署数据脱敏系统..."

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose 未安装，请先安装 Docker Compose"
    exit 1
fi

# 检查必要文件
echo "📋 检查必要文件..."
required_files=("Dockerfile" "docker-compose.yml" "requirements.txt" "app.py")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 缺少必要文件: $file"
        exit 1
    fi
done

# 检查模型文件
echo "🤖 检查模型文件..."
model_dir="models"
if [ ! -d "$model_dir" ]; then
    echo "❌ 模型目录不存在: $model_dir"
    exit 1
fi

required_models=("yolov8n-face.pt" "license_plate_detector.pt")
for model in "${required_models[@]}"; do
    if [ ! -f "$model_dir/$model" ]; then
        echo "⚠️  警告: 缺少模型文件 $model，某些功能可能无法使用"
    fi
done

# 创建必要目录
echo "📁 创建必要目录..."
mkdir -p logs uploads output temp static templates ssl

# 生成自签名 SSL 证书（仅用于开发环境）
if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/key.pem" ]; then
    echo "🔐 生成 SSL 证书..."
    openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
        -subj "/C=CN/ST=State/L=City/O=Organization/CN=localhost"
fi

# 构建 Docker 镜像
echo "🔨 构建 Docker 镜像..."
docker-compose build

# 启动服务
echo "🚀 启动服务..."
if [ "$1" = "production" ]; then
    echo "📦 启动生产环境 (包含 Nginx)..."
    docker-compose --profile production up -d
else
    echo "🔧 启动开发环境..."
    docker-compose up -d
fi

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 健康检查
echo "🏥 执行健康检查..."
if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "✅ 服务启动成功！"
    echo ""
    echo "🌐 访问地址:"
    echo "   开发环境: http://localhost:8000"
    if [ "$1" = "production" ]; then
        echo "   生产环境: https://localhost (需要接受自签名证书)"
    fi
    echo ""
    echo "📊 查看日志: docker-compose logs -f"
    echo "🛑 停止服务: docker-compose down"
else
    echo "❌ 服务启动失败，请检查日志"
    docker-compose logs
    exit 1
fi

echo "🎉 部署完成！"
