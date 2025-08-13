# 数据脱敏系统部署指南

## 🎯 项目完成状态

✅ **已完成的功能**:
- 人脸检测与脱敏 (YOLOv8-face + RetinaFace)
- 车牌检测与脱敏 (YOLOv8 + Haarcascade)  
- 文本检测与脱敏 (DBNet)
- 现代化Web界面 (FastAPI + Bootstrap)
- 隐私协议确认流程
- 多服务器选择 (欧洲/美国/亚洲)
- Docker容器化部署
- 批量处理脚本
- 完整的配置管理

## 🚀 快速部署

### 1. 环境检查
```bash
# 检查系统环境和依赖
python setup.py
```

### 2. Docker部署 (推荐)
```bash
# 开发环境部署
./deploy.sh

# 生产环境部署 (包含Nginx + SSL)
./deploy.sh production
```

### 3. 本地开发部署
```bash
# 创建conda环境
conda env create -f environment.yml
conda activate data-sensitive-cpu

# 安装依赖
pip install -r requirements.txt

# 启动应用
python app.py
```

## 📋 部署检查清单

### 必需文件检查
- [x] `models/yolov8n-face.pt` - 人脸检测模型
- [x] `models/license_plate_detector.pt` - 车牌检测模型  
- [x] `models/haarcascade_russian_plate_number.xml` - 车牌级联分类器
- [x] `models/retinaface_resnet50.pth` - RetinaFace模型
- [x] `models/ocr_*` - PaddleOCR模型文件

### 目录结构检查
```
data_sensitive/test2/
├── 📁 src/pipeline/          # 核心脱敏管道
├── 📁 models/               # AI模型文件
├── 📁 templates/            # Web模板
├── 📁 static/              # 静态资源
├── 📁 tests/               # 测试文件
├── 🐳 Dockerfile           # Docker镜像
├── 🐳 docker-compose.yml   # Docker编排
├── 🚀 deploy.sh            # 部署脚本
├── ⚙️ config.py            # 配置文件
├── 🔧 setup.py             # 环境检查
├── 📦 batch_process.py     # 批量处理
└── 📖 README.md            # 完整文档
```

## 🌐 访问地址

| 环境 | 地址 | 说明 |
|------|------|------|
| 开发环境 | http://localhost:8000 | 直接访问FastAPI应用 |
| 生产环境 | https://localhost | 通过Nginx代理，包含SSL |
| API文档 | http://localhost:8000/docs | FastAPI自动生成的API文档 |
| 健康检查 | http://localhost:8000/api/health | 服务状态检查 |

## 🔧 功能使用

### Web界面流程
1. **隐私协议确认** - 首次访问需要阅读并同意隐私声明
2. **服务器选择** - 根据地理位置选择处理服务器
3. **脱敏配置** - 选择启用的脱敏类型和方法
4. **文件上传** - 支持拖拽上传多个图像文件
5. **自动处理** - 后台自动执行脱敏处理
6. **结果下载** - 提供处理后文件的打包下载

### 批量处理
```bash
# 批量处理目录中的所有图像
python batch_process.py /path/to/input /path/to/output

# 自定义脱敏选项
python batch_process.py /path/to/input /path/to/output \
  --no-faces \
  --method pixelate
```

### API调用
```bash
# 上传并处理文件
curl -X POST "http://localhost:8000/api/upload" \
  -F "files=@image.jpg" \
  -F "enable_anonymization=true" \
  -F "blur_faces=true" \
  -F "blur_plates=true" \
  -F "blur_texts=true"
```

## 🛠️ 配置选项

### 环境变量
```bash
# 服务器配置
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEBUG=false

# 文件限制
MAX_FILE_SIZE=104857600  # 100MB
MAX_FILES_PER_REQUEST=50

# 模型后端选择
FACE_BACKEND=yolov8_face
PLATE_BACKEND=yolov8_plate

# 日志级别
LOG_LEVEL=INFO
```

### Docker配置
```yaml
# docker-compose.yml 中的资源限制
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G
```

## 🔍 监控与维护

### 日志查看
```bash
# 查看应用日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs data-anonymization

# 查看本地日志文件
tail -f logs/app.log
```

### 健康检查
```bash
# 检查服务状态
docker-compose ps

# API健康检查
curl http://localhost:8000/api/health

# 容器资源使用
docker stats
```

### 清理维护
```bash
# 清理临时文件
docker-compose exec data-anonymization find /app/temp -type f -mtime +1 -delete

# 重启服务
docker-compose restart

# 更新镜像
docker-compose pull && docker-compose up -d
```

## 🚨 故障排除

### 常见问题

1. **模型文件缺失**
```bash
# 检查模型文件
ls -la models/
# 确保所有必需的.pt和.xml文件存在
```

2. **内存不足**
```bash
# 增加Docker内存限制
# 编辑docker-compose.yml中的memory限制
```

3. **端口冲突**
```bash
# 修改端口映射
# 编辑docker-compose.yml中的ports配置
```

4. **权限问题**
```bash
# 确保目录权限正确
chmod -R 755 uploads output temp logs
```

### 调试模式
```bash
# 启用详细日志
export TF_CPP_MIN_LOG_LEVEL=0
export LOG_LEVEL=DEBUG

# 查看详细错误信息
docker-compose logs --tail=100
```

## 📈 性能优化

### CPU优化建议
- 使用CPU版本的PyTorch (已配置)
- 限制图像处理尺寸 (默认1920px)
- 启用模型预热 (已实现)
- 合理设置并发数量

### 内存优化建议
- 定期清理临时文件 (已实现)
- 限制同时处理的文件数量
- 监控内存使用情况

## 🔐 安全考虑

### 数据安全
- ✅ 本地处理，不上传外部服务器
- ✅ 临时文件自动清理
- ✅ 随机会话ID
- ✅ HTTPS加密 (生产环境)

### 访问控制
- ✅ 文件类型验证
- ✅ 上传大小限制
- ✅ 请求频率限制 (可配置)

## 📞 支持与反馈

如遇到问题或需要帮助：
1. 查看本文档的故障排除部分
2. 检查 `logs/app.log` 日志文件
3. 运行 `python setup.py` 进行环境检查
4. 提交GitHub Issue (如适用)

---

**部署完成！** 🎉

系统已成功封装为Docker部署，包含完整的Web界面和API服务。
