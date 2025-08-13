# 数据脱敏系统

一个基于AI的智能数据脱敏系统，支持人脸、车牌、街牌文本的自动检测与模糊处理。

## 功能特性

- 🎭 **人脸脱敏**: 使用YOLOv8-face或RetinaFace自动检测并模糊人脸
- 🚗 **车牌脱敏**: 使用YOLOv8或Haarcascade检测并模糊车牌号码
- 📝 **文本脱敏**: 使用DBNet检测并模糊街牌、标识等敏感文本
- 🌐 **Web界面**: 现代化的Web用户界面，支持拖拽上传
- 🔒 **隐私保护**: 本地处理，不上传到外部服务器
- 🐳 **Docker部署**: 一键Docker容器化部署
- 🌍 **多服务器支持**: 支持欧洲、美国、亚洲服务器选择
- ⚡ **CPU优化**: 专为CPU环境优化，无需GPU

## 系统架构

```
数据脱敏系统/
├── src/                    # 核心代码
│   └── pipeline/          # 脱敏管道
│       ├── faces.py       # 人脸检测与模糊
│       ├── plates.py      # 车牌检测与模糊
│       └── texts.py       # 文本检测与模糊
├── models/                # AI模型文件
├── templates/             # Web模板
├── static/               # 静态资源
├── tests/                # 测试文件
├── app.py                # 主应用程序
├── Dockerfile            # Docker镜像构建
├── docker-compose.yml    # Docker编排
└── deploy.sh             # 部署脚本
```

## 快速开始

### 方式一：Docker部署（推荐）

1. **克隆项目**
```bash
git clone <repository-url>
cd data_sensitive/test2
```

2. **确保模型文件存在**
```bash
ls models/
# 应该包含：
# - yolov8n-face.pt
# - license_plate_detector.pt
# - haarcascade_russian_plate_number.xml
# - retinaface_resnet50.pth
```

3. **一键部署**
```bash
# 开发环境
./deploy.sh

# 生产环境（包含Nginx）
./deploy.sh production
```

4. **访问系统**
- 开发环境: http://localhost:8000
- 生产环境: https://localhost

### 方式二：本地开发

1. **创建Conda环境**
```bash
conda env create -f environment.yml
conda activate data-sensitive-cpu
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动应用**
```bash
python app.py
```

## 使用说明

### Web界面使用

1. **访问系统**: 打开浏览器访问 http://localhost:8000
2. **阅读隐私协议**: 首次访问需要阅读并同意隐私声明
3. **选择服务器**: 根据地理位置选择合适的服务器
4. **配置脱敏选项**:
   - 启用/禁用脱敏功能
   - 选择脱敏类型（人脸/车牌/文本）
   - 选择模糊方法（高斯模糊/像素化）
5. **上传文件**: 支持拖拽或点击选择图像文件
6. **处理与下载**: 系统自动处理并提供下载链接

### API使用

#### 上传并处理文件
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "files=@image1.jpg" \
  -F "files=@image2.png" \
  -F "enable_anonymization=true" \
  -F "server_region=europe" \
  -F "blur_faces=true" \
  -F "blur_plates=true" \
  -F "blur_texts=true" \
  -F "blur_method=gaussian"
```

#### 健康检查
```bash
curl http://localhost:8000/api/health
```

## 配置说明

### 环境变量

- `CUDA_VISIBLE_DEVICES=-1`: 强制使用CPU
- `TF_CPP_MIN_LOG_LEVEL=2`: 降低TensorFlow日志级别
- `TF_ENABLE_ONEDNN_OPTS=0`: 禁用OneDNN优化

### 模型配置

系统支持多种AI模型后端：

#### 人脸检测
- **YOLOv8-face**: 需要 `models/yolov8n-face.pt`
- **RetinaFace**: 需要 `models/retinaface_resnet50.pth`

#### 车牌检测
- **YOLOv8**: 需要 `models/license_plate_detector.pt`
- **Haarcascade**: 需要 `models/haarcascade_russian_plate_number.xml`

#### 文本检测
- **DBNet**: 需要DBNet仓库和权重文件

## 部署选项

### 开发环境
```bash
docker-compose up -d
```

### 生产环境
```bash
docker-compose --profile production up -d
```

生产环境包含：
- Nginx反向代理
- SSL/TLS加密
- 负载均衡
- 安全头设置

### 扩展部署

#### 多实例负载均衡
```yaml
# docker-compose.yml
services:
  data-anonymization:
    deploy:
      replicas: 3
```

#### 资源限制
```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2'
```

## 性能优化

### CPU优化
- 使用CPU版本的PyTorch
- 图像预处理尺寸限制
- 模型预热机制
- 批处理支持

### 内存优化
- 临时文件自动清理
- 流式文件处理
- 内存使用监控

## 安全特性

### 数据安全
- 本地处理，不上传外部
- 临时文件自动删除
- 随机会话ID
- HTTPS加密传输

### 访问控制
- 隐私协议确认
- 文件类型验证
- 上传大小限制
- 请求频率限制

## 监控与日志

### 日志配置
```python
# 日志文件: logs/app.log
# 轮转: 每天轮转，保留7天
```

### 健康检查
```bash
# Docker健康检查
docker-compose ps

# 应用健康检查
curl http://localhost:8000/api/health
```

### 性能监控
```bash
# 查看容器资源使用
docker stats

# 查看应用日志
docker-compose logs -f
```

## 故障排除

### 常见问题

1. **模型文件缺失**
```bash
# 检查模型文件
ls -la models/
# 下载缺失的模型文件
```

2. **内存不足**
```bash
# 增加Docker内存限制
# 或减少并发处理数量
```

3. **端口冲突**
```bash
# 修改docker-compose.yml中的端口映射
ports:
  - "8080:8000"  # 改为8080
```

### 调试模式
```bash
# 启用调试日志
export TF_CPP_MIN_LOG_LEVEL=0

# 查看详细错误
docker-compose logs --tail=100
```

## 开发指南

### 添加新的脱敏类型

1. 在 `src/pipeline/` 下创建新模块
2. 实现检测和模糊方法
3. 在 `app.py` 中集成
4. 更新Web界面选项

### 自定义模型

1. 将模型文件放入 `models/` 目录
2. 修改对应的pipeline模块
3. 更新配置文件

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系我们

- 邮箱: support@data-anonymization.com
- 问题反馈: GitHub Issues
- 文档: 项目Wiki

---

**注意**: 本系统基于AI技术，检测准确率可能受图像质量等因素影响，建议使用前进行测试验证。
