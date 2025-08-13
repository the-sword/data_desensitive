"""
数据脱敏系统配置文件
"""

import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"

# 服务器配置
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# 文件上传限制
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", 50))

# 处理配置
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 1920))  # 最大图像尺寸
PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", 300))  # 5分钟超时
CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", 3600))  # 1小时清理一次

# 模型配置
MODEL_CONFIG = {
    "face": {
        "backend": os.getenv("FACE_BACKEND", "yolov8_face"),
        "weights_path": MODELS_DIR / "yolov8n-face.pt",
        "fallback_weights": MODELS_DIR / "retinaface_resnet50.pth"
    },
    "plate": {
        "backend": os.getenv("PLATE_BACKEND", "yolov8_plate"),
        "weights_path": MODELS_DIR / "license_plate_detector.pt",
        "cascade_path": MODELS_DIR / "haarcascade_russian_plate_number.xml"
    },
    "text": {
        "dbnet_root": Path(os.getenv("DBNET_ROOT", str(BASE_DIR.parent / "DBNet"))),
        "weights_path": Path(os.getenv("TEXT_WEIGHTS", str(BASE_DIR.parent / "DBNet" / "weights" / "best.pt"))),
        "input_size": int(os.getenv("TEXT_INPUT_SIZE", 960))
    }
}

# 日志配置
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "file": LOGS_DIR / "app.log",
    "rotation": "1 day",
    "retention": "7 days",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
}

# 安全配置
SECURITY_CONFIG = {
    "session_timeout": int(os.getenv("SESSION_TIMEOUT", 3600)),  # 1小时
    "rate_limit": int(os.getenv("RATE_LIMIT", 10)),  # 每分钟10次请求
    "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    "trusted_hosts": os.getenv("TRUSTED_HOSTS", "localhost,127.0.0.1").split(",")
}

# 服务器区域配置
SERVER_REGIONS = {
    "europe": {
        "name": "欧洲服务器",
        "description": "符合GDPR规范，适合欧洲用户",
        "flag": "🇪🇺"
    },
    "america": {
        "name": "美国服务器", 
        "description": "高速稳定，适合美洲用户",
        "flag": "🇺🇸"
    },
    "asia": {
        "name": "亚洲服务器",
        "description": "低延迟访问，适合亚太用户", 
        "flag": "🌏"
    }
}

# 模糊方法配置
BLUR_METHODS = {
    "gaussian": {
        "name": "高斯模糊",
        "description": "使用高斯核进行平滑模糊"
    },
    "pixelate": {
        "name": "像素化",
        "description": "降低分辨率形成像素化效果"
    },
    "solid": {
        "name": "纯色填充",
        "description": "使用纯色块覆盖敏感区域"
    }
}

# 环境变量设置
def setup_environment():
    """设置环境变量"""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("PYTHONPATH", str(BASE_DIR))

# 创建必要目录
def create_directories():
    """创建必要的目录"""
    directories = [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR, LOGS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# 验证配置
def validate_config():
    """验证配置是否正确"""
    errors = []
    
    # 检查模型文件
    for model_type, config in MODEL_CONFIG.items():
        if "weights_path" in config:
            if not config["weights_path"].exists():
                errors.append(f"缺少{model_type}模型文件: {config['weights_path']}")
    
    # 检查目录权限
    for directory in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR, LOGS_DIR]:
        if not os.access(directory, os.W_OK):
            errors.append(f"目录无写入权限: {directory}")
    
    return errors

if __name__ == "__main__":
    # 配置验证脚本
    setup_environment()
    create_directories()
    
    errors = validate_config()
    if errors:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ 配置验证通过")
