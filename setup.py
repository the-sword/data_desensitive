#!/usr/bin/env python3
"""
数据脱敏系统设置脚本
用于初始化环境和检查依赖
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import hashlib

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    print(f"✅ Python版本: {sys.version}")
    return True

def check_dependencies():
    """检查基本依赖"""
    required_packages = [
        'torch', 'torchvision', 'fastapi', 'uvicorn', 
        'opencv-python', 'numpy', 'pillow', 'loguru'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """检查模型文件"""
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ models目录不存在")
        return False
    
    model_files = {
        "yolov8n-face.pt": "人脸检测模型",
        "license_plate_detector.pt": "车牌检测模型", 
        "haarcascade_russian_plate_number.xml": "车牌级联分类器",
        "retinaface_resnet50.pth": "RetinaFace模型"
    }
    
    missing_models = []
    for model_file, description in model_files.items():
        model_path = models_dir / model_file
        if model_path.exists():
            print(f"✅ {description}: {model_file}")
        else:
            print(f"❌ {description}: {model_file}")
            missing_models.append(model_file)
    
    if missing_models:
        print(f"\n缺少模型文件: {', '.join(missing_models)}")
        print("某些功能可能无法使用")
        return False
    
    return True

def create_directories():
    """创建必要的目录"""
    directories = ["uploads", "output", "temp", "logs", "static", "templates", "ssl"]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建目录: {directory}")
        else:
            print(f"✅ 目录已存在: {directory}")

def check_docker():
    """检查Docker是否可用"""
    try:
        result = subprocess.run(["docker", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker未安装或不可用")
        return False

def check_docker_compose():
    """检查Docker Compose是否可用"""
    try:
        result = subprocess.run(["docker-compose", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker Compose: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker Compose未安装或不可用")
        return False

def download_sample_models():
    """下载示例模型文件（如果用户同意）"""
    print("\n是否下载示例模型文件？这将需要一些时间和网络流量。")
    response = input("输入 'y' 下载，其他任意键跳过: ").lower()
    
    if response != 'y':
        print("跳过模型下载")
        return
    
    # 这里可以添加模型下载逻辑
    # 注意：实际部署时需要提供真实的模型下载链接
    print("模型下载功能需要配置具体的下载源")

def run_tests():
    """运行基本测试"""
    print("\n运行基本功能测试...")
    
    try:
        # 测试导入核心模块
        from src.pipeline.faces import FaceBlurrer
        from src.pipeline.plates import PlateBlurrer
        from src.pipeline.texts import TextBlurrer
        print("✅ 核心模块导入成功")
        
        # 简单的初始化测试
        try:
            face_blurrer = FaceBlurrer(warmup=False)
            print("✅ 人脸检测器初始化成功")
        except Exception as e:
            print(f"⚠️  人脸检测器初始化警告: {e}")
        
        try:
            plate_blurrer = PlateBlurrer(warmup=False)
            print("✅ 车牌检测器初始化成功")
        except Exception as e:
            print(f"⚠️  车牌检测器初始化警告: {e}")
        
        try:
            text_blurrer = TextBlurrer(warmup=False)
            print("✅ 文本检测器初始化成功")
        except Exception as e:
            print(f"⚠️  文本检测器初始化警告: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 数据脱敏系统环境检查")
    print("=" * 50)
    
    # 基本检查
    checks = [
        ("Python版本", check_python_version),
        ("创建目录", create_directories),
        ("Python依赖", check_dependencies),
        ("模型文件", check_model_files),
        ("Docker", check_docker),
        ("Docker Compose", check_docker_compose),
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n📋 检查{name}...")
        if not check_func():
            all_passed = False
    
    # 可选操作
    download_sample_models()
    
    # 运行测试
    print("\n🧪 功能测试...")
    test_passed = run_tests()
    
    # 总结
    print("\n" + "=" * 50)
    if all_passed and test_passed:
        print("🎉 环境检查完成！系统已准备就绪。")
        print("\n下一步:")
        print("1. 启动开发服务器: python app.py")
        print("2. 或使用Docker部署: ./deploy.sh")
        print("3. 访问Web界面: http://localhost:8000")
    else:
        print("⚠️  环境检查发现问题，请解决后重试。")
        print("\n常见解决方案:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 下载模型文件到models/目录")
        print("3. 安装Docker和Docker Compose")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
