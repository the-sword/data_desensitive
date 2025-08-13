#!/usr/bin/env python3
"""
数据脱敏系统主应用
支持人脸、车牌、街牌检测与模糊处理
"""

import os
import sys
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from loguru import logger

# 导入我们的脱敏管道
from src.pipeline.faces import FaceBlurrer
from src.pipeline.plates import PlateBlurrer
from src.pipeline.texts import TextBlurrer

app = FastAPI(title="数据脱敏系统", description="支持人脸、车牌、街牌检测与脱敏")

# 静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 简单中英翻译
TRANSLATIONS = {
    "app_title": {"zh": "数据脱敏系统", "en": "Data Anonymization System"},
    "app_lead": {"zh": "智能检测并保护您图像中的敏感信息", "en": "Intelligently detect and protect sensitive content in your images"},
    "select_server": {"zh": "选择处理服务器", "en": "Select Processing Server"},
    "server_europe": {"zh": "欧洲服务器", "en": "Europe Server"},
    "server_america": {"zh": "美国服务器", "en": "US Server"},
    "server_asia": {"zh": "亚洲服务器", "en": "Asia Server"},
    "upload_drop_hint": {"zh": "拖拽文件到此处或点击选择", "en": "Drag files here or click to select"},
    "upload_types_hint": {"zh": "支持 JPG, PNG, BMP, TIFF 格式图像", "en": "Supports JPG, PNG, BMP, TIFF images"},
    "select_files_btn": {"zh": "选择文件", "en": "Select Files"},
    "start_btn": {"zh": "开始处理", "en": "Start Processing"},
    "lang_label": {"zh": "语言", "en": "Language"},
    # Settings & options
    "settings_title": {"zh": "脱敏设置", "en": "Anonymization Settings"},
    "enable_anonymization": {"zh": "启用数据脱敏", "en": "Enable Anonymization"},
    "face_blur": {"zh": "人脸脱敏", "en": "Face Blur"},
    "plate_blur": {"zh": "车牌脱敏", "en": "License Plate Blur"},
    "text_blur": {"zh": "文本脱敏", "en": "Text Blur"},
    "blur_method": {"zh": "模糊方式", "en": "Blur Method"},
    "blur_gaussian": {"zh": "高斯模糊", "en": "Gaussian Blur"},
    "blur_pixelate": {"zh": "像素化", "en": "Pixelate"},
    # Pick mode
    "pick_mode": {"zh": "选择模式：", "en": "Pick Mode:"},
    "pick_file": {"zh": "文件", "en": "Files"},
    "pick_folder": {"zh": "文件夹", "en": "Folder"},
    "pick_hint": {"zh": "提示：拖拽暂不支持整夹拖入，请使用“选择文件”按钮选择文件夹", "en": "Tip: Folder drag-and-drop may not be supported. Use 'Select Files' to pick a folder."},
    # Banner and actions
    "download_processed": {"zh": "下载处理后的文件", "en": "Download Processed Files"},
    "restart": {"zh": "重新开始", "en": "Restart"},
    # JS messages
    "please_select_files": {"zh": "请选择文件", "en": "Please select files"},
    "uploaded_processing": {"zh": "文件已上传，正在处理...", "en": "Files uploaded, processing..."},
    "processed_count": {"zh": "成功处理了 {n} 个文件", "en": "Successfully processed {n} files"},
    "more_not_shown": {"zh": "还有 {n} 个文件未显示...", "en": "{n} more files not shown..."},
    "processing_failed": {"zh": "处理失败", "en": "Processing failed"},
    "processing": {"zh": "处理中...", "en": "Processing..."},
    "completed": {"zh": "处理完成！", "en": "Completed!"},
    "processing_percent": {"zh": "正在处理：{p}%", "en": "Processing: {p}%"},
    # Banner section
    "banner_face_title": {"zh": "人脸保护", "en": "Face Protection"},
    "banner_face_desc": {"zh": "自动检测图像中的人脸并进行模糊处理，保护个人隐私", "en": "Automatically detect faces and blur them to protect privacy."},
    "banner_plate_title": {"zh": "车牌脱敏", "en": "License Plate Anonymization"},
    "banner_plate_desc": {"zh": "智能识别车牌号码并进行模糊处理，防止车辆信息泄露", "en": "Intelligently detect license plates and blur them to prevent leakage."},
    "banner_text_title": {"zh": "文本保护", "en": "Text Protection"},
    "banner_text_desc": {"zh": "检测街牌、标识等敏感文本信息并进行脱敏处理", "en": "Detect sensitive text like street signs and anonymize it."},
    # Privacy modal
    "privacy_title": {"zh": "隐私声明协议", "en": "Privacy Notice"},
    "privacy_read": {"zh": "请仔细阅读以下隐私声明，了解我们如何处理您的数据。", "en": "Please read the privacy notice to understand how we handle your data."},
    "privacy_data_info": {"zh": "数据处理说明：", "en": "Data Processing Info:"},
    "privacy_local": {"zh": "我们仅在您的设备上本地处理图像数据", "en": "We process images locally on your device."},
    "privacy_no_upload": {"zh": "不会将原始图像上传到外部服务器", "en": "We do not upload originals to external servers."},
    "privacy_models": {"zh": "脱敏处理使用开源AI模型，在CPU上运行", "en": "Anonymization uses open-source AI models on CPU."},
    "privacy_cleanup": {"zh": "处理完成后，临时文件将被自动删除", "en": "Temporary files are removed after processing."},
    "privacy_optional": {"zh": "您可以选择是否启用脱敏功能", "en": "You may choose whether to enable anonymization."},
    "privacy_types": {"zh": "支持的脱敏类型：", "en": "Supported Anonymization Types:"},
    "privacy_face": {"zh": "人脸脱敏：自动检测并模糊人脸区域", "en": "Face: Detect and blur face regions."},
    "privacy_plate": {"zh": "车牌脱敏：检测并模糊车牌号码", "en": "License Plate: Detect and blur plate numbers."},
    "privacy_text": {"zh": "文本脱敏：检测并模糊街牌、标识等文本", "en": "Text: Detect and blur street signs and labels."},
    "btn_cancel": {"zh": "取消", "en": "Cancel"},
    "btn_agree": {"zh": "我同意并继续", "en": "I Agree and Continue"},
    # Nav and footer
    "home_nav": {"zh": "首页", "en": "Home"},
    "privacy_nav": {"zh": "隐私协议", "en": "Privacy"},
    "footer_text": {"zh": "© 2024 数据脱敏系统. 保护您的隐私数据.", "en": "© 2024 Data Anonymization System. Protect your privacy data."},
}

def translate(key: str, lang: str = "zh") -> str:
    entry = TRANSLATIONS.get(key, {})
    return entry.get(lang, entry.get("zh", key))

def get_lang_from_request(request: Request) -> str:
    lang = request.query_params.get("lang")
    if lang in ("zh", "en"):
        return lang
    cookie_lang = request.cookies.get("lang")
    if cookie_lang in ("zh", "en"):
        return cookie_lang
    return "zh"

# 全局配置
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")

# 处理状态跟踪
PROCESSING_SESSIONS = {}

# 区域映射：将前端选项映射到 uploads 下的目录名
REGION_MAP = {
    "america": "us",
    "europe": "un",
    "asia": "asia",
}

# 确保目录存在
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# 初始化脱敏器（延迟加载以避免启动时间过长）
face_blurrer = None
plate_blurrer = None
text_blurrer = None

def get_face_blurrer():
    global face_blurrer
    if face_blurrer is None:
        try:
            face_blurrer = FaceBlurrer(warmup=True)
            logger.info("人脸检测器初始化完成")
        except Exception as e:
            logger.error(f"人脸检测器初始化失败: {e}")
            raise HTTPException(status_code=500, detail="人脸检测器初始化失败")
    return face_blurrer

def get_plate_blurrer():
    global plate_blurrer
    if plate_blurrer is None:
        try:
            plate_blurrer = PlateBlurrer(warmup=True)
            logger.info("车牌检测器初始化完成")
        except Exception as e:
            logger.error(f"车牌检测器初始化失败: {e}")
            raise HTTPException(status_code=500, detail="车牌检测器初始化失败")
    return plate_blurrer

def get_text_blurrer():
    global text_blurrer
    if text_blurrer is None:
        try:
            text_blurrer = TextBlurrer(warmup=True)
            logger.info("文本检测器初始化完成")
        except Exception as e:
            logger.error(f"文本检测器初始化失败: {e}")
            raise HTTPException(status_code=500, detail="文本检测器初始化失败")
    return text_blurrer

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页"""
    lang = get_lang_from_request(request)
    # 提供一个基于当前语言的翻译函数给模板
    def t(key: str):
        return translate(key, lang)
    context = {"request": request, "lang": lang, "t": t}
    response = templates.TemplateResponse("index.html", context)
    response.set_cookie("lang", lang, max_age=3600*24*365)
    return response

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_agreement(request: Request):
    """隐私协议页面"""
    lang = get_lang_from_request(request)
    def t(key: str):
        return translate(key, lang)
    context = {"request": request, "lang": lang, "t": t}
    response = templates.TemplateResponse("privacy.html", context)
    response.set_cookie("lang", lang, max_age=3600*24*365)
    return response

@app.post("/api/upload")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    enable_anonymization: bool = Form(True),
    server_region: str = Form("europe"),
    blur_faces: bool = Form(True),
    blur_plates: bool = Form(True),
    blur_texts: bool = Form(True),
    blur_method: str = Form("gaussian")
):
    """处理文件上传和脱敏"""
    
    if not files:
        raise HTTPException(status_code=400, detail="未选择文件")
    
    # 创建唯一的处理会话ID
    session_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    # 初始化处理状态
    PROCESSING_SESSIONS[session_id] = {
        "status": "processing",
        "progress": 0,
        "total_files": len(files),
        "processed_files": 0,
        "error": None
    }
    
    input_dir = session_dir / "input"
    output_dir = session_dir / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # 解析目标服务器（本地模拟）目录
    region_code = REGION_MAP.get(server_region, "eu")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = UPLOAD_DIR / region_code / f"{ts}_{session_id[:8]}"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 保存上传的文件
        uploaded_files = []
        for file in files:
            if file.filename:
                file_path = input_dir / file.filename
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                uploaded_files.append(file_path)
        
        logger.info(f"会话 {session_id}: 上传了 {len(uploaded_files)} 个文件")
        
        # 如果未启用脱敏：直接按照规则复制到目标“服务器”目录
        if not enable_anonymization:
            processed_count = 0
            for file_path in uploaded_files:
                try:
                    # 直接复制所有文件（包含非图像）
                    shutil.copy2(file_path, target_dir / file_path.name)
                    processed_count += 1
                    PROCESSING_SESSIONS[session_id]["processed_files"] = processed_count
                    PROCESSING_SESSIONS[session_id]["progress"] = processed_count / len(files)
                except Exception as e:
                    logger.error(f"复制文件 {file_path.name} 到 {target_dir} 时出错: {e}")

            # 为方便用户依旧提供下载包（从目标目录打包）
            zip_path = session_dir / "processed_files.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for f in target_dir.iterdir():
                    if f.is_file():
                        zipf.write(f, f.name)

            PROCESSING_SESSIONS[session_id]["progress"] = 1.0
            return JSONResponse({
                "success": True,
                "session_id": session_id,
                "processed_files": processed_count,
                "download_url": f"/api/download/{session_id}",
                "target_dir": str(target_dir)
            })

        # 执行脱敏处理
        processed_count = 0
        for file_path in uploaded_files:
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                try:
                    output_path = output_dir / file_path.name
                    current_path = str(file_path)
                    
                    # 按顺序应用各种脱敏
                    if blur_faces:
                        blurrer = get_face_blurrer()
                        temp_path = str(session_dir / f"temp_face_{file_path.name}")
                        blurrer.process_image(current_path, temp_path, method=blur_method)
                        current_path = temp_path
                    
                    if blur_plates:
                        blurrer = get_plate_blurrer()
                        temp_path = str(session_dir / f"temp_plate_{file_path.name}")
                        blurrer.process_image(current_path, temp_path, method=blur_method)
                        current_path = temp_path
                    
                    if blur_texts:
                        blurrer = get_text_blurrer()
                        temp_path = str(session_dir / f"temp_text_{file_path.name}")
                        blurrer.process_image(current_path, temp_path, method=blur_method)
                        current_path = temp_path
                    
                    # 复制最终结果到会话输出目录
                    shutil.copy2(current_path, output_path)
                    processed_count += 1
                    PROCESSING_SESSIONS[session_id]["processed_files"] = processed_count
                    PROCESSING_SESSIONS[session_id]["progress"] = processed_count / len(files)
                    
                except Exception as e:
                    logger.error(f"处理文件 {file_path.name} 时出错: {e}")
                    # 如果处理失败，复制原文件
                    shutil.copy2(file_path, output_dir / file_path.name)
            else:
                # 非图像文件直接复制
                shutil.copy2(file_path, output_dir / file_path.name)
        
        logger.info(f"会话 {session_id}: 成功处理了 {processed_count} 个图像文件")

        # 将会话输出目录内容复制到目标“服务器”目录
        for f in output_dir.iterdir():
            if f.is_file():
                try:
                    shutil.copy2(f, target_dir / f.name)
                except Exception as e:
                    logger.error(f"复制处理结果 {f.name} 到 {target_dir} 时出错: {e}")
        
        # 创建下载包
        zip_path = session_dir / "processed_files.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in output_dir.iterdir():
                if file_path.is_file():
                    zipf.write(file_path, file_path.name)
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "processed_files": len(list(output_dir.iterdir())),
            "download_url": f"/api/download/{session_id}",
            "target_dir": str(target_dir)
        })
        
    except Exception as e:
        logger.error(f"处理会话 {session_id} 时出错: {e}")
        # 清理临时文件
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/api/download/{session_id}")
async def download_processed_files(session_id: str):
    """下载处理后的文件"""
    session_dir = TEMP_DIR / session_id
    zip_path = session_dir / "processed_files.zip"
    
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在或已过期")
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"processed_files_{session_id[:8]}.zip"
    )

@app.delete("/api/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """清理会话文件"""
    session_dir = TEMP_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
        return {"success": True, "message": "会话文件已清理"}
    return {"success": False, "message": "会话不存在"}

@app.get("/api/health")
@app.head("/api/health")
async def health_check():
    """健康检查 - 支持GET和HEAD请求"""
    return {"status": "healthy", "service": "data-anonymization"}

@app.get("/api/status/{session_id}")
async def check_processing_status(session_id: str):
    """检查处理状态"""
    if session_id not in PROCESSING_SESSIONS:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session_status = PROCESSING_SESSIONS[session_id]
    
    # 如果处理完成，返回下载链接
    if session_status["progress"] >= 1:
        session_status["status"] = "completed"
        session_status["download_url"] = f"/api/download/{session_id}"
    
    return session_status

if __name__ == "__main__":
    # 配置日志
    # Use stdout for Docker logging
    logger.add(sys.stdout, colorize=True)
    
    # 启动服务器
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
