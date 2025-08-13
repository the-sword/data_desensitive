#!/usr/bin/env python3
"""
数据脱敏系统主应用
支持人脸、车牌、街牌检测与模糊处理
"""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional
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

# 全局配置
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")

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
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_agreement(request: Request):
    """隐私协议页面"""
    return templates.TemplateResponse("privacy.html", {"request": request})

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
    
    input_dir = session_dir / "input"
    output_dir = session_dir / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
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
        
        # 如果不需要脱敏，直接复制文件
        if not enable_anonymization:
            for file_path in uploaded_files:
                shutil.copy2(file_path, output_dir / file_path.name)
            logger.info(f"会话 {session_id}: 跳过脱敏处理")
        else:
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
                        
                        # 复制最终结果
                        shutil.copy2(current_path, output_path)
                        processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"处理文件 {file_path.name} 时出错: {e}")
                        # 如果处理失败，复制原文件
                        shutil.copy2(file_path, output_dir / file_path.name)
                else:
                    # 非图像文件直接复制
                    shutil.copy2(file_path, output_dir / file_path.name)
            
            logger.info(f"会话 {session_id}: 成功处理了 {processed_count} 个图像文件")
        
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
            "download_url": f"/api/download/{session_id}"
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

if __name__ == "__main__":
    # 配置日志
    logger.add("logs/app.log", rotation="1 day", retention="7 days")
    
    # 启动服务器
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
