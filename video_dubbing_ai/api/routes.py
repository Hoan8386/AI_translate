"""
API Routes
============
REST API endpoints cho hệ thống dubbing.
"""

import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from api import pipeline_runner
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["dubbing"])

# Giới hạn upload
MAX_UPLOAD_SIZE_MB = 500
ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload video tiếng Trung.
    
    Returns:
        job_id và thông tin file
    """
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ: {ext}. Chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Tạo job ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(file.filename).stem
    job_id = f"{safe_name}_{timestamp}"
    
    # Lưu file
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = INPUT_DIR / f"{job_id}{ext}"
    
    try:
        # Đọc và ghi file theo chunks để hỗ trợ file lớn
        total_size = 0
        max_size = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        
        with open(save_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > max_size:
                    # Xóa file đã ghi dở
                    save_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File quá lớn. Giới hạn: {MAX_UPLOAD_SIZE_MB}MB"
                    )
                f.write(chunk)
        
        # Tạo job
        job_info = pipeline_runner.create_job(
            job_id=job_id,
            input_path=str(save_path),
            filename=file.filename,
        )
        
        size_mb = total_size / (1024 * 1024)
        logger.info(f"Uploaded: {file.filename} ({size_mb:.1f}MB) → Job: {job_id}")
        
        return {
            "job_id": job_id,
            "filename": file.filename,
            "size_mb": round(size_mb, 1),
            "status": "pending",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload thất bại: {str(e)}")


@router.post("/jobs/{job_id}/start")
async def start_job(
    job_id: str,
    skip_lipsync: bool = Query(default=False, description="Bỏ qua lip sync"),
    translation_provider: str = Query(default="gemini", description="gemini hoặc openai"),
):
    """
    Bắt đầu chạy pipeline cho job đã upload.
    """
    job_info = pipeline_runner.get_job_info(job_id)
    if job_info is None:
        raise HTTPException(status_code=404, detail=f"Job không tồn tại: {job_id}")
    
    if job_info["status"] == "processing":
        raise HTTPException(status_code=409, detail="Job đang xử lý")
    
    if job_info["status"] == "completed":
        raise HTTPException(status_code=409, detail="Job đã hoàn thành. Tạo job mới để chạy lại.")
    
    # Lấy event loop hiện tại để pipeline runner gửi WebSocket updates
    loop = asyncio.get_event_loop()
    
    pipeline_runner.start_pipeline(
        job_id=job_id,
        skip_lipsync=skip_lipsync,
        translation_provider=translation_provider,
        event_loop=loop,
    )
    
    return {"job_id": job_id, "status": "processing", "message": "Pipeline đã bắt đầu"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Lấy trạng thái chi tiết của job"""
    job_info = pipeline_runner.get_job_info(job_id)
    if job_info is None:
        raise HTTPException(status_code=404, detail=f"Job không tồn tại: {job_id}")
    
    return job_info


@router.get("/jobs")
async def list_jobs():
    """Liệt kê tất cả jobs"""
    jobs = pipeline_runner.get_all_jobs()
    # Trả về danh sách tóm tắt (không bao gồm logs và segments chi tiết)
    summary = []
    for job_id, info in jobs.items():
        summary.append({
            "id": info["id"],
            "filename": info["filename"],
            "status": info["status"],
            "progress": info["progress"],
            "current_stage": info["current_stage"],
            "created_at": info["created_at"],
        })
    return {"jobs": summary}


@router.get("/jobs/{job_id}/download")
async def download_result(job_id: str):
    """Tải video kết quả"""
    job_info = pipeline_runner.get_job_info(job_id)
    if job_info is None:
        raise HTTPException(status_code=404, detail=f"Job không tồn tại: {job_id}")
    
    if job_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job chưa hoàn thành")
    
    output_path = job_info.get("output_path", "")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="File output không tìm thấy")
    
    return FileResponse(
        path=output_path,
        filename=Path(output_path).name,
        media_type="video/mp4",
    )


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Xóa job và file tạm"""
    job_info = pipeline_runner.get_job_info(job_id)
    if job_info is None:
        raise HTTPException(status_code=404, detail=f"Job không tồn tại: {job_id}")
    
    if job_info["status"] == "processing":
        raise HTTPException(status_code=409, detail="Không thể xóa job đang xử lý")
    
    # Xóa file input
    input_path = Path(job_info["input_path"])
    if input_path.exists():
        input_path.unlink()
    
    # Xóa job directory trong temp
    temp_job_dir = PROJECT_ROOT / "temp" / job_id
    if temp_job_dir.exists():
        shutil.rmtree(temp_job_dir, ignore_errors=True)
    
    # Xóa khỏi memory
    pipeline_runner.delete_job(job_id)
    
    return {"message": f"Job {job_id} đã được xóa"}
