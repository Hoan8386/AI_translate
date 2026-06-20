"""
Pipeline Runner
==================
Chạy pipeline dubbing trong background thread,
gửi updates real-time qua WebSocket.
"""

import asyncio
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

from api.websocket import manager as ws_manager
from utils.logger import get_logger

logger = get_logger(__name__)

# Lưu trạng thái jobs đang chạy
# job_id -> job_info dict
_active_jobs = {}
_jobs_lock = threading.Lock()


def get_job_info(job_id: str) -> Optional[dict]:
    """Lấy thông tin job"""
    with _jobs_lock:
        return _active_jobs.get(job_id)


def get_all_jobs() -> dict:
    """Lấy tất cả jobs"""
    with _jobs_lock:
        return dict(_active_jobs)


def create_job(job_id: str, input_path: str, filename: str) -> dict:
    """Tạo job mới (chưa chạy)"""
    job_info = {
        "id": job_id,
        "input_path": input_path,
        "filename": filename,
        "status": "pending",
        "current_stage": 0,
        "stage_name": "",
        "progress": 0,
        "error": "",
        "output_path": "",
        "created_at": datetime.now().isoformat(),
        "started_at": "",
        "completed_at": "",
        "segments": [],
        "logs": [],
    }
    with _jobs_lock:
        _active_jobs[job_id] = job_info
    return job_info


def delete_job(job_id: str) -> bool:
    """Xóa job"""
    with _jobs_lock:
        if job_id in _active_jobs:
            del _active_jobs[job_id]
            return True
        return False


def start_pipeline(job_id: str, skip_lipsync: bool = False,
                   translation_provider: str = "gemini",
                   event_loop: asyncio.AbstractEventLoop = None):
    """
    Bắt đầu chạy pipeline trong background thread.
    
    Args:
        job_id: ID của job đã tạo bằng create_job()
        skip_lipsync: Bỏ qua lip sync
        translation_provider: "gemini" hoặc "openai"
        event_loop: Event loop chính để gửi WebSocket updates
    """
    job_info = get_job_info(job_id)
    if job_info is None:
        raise ValueError(f"Job {job_id} không tồn tại")
    
    if job_info["status"] == "processing":
        raise ValueError(f"Job {job_id} đang chạy")
    
    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(job_id, skip_lipsync, translation_provider, event_loop),
        daemon=True,
        name=f"pipeline-{job_id}",
    )
    thread.start()


def _run_pipeline_thread(job_id: str, skip_lipsync: bool,
                         translation_provider: str,
                         event_loop: asyncio.AbstractEventLoop):
    """Hàm chạy trong background thread"""
    import sys
    import os
    
    # Đảm bảo project root trong path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    job_info = get_job_info(job_id)
    if job_info is None:
        return
    
    input_path = job_info["input_path"]
    
    def send_update(data: dict):
        """Helper gửi WebSocket update"""
        ws_manager.send_to_job_sync(job_id, data, event_loop)
    
    def progress_callback(stage: int, stage_name: str, progress: float,
                          message: str = "", log_type: str = "info",
                          segments: list = None, vram_used: int = 0,
                          vram_model: str = ""):
        """Callback được gọi từ pipeline để báo tiến trình"""
        with _jobs_lock:
            if job_id in _active_jobs:
                _active_jobs[job_id]["current_stage"] = stage
                _active_jobs[job_id]["stage_name"] = stage_name
                _active_jobs[job_id]["progress"] = progress
                if segments:
                    _active_jobs[job_id]["segments"] = segments
                if message:
                    _active_jobs[job_id]["logs"].append({
                        "type": log_type,
                        "message": message,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    })
        
        update = {
            "type": "progress",
            "stage": stage,
            "stage_name": stage_name,
            "progress": progress,
        }
        
        if message:
            update["log"] = {"type": log_type, "message": message}
        if segments:
            update["segments"] = segments
        if vram_used > 0:
            update["vram"] = {"used": vram_used, "model": vram_model}
        
        send_update(update)
    
    try:
        # Đánh dấu bắt đầu
        with _jobs_lock:
            _active_jobs[job_id]["status"] = "processing"
            _active_jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        send_update({
            "type": "status",
            "status": "processing",
            "message": f"Bắt đầu pipeline: {job_info['filename']}",
        })
        
        # Override translation provider nếu cần
        if translation_provider:
            os.environ["TRANSLATION_PROVIDER"] = translation_provider
        
        # Import và chạy pipeline
        from main import process_video
        
        output_path = process_video(
            input_path=input_path,
            skip_lipsync=skip_lipsync,
            progress_callback=progress_callback,
        )
        
        # Hoàn thành
        with _jobs_lock:
            if job_id in _active_jobs:
                _active_jobs[job_id]["status"] = "completed"
                _active_jobs[job_id]["progress"] = 100
                _active_jobs[job_id]["output_path"] = str(output_path) if output_path else ""
                _active_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        send_update({
            "type": "status",
            "status": "completed",
            "progress": 100,
            "output_path": str(output_path) if output_path else "",
            "message": "🎉 DUBBING HOÀN THÀNH!",
        })
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Pipeline failed for job {job_id}: {error_msg}\n{error_trace}")
        
        with _jobs_lock:
            if job_id in _active_jobs:
                _active_jobs[job_id]["status"] = "failed"
                _active_jobs[job_id]["error"] = error_msg
                _active_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        send_update({
            "type": "status",
            "status": "failed",
            "error": error_msg,
            "message": f"❌ Pipeline thất bại: {error_msg}",
        })
