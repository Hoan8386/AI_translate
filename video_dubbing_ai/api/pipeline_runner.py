"""
Pipeline Runner - Tối ưu hóa VRAM cho RTX 5060 8GB
==================================================
Bộ điều phối trung tâm quản lý vòng đời của các mô hình AI trên GPU.
Đảm bảo tuân thủ nguyên tắc: CHỈ 1 MODEL TRÊN VRAM TẠI 1 THỜI ĐIỂM.

Technology:
    Python 3.10 + PyTorch (CUDA 12.1)
"""

import gc
import json
import shutil
from pathlib import Path
import torch

from utils.logger import get_logger, log_stage
from utils.timer import Timer
from config.settings import get_settings

# Import các Stage từ các package trong hệ thống của bạn
from pipeline.p01_video_processor import VideoProcessor
from pipeline.p02_audio_extractor import AudioExtractor
from pipeline.p04_segment_creator import SegmentCreator
from pipeline.p03_speaker_detector import SpeakerDetector
from pipeline.p05_asr import ASRProcessor
from pipeline.p06_translation import Translator
from pipeline.p07_voice_clone import VoiceCloner
from pipeline.p08_audio_alignment import AudioAligner
from pipeline.p09_lipsync import LipSyncProcessor
from pipeline.p10_renderer import VideoRenderer

logger = get_logger(__name__)


class PipelineRunner:
    """
    Quản lý và thực thi tuần tự 10 bước của hệ thống Auto Dubbing,
    chủ động giải phóng VRAM ngay sau khi kết thúc từng bước AI.
    """

    def __init__(self):
        self.settings = get_settings()

    def run(self, input_video_path: str) -> str:
        """
        Kích hoạt luồng chạy toàn bộ Pipeline xử lý video.
        
        Args:
            input_video_path: Đường dẫn tới file video gốc (.mp4)
            
        Returns:
            Đường dẫn tới video output đã được dubbing và render hoàn chỉnh
        """
        logger.info("=======================================================")
        logger.info("   BẮT ĐẦU HỆ THỐNG DUBBING AI CHUYÊN NGHIỆP (LOCAL)   ")
        logger.info("=======================================================")

        video_p = Path(input_video_path)
        if not video_p.exists():
            raise FileNotFoundError(f"Video đầu vào không tồn tại: {input_video_path}")

        # Khởi tạo các biến lưu trữ dữ liệu trung gian qua các Stage
        normalized_video = None
        extracted_audio = None
        speaker_segments = None
        structured_segments = None
        asr_segments = None
        translated_segments = None
        cloned_audio_segments = None
        aligned_audio_path = None
        synced_video_path = None
        final_output_path = None

        try:
            # -----------------------------------------------------------------
            # Stage 1: Video Processor
            # -----------------------------------------------------------------
            p01 = VideoProcessor()
            normalized_video = p01.process(str(video_p))
            self._persist_stage_artifacts(
                1,
                "Video Processor",
                [str(video_p)],
                [normalized_video],
                {"source_video": str(video_p), "normalized_video": normalized_video},
            )

            # -----------------------------------------------------------------
            # Stage 2: Audio Extractor
            # -----------------------------------------------------------------
            p02 = AudioExtractor()
            extracted_audio_name = f"{video_p.stem}_extracted.wav"
            target_audio_path = str(self.settings.temp_dir / extracted_audio_name)
            extracted_audio = p02.process(normalized_video, target_audio_path)
            self._persist_stage_artifacts(
                2,
                "Audio Extractor",
                [normalized_video],
                [extracted_audio],
                {"source_video": normalized_video, "extracted_audio": extracted_audio},
            )

            # -----------------------------------------------------------------
            # Stage 3: Speaker Detector (Cần ~2GB VRAM - Kéo mô hình Pyannote)
            # -----------------------------------------------------------------
            p03 = SpeakerDetector()
            speaker_segments = p03.process(extracted_audio)
            self._persist_stage_artifacts(
                3,
                "Speaker Detector",
                [extracted_audio],
                [],
                {"speaker_segments": [seg if isinstance(seg, dict) else seg.to_dict() for seg in speaker_segments]},
            )
            SpeakerDetector.unload_model()

            # -----------------------------------------------------------------
            # Stage 4: Segment Creator
            # -----------------------------------------------------------------
            p04 = SegmentCreator()
            structured_segments = p04.process(speaker_segments)
            self._persist_stage_artifacts(
                4,
                "Segment Creator",
                [],
                [],
                {"segments": [seg.to_dict() for seg in structured_segments]},
            )

            # -----------------------------------------------------------------
            # Stage 5: ASR - SenseVoice (Cần ~1.5GB VRAM - Trích xuất tiếng Trung)
            # -----------------------------------------------------------------
            p05 = ASRProcessor()
            asr_segments = p05.process(extracted_audio, structured_segments)
            self._persist_stage_artifacts(
                5,
                "ASR",
                [extracted_audio],
                [],
                {"segments": [seg.to_dict() for seg in asr_segments]},
            )
            ASRProcessor.unload_model()

            # -----------------------------------------------------------------
            # Stage 6: Translation Local - Qwen2.5 (Cần ~2GB - 3GB VRAM)
            # -----------------------------------------------------------------
            p06 = Translator()
            translated_segments = p06.process(asr_segments)
            self._persist_stage_artifacts(
                6,
                "Translation",
                [],
                [],
                {"segments": [seg.to_dict() for seg in translated_segments]},
            )
            Translator.unload_model()

            # -----------------------------------------------------------------
            # Stage 7: Voice Clone - Fish Speech Local (Cần ~4GB - 6GB VRAM)
            # -----------------------------------------------------------------
            p07 = VoiceCloner()
            cloned_audio_segments = p07.process(translated_segments)
            self._persist_stage_artifacts(
                7,
                "Voice Cloning",
                [],
                [seg.generated_audio for seg in cloned_audio_segments if getattr(seg, "generated_audio", "")],
                {"segments": [seg.to_dict() for seg in cloned_audio_segments]},
            )
            if hasattr(VoiceCloner, 'unload_model'):
                VoiceCloner.unload_model()

            # -----------------------------------------------------------------
            # Stage 8: Audio Alignment
            # -----------------------------------------------------------------
            p08 = AudioAligner()
            stage8_output_dir = self.settings.output_dir / "step_8_audio_alignment"
            stage8_merged_audio = stage8_output_dir / "merged_audio.wav"
            p08.process(cloned_audio_segments, str(stage8_output_dir), str(stage8_merged_audio))
            aligned_audio_path = str(stage8_merged_audio)
            self._persist_stage_artifacts(
                8,
                "Audio Alignment",
                [seg.generated_audio for seg in cloned_audio_segments if getattr(seg, "generated_audio", "")],
                [str(stage8_merged_audio)],
                {"segments": [seg.to_dict() for seg in cloned_audio_segments], "merged_audio": str(stage8_merged_audio)},
            )

            # -----------------------------------------------------------------
            # Stage 9: Lip Sync - Wav2Lip Local (Cần ~3GB - 4GB VRAM)
            # -----------------------------------------------------------------
            p09 = LipSyncProcessor()
            stage9_output_path = str(self.settings.output_dir / "step_9_lipsync" / f"{video_p.stem}_lipsync.mp4")
            synced_video_path = p09.process(normalized_video, aligned_audio_path, stage9_output_path)
            self._persist_stage_artifacts(
                9,
                "Lip Sync",
                [normalized_video, aligned_audio_path],
                [stage9_output_path],
                {"input_video": normalized_video, "input_audio": aligned_audio_path, "output_video": stage9_output_path},
            )

            # GIẢI PHÓNG VRAM NGAY LẬP TỨC
            if hasattr(LipSyncProcessor, 'unload_model'):
                LipSyncProcessor.unload_model()

            # -----------------------------------------------------------------
            # Stage 10: Renderer
            # -----------------------------------------------------------------
            p10 = VideoRenderer()
            final_output_name = f"{video_p.stem}_dubbed_final.mp4"
            target_output_path = str(self.settings.output_dir / final_output_name)
            final_output_path = p10.process(synced_video_path, aligned_audio_path, target_output_path)
            self._persist_stage_artifacts(
                10,
                "Video Renderer",
                [synced_video_path, aligned_audio_path],
                [target_output_path],
                {"input_video": synced_video_path, "input_audio": aligned_audio_path, "output_video": target_output_path},
            )

            logger.info("=======================================================")
            logger.info(f"    PIPELINE HOÀN THÀNH XUẤT SẮC: {final_output_path}   ")
            logger.info("=======================================================")
            
            return final_output_path

        except Exception as e:
            logger.error(f"[CRITICAL ERROR] Pipeline dừng hoạt động đột ngột tại một Stage: {e}")
            # Thực hiện cứu hộ dọn dẹp bộ nhớ khẩn cấp phòng trừ dính treo tài nguyên GPU
            self._emergency_clean_gpu()
            raise e

    def _persist_stage_artifacts(
        self,
        stage_num: int,
        stage_name: str,
        input_paths: list | None = None,
        output_paths: list | None = None,
        payload: object | None = None,
        job_id: str | None = None,
    ) -> Path:
        """Ghi thông tin và file trung gian của từng stage vào thư mục output.
        
        Nếu job_id được cung cấp, lưu vào output/stages/<job_id>/stage_XX_name/
        Nếu không, lưu vào output/stage_XX_name/ (fallback tương thích cũ)
        """
        safe_name = stage_name.lower().replace(' ', '_')
        if job_id:
            stage_dir = (self.settings.output_stages_dir / job_id
                         / f"stage_{stage_num:02d}_{safe_name}")
        else:
            stage_dir = self.settings.output_dir / f"stage_{stage_num:02d}_{safe_name}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        for item in input_paths or []:
            if not item:
                continue
            source = Path(item)
            if source.exists() and source.is_file():
                destination = stage_dir / source.name
                if destination.exists() and destination.resolve() != source.resolve():
                    destination.unlink()
                if destination.exists() and destination.resolve() == source.resolve():
                    continue
                shutil.copy2(source, destination)

        for item in output_paths or []:
            if not item:
                continue
            source = Path(item)
            if source.exists() and source.is_file():
                destination = stage_dir / source.name
                if destination.exists() and destination.resolve() != source.resolve():
                    destination.unlink()
                if destination.exists() and destination.resolve() == source.resolve():
                    continue
                shutil.copy2(source, destination)

        if payload is not None:
            with (stage_dir / "data.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)

        manifest = {
            "stage": stage_num,
            "name": stage_name,
            "output_dir": str(stage_dir),
            "input_paths": [str(item) for item in input_paths or [] if item],
            "output_paths": [str(item) for item in output_paths or [] if item],
        }
        with (stage_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

        return stage_dir

    def _emergency_clean_gpu(self):
        """
        Dọn dẹp VRAM cưỡng bức trong trường hợp xảy ra ngoại lệ (Exception) 
        để tránh GPU bị khóa bộ nhớ ở các tiến trình chạy sau.
        """
        logger.warning("Đang chạy tiến trình dọn dẹp VRAM khẩn cấp (Emergency Clean)...")
        try:
            SpeakerDetector.unload_model()
        except: pass
        try:
            ASRProcessor.unload_model()
        except: pass
        try:
            Translator.unload_model()
        except: pass
        try:
            if hasattr(VoiceCloner, 'unload_model'): VoiceCloner.unload_model()
        except: pass
        try:
            if hasattr(LipSyncProcessor, 'unload_model'): LipSyncProcessor.unload_model()
        except: pass

        # Thu dọn rác hệ thống
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Dọn dẹp khẩn cấp GPU hoàn tất.")


# =====================================================================
# Module-level Job Management (dùng bởi api/routes.py)
# =====================================================================

import asyncio
import threading
import traceback
from datetime import datetime
from typing import Optional

from api.websocket import manager as ws_manager

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
        "stage_data": {},   # {"1": {...}, "2": {...}} - dữ liệu từng stage
        "stage_timings": {},  # {"1": {"start": ..., "end": ..., "duration_s": ...}}
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
                          vram_model: str = "", stage_data: dict = None,
                          timing: dict = None):
        """Callback được gọi từ pipeline để báo tiến trình"""
        with _jobs_lock:
            if job_id in _active_jobs:
                _active_jobs[job_id]["current_stage"] = stage
                _active_jobs[job_id]["stage_name"] = stage_name
                _active_jobs[job_id]["progress"] = progress
                if segments:
                    _active_jobs[job_id]["segments"] = segments
                if stage_data:
                    _active_jobs[job_id]["stage_data"][str(stage)] = stage_data
                if timing:
                    _active_jobs[job_id]["stage_timings"][str(stage)] = timing
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
        if stage_data:
            update["stage_data"] = {"stage": stage, "name": stage_name, "data": stage_data}
        if timing:
            update["timing"] = timing

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