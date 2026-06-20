"""
Stage 9: Lip Sync
===================
Đồng bộ khẩu hình video với audio mới.

Đây là phần NẶNG NHẤT.

Input:  video.mp4 + final_audio.wav
Output: video_lipsync.mp4

Model: Wav2Lip (KHÔNG dùng SadTalker vì quá nặng)
"""

import os
import sys
import subprocess
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from utils.logger import get_logger, log_stage
from utils.timer import Timer
from utils.gpu_manager import GPUManager
from config.settings import get_settings

logger = get_logger(__name__)


class LipSyncer:
    """
    Đồng bộ khẩu hình video với Wav2Lip.
    
    Wav2Lip nhẹ hơn SadTalker, phù hợp RTX 5060 8GB.
    
    Quy trình:
    1. Load Wav2Lip model
    2. Detect face trong video
    3. Generate lip movements matching audio
    4. Render output video
    """
    
    STAGE_NUM = 9
    STAGE_NAME = "Lip Sync (Wav2Lip)"
    
    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()
    
    def _get_wav2lip_dir(self) -> Path:
        """Lấy đường dẫn thư mục Wav2Lip"""
        wav2lip_dir = self.settings.third_party_dir / "Wav2Lip"
        if not wav2lip_dir.exists():
            raise FileNotFoundError(
                f"Wav2Lip chưa được cài đặt tại: {wav2lip_dir}\n"
                f"Chạy setup_env.bat để cài đặt."
            )
        return wav2lip_dir
    
    def _check_model_weights(self) -> str:
        """
        Kiểm tra model weights đã tải chưa.
        
        Returns:
            Đường dẫn checkpoint file
        """
        checkpoint = self.settings.lipsync.checkpoint_path
        
        if not os.path.exists(checkpoint):
            # Thử tìm trong thư mục Wav2Lip
            wav2lip_dir = self._get_wav2lip_dir()
            alt_paths = [
                wav2lip_dir / "checkpoints" / "wav2lip_gan.pth",
                wav2lip_dir / "checkpoints" / "wav2lip.pth",
                self.settings.models_dir / "wav2lip" / "wav2lip_gan.pth",
                self.settings.models_dir / "wav2lip" / "wav2lip.pth",
            ]
            
            checkpoint = None
            for alt in alt_paths:
                if alt.exists():
                    checkpoint = str(alt)
                    break
            
            if checkpoint is None:
                raise FileNotFoundError(
                    "Wav2Lip model weights chưa được tải!\n"
                    "Tải từ:\n"
                    "  - wav2lip_gan.pth: https://github.com/Rudrabha/Wav2Lip#getting-the-weights\n"
                    f"  - Lưu vào: {self.settings.models_dir / 'wav2lip' / 'wav2lip_gan.pth'}"
                )
        
        logger.info(f"Wav2Lip checkpoint: {checkpoint}")
        return checkpoint
    
    def process(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
    ) -> str:
        """
        Chạy lip sync.
        
        Args:
            video_path: Video gốc (đã chuẩn hóa)
            audio_path: Audio mới (merged Vietnamese audio)
            output_path: Đường dẫn output video
            
        Returns:
            Đường dẫn video đã lip sync
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")
        
        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            # Kiểm tra files
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video không tồn tại: {video_path}")
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio không tồn tại: {audio_path}")
            
            # Kiểm tra model weights
            checkpoint = self._check_model_weights()
            
            # Đảm bảo output dir tồn tại
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure GPU free cho Wav2Lip (nặng nhất)
            self.gpu.ensure_free(4000)
            
            try:
                # Chạy Wav2Lip inference
                self._run_wav2lip(video_path, audio_path, output_path, checkpoint)
                
                if not Path(output_path).exists():
                    raise RuntimeError("Wav2Lip không tạo được output video")
                
                size_mb = Path(output_path).stat().st_size / (1024 * 1024)
                logger.info(f"Lip sync output: {output_path} ({size_mb:.1f}MB)")
                
            finally:
                # Cleanup GPU
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Wav2Lip GPU resources released")
        
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return output_path
    
    def _run_wav2lip(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        checkpoint: str,
    ):
        """
        Chạy Wav2Lip inference qua subprocess.
        
        Dùng subprocess thay vì import trực tiếp để:
        - Tránh conflict dependencies
        - Dễ quản lý GPU memory (process tự giải phóng khi kết thúc)
        """
        wav2lip_dir = self._get_wav2lip_dir()
        
        cmd = [
            sys.executable,  # Python hiện tại (trong venv)
            str(wav2lip_dir / "inference.py"),
            "--checkpoint_path", checkpoint,
            "--face", str(video_path),
            "--audio", str(audio_path),
            "--outfile", str(output_path),
            "--resize_factor", str(self.settings.lipsync.resize_factor),
            "--face_det_batch_size", str(self.settings.lipsync.face_det_batch_size),
            "--wav2lip_batch_size", str(self.settings.lipsync.wav2lip_batch_size),
            "--pads",
            str(self.settings.lipsync.pads[0]),
            str(self.settings.lipsync.pads[1]),
            str(self.settings.lipsync.pads[2]),
            str(self.settings.lipsync.pads[3]),
        ]
        
        logger.info("Running Wav2Lip inference...")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(wav2lip_dir),
            timeout=1800,  # 30 min timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Wav2Lip stderr:\n{result.stderr}")
            raise RuntimeError(f"Wav2Lip inference failed:\n{result.stderr[-500:]}")
        
        if result.stdout:
            logger.debug(f"Wav2Lip stdout:\n{result.stdout[-500:]}")
    
    def process_simple(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
    ) -> str:
        """
        Phương án đơn giản: chỉ thay audio, không lip sync.
        
        Dùng khi Wav2Lip không khả dụng hoặc video không có face.
        """
        log_stage(self.STAGE_NUM, f"{self.STAGE_NAME} (Simple)", "START")
        
        logger.info("Fallback: Thay audio mà không lip sync")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y",
            str(output_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error:\n{result.stderr}")
        
        logger.info(f"Simple audio replacement → {output_path}")
        log_stage(self.STAGE_NUM, f"{self.STAGE_NAME} (Simple)", "DONE")
        
        return output_path
