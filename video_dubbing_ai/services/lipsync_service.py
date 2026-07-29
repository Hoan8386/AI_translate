"""
Lip Sync Service
==================
Service layer cho Lip Sync (Stage 9).
"""

import subprocess
from pathlib import Path

from pipeline.p09_lipsync import LipSyncProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


class LipSyncService:
    """Service wrapper cho LipSyncProcessor"""
    
    def __init__(self):
        self.syncer = LipSyncProcessor()
    
    def sync(self, video_path: str, audio_path: str, output_path: str) -> str:
        """
        Đồng bộ khẩu hình.
        
        Args:
            video_path: Video gốc
            audio_path: Audio tiếng Việt
            output_path: Output path
            
        Returns:
            Đường dẫn video đã lip sync
        """
        try:
            return self.syncer.process(video_path, audio_path, output_path)
        except Exception as e:
            logger.warning(f"Lip sync thất bại: {e}")
            logger.info("Fallback: thay audio không lip sync")
            return self.simple_replace(video_path, audio_path, output_path)
    
    def simple_replace(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Thay audio đơn giản (không lip sync) bằng ffmpeg"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path,
        ]
        
        logger.info(f"Simple audio replace: {Path(video_path).name} + {Path(audio_path).name}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
        
        logger.info(f"Simple replace done: {output_path}")
        return output_path
