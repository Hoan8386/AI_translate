"""
Stage 2: Audio Extractor
==========================
Trích xuất audio từ video.

Input:  normalized_video.mp4
Output: audio.wav (16KHz, Mono)
"""

import subprocess
from pathlib import Path

from utils.logger import get_logger, log_stage
from utils.timer import Timer

logger = get_logger(__name__)


class AudioExtractor:
    """Trích xuất audio từ video chuẩn hóa"""
    
    STAGE_NUM = 2
    STAGE_NAME = "Audio Extractor"
    
    # Chuẩn output
    SAMPLE_RATE = 16000
    CHANNELS = 1  # Mono
    FORMAT = "wav"
    
    def process(self, video_path: str, output_path: str) -> str:
        """
        Trích xuất audio từ video.
        
        Args:
            video_path: Đường dẫn video (đã chuẩn hóa)
            output_path: Đường dẫn output audio (wav)
            
        Returns:
            Đường dẫn audio file
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")
        
        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            input_file = Path(video_path)
            if not input_file.exists():
                raise FileNotFoundError(f"Video không tồn tại: {video_path}")
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract audio bằng ffmpeg
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",                          # Bỏ video
                "-acodec", "pcm_s16le",         # PCM 16-bit
                "-ar", str(self.SAMPLE_RATE),    # 16KHz
                "-ac", str(self.CHANNELS),       # Mono
                "-y",                            # Overwrite
                str(output_path)
            ]
            
            logger.info(f"Extracting audio: {self.SAMPLE_RATE}Hz, Mono, WAV")
            logger.debug(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg error:\n{result.stderr}")
            
            # Verify output
            if not output_file.exists():
                raise RuntimeError(f"Audio file không được tạo: {output_path}")
            
            size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"Audio extracted: {output_path} ({size_mb:.1f}MB)")
        
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return output_path
