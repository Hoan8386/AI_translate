"""
Stage 10: Video Renderer
==========================
Ghép tất cả lại thành video cuối cùng.

Input:  video_lipsync.mp4 + final_audio.wav
Output: output.mp4
"""

import subprocess
from pathlib import Path
from typing import Optional

from utils.logger import get_logger, log_stage
from utils.timer import Timer

logger = get_logger(__name__)


class VideoRenderer:
    """
    Render video cuối cùng.
    
    Ghép video (đã lip sync) + audio (tiếng Việt) thành output.mp4
    """
    
    STAGE_NUM = 10
    STAGE_NAME = "Video Renderer"
    
    def process(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        original_video: Optional[str] = None,
    ) -> str:
        """
        Render video cuối cùng.
        
        Args:
            video_path: Video (đã lip sync hoặc original)
            audio_path: Audio tiếng Việt (merged)
            output_path: Đường dẫn output
            original_video: Video gốc (để lấy subtitle track nếu có)
            
        Returns:
            Đường dẫn output video
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")
        
        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            # Kiểm tra files
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video không tồn tại: {video_path}")
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio không tồn tại: {audio_path}")
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Render
            self._render(video_path, audio_path, output_path)
            
            # Verify
            if not Path(output_path).exists():
                raise RuntimeError(f"Render thất bại, output không tồn tại: {output_path}")
            
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"Output video: {output_path} ({size_mb:.1f}MB)")
        
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return output_path
    
    def _render(self, video_path: str, audio_path: str, output_path: str):
        """
        Ghép video + audio bằng ffmpeg.
        """
        cmd = [
            "ffmpeg",
            "-i", str(video_path),       # Video input
            "-i", str(audio_path),        # Audio input
            "-c:v", "libx264",            # Re-encode video (H264)
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",                # Audio AAC
            "-b:a", "192k",
            "-map", "0:v:0",             # Lấy video từ input 1
            "-map", "1:a:0",             # Lấy audio từ input 2
            "-shortest",                  # Dừng ở stream ngắn nhất
            "-movflags", "+faststart",    # Optimize cho web playback
            "-y",                         # Overwrite
            str(output_path),
        ]
        
        logger.info("Rendering final video...")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg render error:\n{result.stderr}")
        
        logger.info("Render hoàn thành!")
    
    def add_subtitles(
        self,
        video_path: str,
        srt_path: str,
        output_path: str,
    ) -> str:
        """
        Thêm subtitle vào video (optional, cho phiên bản sau).
        
        Args:
            video_path: Video input
            srt_path: File subtitle .srt
            output_path: Output video
            
        Returns:
            Đường dẫn output
        """
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"subtitles={srt_path}",
            "-c:a", "copy",
            "-y",
            str(output_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            raise RuntimeError(f"Subtitle error:\n{result.stderr}")
        
        return output_path
