"""
Stage 1: Video Processor
=========================
Kiểm tra định dạng và chuẩn hóa video.

Input:  video.mp4 (bất kỳ format)
Output: normalized_video.mp4 (H264, 30 FPS, AAC)
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional

from utils.logger import get_logger, log_stage
from utils.timer import Timer

logger = get_logger(__name__)


class VideoProcessor:
    """Kiểm tra và chuẩn hóa video đầu vào"""
    
    STAGE_NUM = 1
    STAGE_NAME = "Video Processor"
    
    # Chuẩn output
    TARGET_CODEC = "libx264"
    TARGET_FPS = 30
    TARGET_AUDIO_CODEC = "aac"
    TARGET_PRESET = "medium"
    TARGET_CRF = 23
    
    def __init__(self):
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Kiểm tra ffmpeg đã được cài đặt"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("ffmpeg không khả dụng")
            logger.debug("ffmpeg OK")
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg không được tìm thấy! "
                "Vui lòng cài đặt ffmpeg: https://ffmpeg.org/download.html"
            )
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Lấy thông tin chi tiết video.
        
        Returns:
            Dict với thông tin: codec, fps, resolution, duration, etc.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise ValueError(f"Không thể đọc video: {video_path}\n{result.stderr}")
        
        probe = json.loads(result.stdout)
        
        # Tìm video stream
        video_stream = None
        audio_stream = None
        for stream in probe.get("streams", []):
            if stream["codec_type"] == "video" and video_stream is None:
                video_stream = stream
            elif stream["codec_type"] == "audio" and audio_stream is None:
                audio_stream = stream
        
        if video_stream is None:
            raise ValueError(f"Không tìm thấy video stream trong: {video_path}")
        
        # Parse FPS
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        
        info = {
            "path": str(video_path),
            "duration": float(probe.get("format", {}).get("duration", 0)),
            "size_mb": float(probe.get("format", {}).get("size", 0)) / (1024 * 1024),
            "video_codec": video_stream.get("codec_name", "unknown"),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": round(fps, 2),
            "audio_codec": audio_stream.get("codec_name", "unknown") if audio_stream else "none",
            "audio_sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
            "has_audio": audio_stream is not None,
        }
        
        return info
    
    def needs_normalization(self, info: Dict[str, Any]) -> bool:
        """Kiểm tra video có cần chuẩn hóa không"""
        needs = False
        reasons = []
        
        if info["video_codec"] != "h264":
            reasons.append(f"codec: {info['video_codec']} → h264")
            needs = True
        
        if abs(info["fps"] - self.TARGET_FPS) > 1:
            reasons.append(f"fps: {info['fps']} → {self.TARGET_FPS}")
            needs = True
        
        if info["audio_codec"] != "aac" and info["has_audio"]:
            reasons.append(f"audio: {info['audio_codec']} → aac")
            needs = True
        
        if reasons:
            logger.info(f"Cần chuẩn hóa: {', '.join(reasons)}")
        else:
            logger.info("Video đã đúng chuẩn")
        
        return needs
    
    def process(self, input_path: str, output_path: str) -> str:
        """
        Chuẩn hóa video.
        
        Args:
            input_path: Đường dẫn video input
            output_path: Đường dẫn video output (chuẩn hóa)
            
        Returns:
            Đường dẫn video đã chuẩn hóa
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")
        
        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}") as timer:
            input_file = Path(input_path)
            
            # Kiểm tra file tồn tại
            if not input_file.exists():
                raise FileNotFoundError(f"Video không tồn tại: {input_path}")
            
            # Lấy thông tin video
            info = self.get_video_info(input_path)
            logger.info(
                f"Video info: {info['width']}x{info['height']}, "
                f"{info['fps']}fps, {info['video_codec']}, "
                f"{info['duration']:.1f}s, {info['size_mb']:.1f}MB"
            )
            
            if not info["has_audio"]:
                raise ValueError("Video không có audio track!")
            
            # Kiểm tra có cần normalize không
            if not self.needs_normalization(info):
                # Copy trực tiếp nếu đã chuẩn
                import shutil
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_path, output_path)
                logger.info(f"Video đã chuẩn, copy trực tiếp → {output_path}")
            else:
                # Chuẩn hóa bằng ffmpeg
                self._normalize(input_path, output_path)
            
            # Verify output
            output_info = self.get_video_info(output_path)
            logger.info(
                f"Output: {output_info['width']}x{output_info['height']}, "
                f"{output_info['fps']}fps, {output_info['video_codec']}"
            )
        
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return output_path
    
    def _normalize(self, input_path: str, output_path: str):
        """Chạy ffmpeg để chuẩn hóa video"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-c:v", self.TARGET_CODEC,
            "-preset", self.TARGET_PRESET,
            "-crf", str(self.TARGET_CRF),
            "-r", str(self.TARGET_FPS),
            "-c:a", self.TARGET_AUDIO_CODEC,
            "-y",  # Overwrite
            str(output_path)
        ]
        
        logger.info(f"Normalizing video...")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error:\n{result.stderr}")
        
        logger.info(f"Normalized → {output_path}")
