"""
Stage 2: Audio Extractor (Nâng cấp: Lọc nhiễu & Chống lệch pha)
==============================================================
Trích xuất và tiền xử lý âm thanh từ video đã chuẩn hóa.
Tích hợp bộ lọc âm thanh nâng cao để cô lập Vocal sạch và triệt tiêu Phase Shift.

Input:  normalized_video.mp4
Output: audio.wav (16KHz, Mono - Đã lọc nhiễu và nhạc nền nhẹ)

Technology:
    FFmpeg (Advanced Audio Filters: pan, highpass, lowpass, afftdn)
"""

import subprocess
from pathlib import Path

from utils.logger import get_logger, log_stage
from utils.timer import Timer

logger = get_logger(__name__)


class AudioExtractor:
    """Trích xuất và cô lập Vocal sạch từ video chuẩn hóa"""
    
    STAGE_NUM = 2
    STAGE_NAME = "Audio Extractor & Denoise"
    
    # Chuẩn output tối ưu cho các mô hình AI (Pyannote, SenseVoice)
    SAMPLE_RATE = 16000
    FORMAT = "wav"
    
    def process(self, video_path: str, output_path: str) -> str:
        """
        Trích xuất audio, lọc bỏ tạp âm/BGM và xuất file wav Mono sạch pha.
        
        Args:
            video_path: Đường dẫn video đã chuẩn hóa
            output_path: Đường dẫn xuất file audio (.wav)
            
        Returns:
            Đường dẫn file audio output
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")
        
        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            input_file = Path(video_path)
            if not input_file.exists():
                raise FileNotFoundError(f"Video không tồn tại: {video_path}")
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # --- XÂY DỰNG BỘ LỌC ÂM THANH NÂNG CAO (AUDIO FILTER GRAPH) ---
            # 1. pan=mono|c0=FL: Chỉ bốc kênh Front Left gốc để tránh triệt tiêu pha (Phase Cancellation) khi gộp kênh.
            # 2. highpass=f=80: Lọc bỏ các tần số siêu trầm nhiễu nền (< 80Hz) như tiếng ù của máy móc, tiếng gió.
            # 3. lowpass=f=7500: Lọc bỏ các tần số siêu cao (> 7500Hz) như tiếng rít rít, nhiễu điện tử.
            # 4. afftdn=nr=12: Bộ lọc nhiễu dựa trên biến đổi Fourier nhanh (FFT), giảm 12dB tạp âm môi trường/BGM mà không làm méo giọng nói.
            audio_filter = "pan=mono|c0=FL,highpass=f=80,lowpass=f=7500,afftdn=nr=12"
            
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",                            # Loại bỏ luồng hình ảnh video
                "-acodec", "pcm_s16le",           # Ép chuẩn PCM 16-bit không nén
                "-af", audio_filter,              # Áp dụng chuỗi filter xử lý Vocal sạch
                "-ar", str(self.SAMPLE_RATE),      # Hạ tầng tần số lấy mẫu về 16KHz chuẩn AI
                "-y",                              # Tự động ghi đè nếu file đã tồn tại
                str(output_path)
            ]
            
            logger.info("Extracting & Denoising Audio: 16KHz, Mono (Phase-Safe), Advanced Filtered WAV")
            logger.debug(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg error:\n{result.stderr}")
            
            if not output_file.exists():
                raise RuntimeError(f"Audio file không được tạo: {output_path}")
            
            size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"Vocal extracted & isolated: {output_path} ({size_mb:.1f}MB)")

            # Xuất bản lưu trữ theo từng bước (Step Export) để phục vụ debug hệ thống
            try:
                from config.settings import get_settings
                import shutil
                settings = get_settings()
                
                video_p = Path(video_path)
                if video_p.parent.name == "normalized":
                    job_id = video_p.parent.parent.name
                else:
                    job_id = video_p.stem
                    
                step_output_dir = settings.output_dir / "step_2_audio_extractor"
                step_output_dir.mkdir(parents=True, exist_ok=True)
                
                dest_path = step_output_dir / f"{job_id}_vocal_clean.wav"
                shutil.copy2(output_path, dest_path)
                logger.info(f"Exported Isolated Vocal to: {dest_path}")
            except Exception as e:
                logger.warning(f"Failed to export Step 2 output: {e}")
        
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return output_path