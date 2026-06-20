"""
Stage 8: Audio Alignment
===========================
Co/giãn audio sinh ra để khớp thời lượng audio gốc.

Ví dụ:
    Original: 5s
    Generated: 8s
    → Co lại thành 5s (speed=1.6)
"""

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import List

from models_data.segment import Segment
from utils.logger import get_logger, log_stage
from utils.timer import Timer

logger = get_logger(__name__)


class AudioAligner:
    """
    Điều chỉnh tốc độ audio để khớp thời lượng gốc.
    
    Sử dụng librosa time-stretch để thay đổi tốc độ
    mà không ảnh hưởng pitch (cao độ).
    """
    
    STAGE_NUM = 8
    STAGE_NAME = "Audio Alignment"
    
    # Giới hạn speed factor
    MIN_SPEED = 0.5   # Không chậm quá 2x
    MAX_SPEED = 2.5   # Không nhanh quá 2.5x
    
    def process(
        self,
        segments: List[Segment],
        output_dir: str,
    ) -> List[Segment]:
        """
        Align audio cho tất cả segments.
        
        Args:
            segments: Danh sách Segment đã có generated_audio
            output_dir: Thư mục lưu aligned audio
            
        Returns:
            Danh sách Segment đã có aligned_audio
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")
        
        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for i, segment in enumerate(segments):
                if not segment.generated_audio:
                    logger.debug(f"Segment {segment.id}: không có generated_audio, bỏ qua")
                    continue
                
                if not Path(segment.generated_audio).exists():
                    logger.warning(
                        f"Segment {segment.id}: generated_audio không tồn tại: "
                        f"{segment.generated_audio}"
                    )
                    continue
                
                # Tính target duration
                target_duration = segment.duration
                
                # Đọc generated audio
                gen_audio, gen_sr = sf.read(segment.generated_audio)
                gen_duration = len(gen_audio) / gen_sr
                
                # Tính speed factor
                if gen_duration <= 0 or target_duration <= 0:
                    logger.warning(f"Segment {segment.id}: duration invalid")
                    continue
                
                speed_factor = gen_duration / target_duration
                
                logger.info(
                    f"Align [{i+1}/{len(segments)}]: Segment {segment.id} "
                    f"({gen_duration:.1f}s → {target_duration:.1f}s, "
                    f"speed={speed_factor:.2f}x)"
                )
                
                # Output path
                aligned_filename = f"aligned_{segment.id:03d}.wav"
                aligned_path = str(output_path / aligned_filename)
                
                if abs(speed_factor - 1.0) < 0.05:
                    # Gần bằng nhau, không cần điều chỉnh
                    import shutil
                    shutil.copy2(segment.generated_audio, aligned_path)
                    logger.info(f"  → Không cần điều chỉnh (speed ≈ 1.0)")
                else:
                    # Clamp speed factor
                    clamped_speed = max(self.MIN_SPEED, min(self.MAX_SPEED, speed_factor))
                    if clamped_speed != speed_factor:
                        logger.warning(
                            f"  Speed factor {speed_factor:.2f} ngoài giới hạn, "
                            f"clamp thành {clamped_speed:.2f}"
                        )
                        speed_factor = clamped_speed
                    
                    # Time-stretch
                    aligned_audio = self._time_stretch(gen_audio, gen_sr, speed_factor)
                    
                    # Trim/pad để khớp chính xác target duration
                    aligned_audio = self._match_duration(
                        aligned_audio, gen_sr, target_duration
                    )
                    
                    # Lưu
                    sf.write(aligned_path, aligned_audio, gen_sr)
                    
                    actual_duration = len(aligned_audio) / gen_sr
                    logger.info(
                        f"  → {aligned_filename} "
                        f"({actual_duration:.2f}s, speed={speed_factor:.2f}x)"
                    )
                
                segment.aligned_audio = aligned_path
                segment.speed_factor = speed_factor
            
            # Thống kê
            aligned = sum(1 for s in segments if s.aligned_audio)
            logger.info(
                f"Audio alignment hoàn thành: {aligned}/{len(segments)} segments"
            )
        
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return segments
    
    def _time_stretch(
        self,
        audio: np.ndarray,
        sr: int,
        speed_factor: float,
    ) -> np.ndarray:
        """
        Time-stretch audio mà không thay đổi pitch.
        
        Args:
            audio: Audio data (numpy array)
            sr: Sample rate
            speed_factor: Hệ số tốc độ (>1 = nhanh hơn, <1 = chậm hơn)
            
        Returns:
            Audio đã được stretch
        """
        # Chuyển sang float32 cho librosa
        audio_float = audio.astype(np.float32)
        
        # Time stretch
        stretched = librosa.effects.time_stretch(audio_float, rate=speed_factor)
        
        return stretched
    
    def _match_duration(
        self,
        audio: np.ndarray,
        sr: int,
        target_duration: float,
    ) -> np.ndarray:
        """
        Trim hoặc pad audio để khớp chính xác target duration.
        """
        target_samples = int(target_duration * sr)
        current_samples = len(audio)
        
        if current_samples > target_samples:
            # Trim
            audio = audio[:target_samples]
        elif current_samples < target_samples:
            # Pad with silence
            padding = np.zeros(target_samples - current_samples, dtype=audio.dtype)
            audio = np.concatenate([audio, padding])
        
        return audio
