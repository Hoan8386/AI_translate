"""
Audio Service
===============
Service layer cho audio processing (Stage 2, 4, 8 + merge).
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict

from pipeline.p02_audio_extractor import AudioExtractor
from pipeline.p04_segment_creator import SegmentCreator
from pipeline.p08_audio_alignment import AudioAligner
from models_data.segment import Segment
from models_data.speaker import Speaker
from utils.logger import get_logger

logger = get_logger(__name__)


class AudioService:
    """Service cho tất cả audio operations"""
    
    def __init__(self):
        self.extractor = AudioExtractor()
        self.segment_creator = SegmentCreator()
        self.aligner = AudioAligner()
    
    def extract(self, video_path: str, output_path: str) -> str:
        """Extract audio từ video (Stage 2)"""
        return self.extractor.process(video_path, output_path)
    
    def create_segments(
        self, audio_path: str, diarization: List[Dict], output_dir: str
    ) -> tuple:
        """Tạo segments (Stage 4)"""
        return self.segment_creator.process(audio_path, diarization, output_dir)
    
    def align(self, segments: List[Segment], output_dir: str) -> List[Segment]:
        """Align audio duration (Stage 8)"""
        return self.aligner.process(segments, output_dir)
    
    def merge_segments(
        self,
        segments: List[Segment],
        total_duration: float,
        output_path: str,
        sample_rate: int = 16000,
    ) -> str:
        """
        Merge tất cả aligned segments thành 1 file audio.
        
        Đặt từng segment vào đúng vị trí thời gian gốc.
        
        Args:
            segments: Danh sách segments đã align
            total_duration: Tổng thời lượng audio gốc
            output_path: Đường dẫn output
            sample_rate: Sample rate
            
        Returns:
            Đường dẫn merged audio
        """
        logger.info("Merging all segments into final audio...")
        
        # Tạo audio rỗng với full duration
        total_samples = int(total_duration * sample_rate)
        merged = np.zeros(total_samples, dtype=np.float32)
        
        for segment in segments:
            audio_path = segment.aligned_audio or segment.generated_audio
            
            if not audio_path or not Path(audio_path).exists():
                logger.debug(f"Segment {segment.id}: không có audio, để silence")
                continue
            
            # Đọc segment audio
            seg_audio, seg_sr = sf.read(audio_path)
            
            # Resample nếu cần
            if seg_sr != sample_rate:
                import librosa
                seg_audio = librosa.resample(
                    seg_audio.astype(np.float32), 
                    orig_sr=seg_sr, 
                    target_sr=sample_rate
                )
            
            # Tính vị trí
            start_sample = int(segment.start * sample_rate)
            end_sample = start_sample + len(seg_audio)
            
            # Clamp
            if end_sample > total_samples:
                seg_audio = seg_audio[:total_samples - start_sample]
                end_sample = total_samples
            
            if start_sample < total_samples:
                # Mix vào (thay thế, không cộng)
                merged[start_sample:start_sample + len(seg_audio)] = seg_audio
        
        # Normalize
        max_val = np.max(np.abs(merged))
        if max_val > 0:
            merged = merged / max_val * 0.95
        
        # Lưu
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, merged, sample_rate)
        
        duration = len(merged) / sample_rate
        logger.info(f"Merged audio: {output_path} ({duration:.1f}s)")
        
        return output_path
