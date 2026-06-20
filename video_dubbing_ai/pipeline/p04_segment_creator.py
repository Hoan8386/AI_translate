"""
Stage 4: Segment Creator
==========================
Cắt audio thành các đoạn nhỏ theo kết quả speaker detection.

Input:  audio.wav + speaker diarization results
Output: segment_001.wav, segment_002.wav, segment_003.wav, ...
"""

import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Dict

from models_data.segment import Segment
from models_data.speaker import Speaker
from utils.logger import get_logger, log_stage
from utils.timer import Timer

logger = get_logger(__name__)


class SegmentCreator:
    """Cắt audio thành segments theo speaker detection"""
    
    STAGE_NUM = 4
    STAGE_NAME = "Segment Creator"
    
    def process(
        self,
        audio_path: str,
        diarization: List[Dict],
        output_dir: str,
    ) -> tuple:
        """
        Cắt audio thành segments.
        
        Args:
            audio_path: Đường dẫn audio gốc (16KHz, Mono, WAV)
            diarization: Kết quả speaker detection [{speaker, start, end}, ...]
            output_dir: Thư mục lưu segments
            
        Returns:
            Tuple of (List[Segment], Dict[str, Speaker])
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")
        
        segments = []
        speakers = {}
        
        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            # Đọc audio file
            audio_data, sample_rate = sf.read(audio_path)
            logger.info(
                f"Audio loaded: {len(audio_data)} samples, "
                f"{sample_rate}Hz, {len(audio_data)/sample_rate:.1f}s"
            )
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for idx, entry in enumerate(diarization):
                seg_id = idx + 1
                speaker_id = entry["speaker"]
                start_time = entry["start"]
                end_time = entry["end"]
                
                # Tính vị trí samples
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                # Clamp
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)
                
                if end_sample <= start_sample:
                    logger.warning(f"Segment {seg_id} rỗng, bỏ qua")
                    continue
                
                # Cắt audio
                segment_audio = audio_data[start_sample:end_sample]
                
                # Lưu segment
                segment_filename = f"segment_{seg_id:03d}.wav"
                segment_path = output_path / segment_filename
                sf.write(str(segment_path), segment_audio, sample_rate)
                
                # Tạo Segment object
                segment = Segment(
                    id=seg_id,
                    speaker=speaker_id,
                    start=start_time,
                    end=end_time,
                    source_audio=str(segment_path),
                )
                segments.append(segment)
                
                # Cập nhật speaker profile
                if speaker_id not in speakers:
                    speakers[speaker_id] = Speaker(id=speaker_id)
                speakers[speaker_id].add_segment(seg_id, end_time - start_time)
                
                logger.debug(
                    f"  Segment {seg_id:03d}: {speaker_id} "
                    f"({start_time:.1f}s - {end_time:.1f}s) "
                    f"= {end_time-start_time:.1f}s"
                )
            
            # Tạo reference audio cho mỗi speaker
            self._create_reference_audio(
                audio_data, sample_rate, segments, speakers, output_path
            )
            
            logger.info(
                f"Created {len(segments)} segments, "
                f"{len(speakers)} speakers"
            )
        
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return segments, speakers
    
    def _create_reference_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        segments: List[Segment],
        speakers: Dict[str, Speaker],
        output_dir: Path,
    ):
        """
        Tạo reference audio cho mỗi speaker (dùng cho voice cloning).
        
        Chọn segment dài nhất (3-15s) của mỗi speaker làm reference.
        Nếu không đủ dài, ghép nhiều segments lại.
        """
        MIN_REF_DURATION = 3.0   # Tối thiểu 3s
        MAX_REF_DURATION = 15.0  # Tối đa 15s
        
        ref_dir = output_dir.parent / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        
        for speaker_id, speaker in speakers.items():
            # Lấy segments của speaker này
            spk_segments = [s for s in segments if s.speaker == speaker_id]
            
            # Sort theo duration (dài nhất trước)
            spk_segments.sort(key=lambda s: s.duration, reverse=True)
            
            # Thu thập audio cho reference
            ref_audio_parts = []
            total_ref_duration = 0
            
            for seg in spk_segments:
                if total_ref_duration >= MAX_REF_DURATION:
                    break
                
                start_sample = int(seg.start * sample_rate)
                end_sample = int(seg.end * sample_rate)
                
                # Giới hạn phần thêm vào
                remaining = MAX_REF_DURATION - total_ref_duration
                max_samples = int(remaining * sample_rate)
                segment_audio = audio_data[start_sample:end_sample][:max_samples]
                
                ref_audio_parts.append(segment_audio)
                total_ref_duration += len(segment_audio) / sample_rate
            
            if total_ref_duration < MIN_REF_DURATION:
                logger.warning(
                    f"Speaker {speaker_id}: reference chỉ {total_ref_duration:.1f}s "
                    f"(tối thiểu {MIN_REF_DURATION}s). Chất lượng voice clone có thể kém."
                )
            
            # Ghép tất cả thành 1 file reference
            ref_audio = np.concatenate(ref_audio_parts)
            ref_filename = f"{speaker_id}_reference.wav"
            ref_path = ref_dir / ref_filename
            sf.write(str(ref_path), ref_audio, sample_rate)
            
            # Cập nhật speaker profile
            speaker.reference_audio = str(ref_path)
            
            # Cập nhật reference path trong segments
            for seg in segments:
                if seg.speaker == speaker_id:
                    seg.reference_audio = str(ref_path)
            
            logger.info(
                f"Reference audio: {speaker_id} → {ref_filename} "
                f"({total_ref_duration:.1f}s)"
            )
