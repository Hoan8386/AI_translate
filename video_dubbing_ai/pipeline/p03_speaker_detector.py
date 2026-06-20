"""
Stage 3: Speaker Detector
===========================
Nhận diện và phân biệt nhiều người nói trong audio.

Input:  audio.wav
Output: List of {speaker, start, end}

Công nghệ: pyannote.audio
"""

import torch
from pathlib import Path
from typing import List, Dict

from utils.logger import get_logger, log_stage
from utils.timer import Timer
from utils.gpu_manager import GPUManager
from config.settings import get_settings

logger = get_logger(__name__)


class SpeakerDetector:
    """
    Phát hiện và phân biệt người nói bằng pyannote.audio.
    
    Output ví dụ:
    [
        {"speaker": "speaker_1", "start": 0.0, "end": 5.0},
        {"speaker": "speaker_2", "start": 5.0, "end": 11.0},
        {"speaker": "speaker_1", "start": 11.0, "end": 18.0}
    ]
    """
    
    STAGE_NUM = 3
    STAGE_NAME = "Speaker Detector"
    
    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()
        self._pipeline = None
    
    def _load_model(self):
        """Load pyannote pipeline lên GPU"""
        from pyannote.audio import Pipeline
        
        hf_token = self.settings.speaker.hf_token
        if not hf_token:
            raise ValueError(
                "Cần Hugging Face token để dùng pyannote.audio!\n"
                "1. Đăng ký tại https://huggingface.co\n"
                "2. Accept license tại https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "3. Set HF_TOKEN trong file .env"
            )
        
        logger.info(f"Loading pyannote model: {self.settings.speaker.model_name}")
        
        pipeline = Pipeline.from_pretrained(
            self.settings.speaker.model_name,
            use_auth_token=hf_token,
        )
        
        # Load lên GPU
        if self.gpu.device == "cuda":
            pipeline = pipeline.to(torch.device("cuda"))
        
        self._pipeline = pipeline
        logger.info("pyannote model loaded")
    
    def _unload_model(self):
        """Unload pyannote model"""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("pyannote model unloaded")
    
    def process(self, audio_path: str) -> List[Dict]:
        """
        Phát hiện speakers trong audio.
        
        Args:
            audio_path: Đường dẫn audio file (16KHz, Mono, WAV)
            
        Returns:
            List of diarization results: [{speaker, start, end}, ...]
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")
        
        results = []
        
        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            audio_file = Path(audio_path)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio không tồn tại: {audio_path}")
            
            try:
                # Load model
                self.gpu.ensure_free(2000)
                self._load_model()
                
                # Run diarization
                logger.info("Running speaker diarization...")
                
                diarization_params = {}
                if self.settings.speaker.max_speakers:
                    diarization_params["max_speakers"] = self.settings.speaker.max_speakers
                
                diarization = self._pipeline(
                    str(audio_path),
                    **diarization_params
                )
                
                # Parse results
                speaker_map = {}  # Đổi tên speaker cho đẹp
                speaker_counter = 0
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    # Map speaker label sang speaker_1, speaker_2, ...
                    if speaker not in speaker_map:
                        speaker_counter += 1
                        speaker_map[speaker] = f"speaker_{speaker_counter}"
                    
                    duration = turn.end - turn.start
                    
                    # Bỏ qua segments quá ngắn
                    if duration < self.settings.speaker.min_segment_duration:
                        logger.debug(
                            f"Bỏ qua segment ngắn: {speaker} "
                            f"({turn.start:.1f}-{turn.end:.1f}s = {duration:.2f}s)"
                        )
                        continue
                    
                    results.append({
                        "speaker": speaker_map[speaker],
                        "start": round(turn.start, 3),
                        "end": round(turn.end, 3),
                    })
                
                # Merge segments liền kề cùng speaker
                results = self._merge_adjacent(results)
                
                # Log kết quả
                unique_speakers = set(r["speaker"] for r in results)
                logger.info(
                    f"Phát hiện {len(unique_speakers)} speakers, "
                    f"{len(results)} segments"
                )
                for r in results:
                    logger.debug(
                        f"  {r['speaker']}: {r['start']:.1f}s - {r['end']:.1f}s "
                        f"({r['end']-r['start']:.1f}s)"
                    )
                
            finally:
                # LUÔN unload model dù có lỗi hay không
                self._unload_model()
        
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return results
    
    def _merge_adjacent(self, segments: List[Dict], gap_threshold: float = 0.3) -> List[Dict]:
        """
        Merge các segments liền kề cùng speaker.
        
        Nếu 2 segments cùng speaker cách nhau < gap_threshold giây, merge lại.
        """
        if not segments:
            return segments
        
        merged = [segments[0].copy()]
        
        for seg in segments[1:]:
            last = merged[-1]
            gap = seg["start"] - last["end"]
            
            if seg["speaker"] == last["speaker"] and gap < gap_threshold:
                # Merge
                last["end"] = seg["end"]
            else:
                merged.append(seg.copy())
        
        if len(merged) < len(segments):
            logger.info(
                f"Merged {len(segments)} → {len(merged)} segments "
                f"(gap < {gap_threshold}s)"
            )
        
        return merged
