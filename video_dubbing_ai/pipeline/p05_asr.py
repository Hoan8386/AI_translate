"""
Stage 5: Chinese ASR
======================
Nhận dạng giọng nói tiếng Trung (Speech-to-Text).

Input:  segment_001.wav
Output: {speaker: "speaker_1", zh: "大家好"}

Model: SenseVoice Small (FunASR)
"""

import torch
from pathlib import Path
from typing import List

from models_data.segment import Segment
from utils.logger import get_logger, log_stage
from utils.timer import Timer
from utils.gpu_manager import GPUManager
from config.settings import get_settings

logger = get_logger(__name__)


class ChineseASR:
    """
    Nhận dạng giọng nói tiếng Trung với SenseVoice Small.
    
    Sử dụng FunASR framework.
    GPU load → process tất cả segments → unload.
    """
    
    STAGE_NUM = 5
    STAGE_NAME = "Chinese ASR (SenseVoice)"
    
    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()
        self._model = None
    
    def _load_model(self):
        """Load SenseVoice model lên GPU"""
        from funasr import AutoModel
        
        logger.info(f"Loading ASR model: {self.settings.asr.model_name}")
        
        model = AutoModel(
            model=self.settings.asr.model_name,
            trust_remote_code=True,
            device=self.gpu.device,
        )
        
        self._model = model
        logger.info("SenseVoice model loaded")
    
    def _unload_model(self):
        """Unload SenseVoice model"""
        if self._model is not None:
            del self._model
            self._model = None
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("SenseVoice model unloaded")
    
    def process(self, segments: List[Segment]) -> List[Segment]:
        """
        Chạy ASR cho tất cả segments.
        
        Args:
            segments: Danh sách Segment đã có source_audio
            
        Returns:
            Danh sách Segment đã có zh_text
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")
        
        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            try:
                # Load model 1 lần
                self.gpu.ensure_free(2000)
                self._load_model()
                
                # Process từng segment
                for i, segment in enumerate(segments):
                    audio_path = segment.source_audio
                    
                    if not Path(audio_path).exists():
                        logger.warning(f"Segment {segment.id} audio không tồn tại: {audio_path}")
                        continue
                    
                    logger.info(
                        f"ASR [{i+1}/{len(segments)}]: "
                        f"Segment {segment.id} ({segment.speaker})"
                    )
                    
                    # Run ASR
                    try:
                        result = self._model.generate(
                            input=audio_path,
                            language=self.settings.asr.language,
                            batch_size=self.settings.asr.batch_size,
                        )
                        
                        # Parse result
                        if result and len(result) > 0:
                            text = result[0].get("text", "").strip()
                            
                            # SenseVoice có thể trả về tag, clean up
                            text = self._clean_text(text)
                            
                            segment.zh_text = text
                            segment.confidence = result[0].get("confidence", 0.0)
                            
                            logger.info(
                                f"  → \"{text}\" "
                                f"(confidence: {segment.confidence:.2f})"
                            )
                        else:
                            logger.warning(f"  → Không nhận dạng được text")
                            segment.zh_text = ""
                    
                    except Exception as e:
                        logger.error(f"  ASR error cho segment {segment.id}: {e}")
                        segment.zh_text = ""
                
                # Thống kê
                recognized = sum(1 for s in segments if s.zh_text)
                logger.info(
                    f"ASR hoàn thành: {recognized}/{len(segments)} segments nhận dạng"
                )
                
            finally:
                # LUÔN unload model
                self._unload_model()
        
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return segments
    
    def _clean_text(self, text: str) -> str:
        """
        Loại bỏ các tag/token đặc biệt từ SenseVoice output.
        
        SenseVoice có thể trả về dạng: <|zh|><|NEUTRAL|><|Speech|>大家好
        """
        import re
        # Xóa các tag dạng <|...|>
        text = re.sub(r'<\|[^|]*\|>', '', text)
        # Xóa khoảng trắng thừa
        text = text.strip()
        return text
