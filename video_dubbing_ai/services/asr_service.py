"""
ASR Service
=============
Service layer cho Chinese ASR (Stage 5).
"""

from typing import List
from pipeline.p05_asr import ASRProcessor
from models_data.segment import Segment
from utils.logger import get_logger

logger = get_logger(__name__)


class ASRService:
    """Service wrapper cho ASRProcessor"""
    
    def __init__(self):
        self.asr = ASRProcessor()
    
    def transcribe(self, audio_path: str, segments: List[Segment]) -> List[Segment]:
        """
        Nhận dạng giọng nói tiếng Trung.
        
        Args:
            audio_path: Đường dẫn file audio WAV
            segments: Danh sách segments với source_audio
            
        Returns:
            Segments đã có zh_text
        """
        return self.asr.process(audio_path, segments)
