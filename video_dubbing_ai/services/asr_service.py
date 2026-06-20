"""
ASR Service
=============
Service layer cho Chinese ASR (Stage 5).
"""

from typing import List
from pipeline.p05_asr import ChineseASR
from models_data.segment import Segment
from utils.logger import get_logger

logger = get_logger(__name__)


class ASRService:
    """Service wrapper cho ChineseASR"""
    
    def __init__(self):
        self.asr = ChineseASR()
    
    def transcribe(self, segments: List[Segment]) -> List[Segment]:
        """
        Nhận dạng giọng nói tiếng Trung.
        
        Args:
            segments: Danh sách segments với source_audio
            
        Returns:
            Segments đã có zh_text
        """
        return self.asr.process(segments)
