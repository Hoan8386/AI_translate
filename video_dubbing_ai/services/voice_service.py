"""
Voice Service
===============
Service layer cho Voice Cloning (Stage 7).
"""

from typing import List, Dict
from pipeline.p07_voice_clone import VoiceCloner
from models_data.segment import Segment
from models_data.speaker import Speaker
from utils.logger import get_logger

logger = get_logger(__name__)


class VoiceService:
    """Service wrapper cho VoiceCloner"""
    
    def __init__(self):
        self.cloner = VoiceCloner()
    
    def clone(
        self,
        segments: List[Segment],
        speakers: Dict[str, Speaker],
        output_dir: str,
    ) -> List[Segment]:
        """
        Voice clone cho tất cả segments.
        
        Args:
            segments: Segments với vi_text
            speakers: Speaker profiles
            output_dir: Thư mục output
            
        Returns:
            Segments đã có generated_audio
        """
        return self.cloner.process(segments, speakers, output_dir)
