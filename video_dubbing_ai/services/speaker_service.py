"""
Speaker Service
=================
Service layer cho speaker detection (Stage 3).
"""

from typing import List, Dict
from pipeline.p03_speaker_detector import SpeakerDetector
from utils.logger import get_logger

logger = get_logger(__name__)


class SpeakerService:
    """Service wrapper cho SpeakerDetector"""
    
    def __init__(self):
        self.detector = SpeakerDetector()
    
    def detect(self, audio_path: str) -> List[Dict]:
        """
        Phát hiện speakers trong audio.
        
        Args:
            audio_path: Audio file (16KHz, Mono, WAV)
            
        Returns:
            List of {speaker, start, end}
        """
        return self.detector.process(audio_path)
