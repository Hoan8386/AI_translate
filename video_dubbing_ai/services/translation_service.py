"""
Translation Service
=====================
Service layer cho translation (Stage 6).
"""

from typing import List
from pipeline.p06_translation import Translator
from models_data.segment import Segment
from utils.logger import get_logger

logger = get_logger(__name__)


class TranslationService:
    """Service wrapper cho Translator"""
    
    def __init__(self):
        self.translator = Translator()
    
    def translate(self, segments: List[Segment]) -> List[Segment]:
        """
        Dịch ZH → VI.
        
        Args:
            segments: Danh sách segments với zh_text
            
        Returns:
            Segments đã có vi_text
        """
        return self.translator.process(segments)
