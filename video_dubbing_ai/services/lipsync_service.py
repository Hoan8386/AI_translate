"""
Lip Sync Service
==================
Service layer cho Lip Sync (Stage 9).
"""

from pipeline.p09_lipsync import LipSyncer
from utils.logger import get_logger

logger = get_logger(__name__)


class LipSyncService:
    """Service wrapper cho LipSyncer"""
    
    def __init__(self):
        self.syncer = LipSyncer()
    
    def sync(self, video_path: str, audio_path: str, output_path: str) -> str:
        """
        Đồng bộ khẩu hình.
        
        Args:
            video_path: Video gốc
            audio_path: Audio tiếng Việt
            output_path: Output path
            
        Returns:
            Đường dẫn video đã lip sync
        """
        try:
            return self.syncer.process(video_path, audio_path, output_path)
        except Exception as e:
            logger.warning(f"Lip sync thất bại: {e}")
            logger.info("Fallback: thay audio không lip sync")
            return self.syncer.process_simple(video_path, audio_path, output_path)
    
    def simple_replace(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Thay audio đơn giản (không lip sync)"""
        return self.syncer.process_simple(video_path, audio_path, output_path)
