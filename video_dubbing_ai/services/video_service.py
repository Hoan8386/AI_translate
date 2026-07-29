"""
Video Service
===============
Service layer cho video processing (Stage 1).
"""

from pipeline.p01_video_processor import VideoProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


class VideoService:
    """Service wrapper cho VideoProcessor"""
    
    def __init__(self):
        self.processor = VideoProcessor()
    
    def normalize(self, input_path: str, output_path: str) -> str:
        """
        Chuẩn hóa video: H264, 30FPS, AAC.
        
        Args:
            input_path: Video input
            output_path: Video output (normalized)
            
        Returns:
            Đường dẫn video chuẩn hóa
        """
        return self.processor.process(input_path, output_path)
    
    def get_info(self, video_path: str) -> dict:
        """Lấy thông tin video"""
        return self.processor.get_video_info(video_path)
