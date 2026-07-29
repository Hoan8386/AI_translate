"""
Renderer Service
==================
Service layer cho Video Renderer (Stage 10).
"""

from typing import Optional
from pipeline.p10_renderer import VideoRenderer
from utils.logger import get_logger

logger = get_logger(__name__)


class RendererService:
    """Service wrapper cho VideoRenderer"""
    
    def __init__(self):
        self.renderer = VideoRenderer()
    
    def render(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        original_video: Optional[str] = None,
    ) -> str:
        """
        Render video cuối cùng.
        
        Args:
            video_path: Video (đã lip sync)
            audio_path: Audio tiếng Việt
            output_path: Output path
            
        Returns:
            Đường dẫn output video
        """
        return self.renderer.process(video_path, audio_path, output_path, original_video)
