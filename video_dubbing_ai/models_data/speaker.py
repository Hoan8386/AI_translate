"""
Speaker Data Model
===================
Đại diện cho một người nói trong video.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Speaker:
    """
    Đại diện cho 1 người nói.
    
    Chứa thông tin về giọng nói gốc (reference) để dùng cho voice cloning.
    """
    
    # ID duy nhất (speaker_1, speaker_2, ...)
    id: str = ""
    
    # Tổng thời lượng nói (giây)
    total_duration: float = 0.0
    
    # Số lượng segments
    segment_count: int = 0
    
    # Đường dẫn reference audio (dùng cho voice cloning)
    reference_audio: str = ""
    
    # Danh sách segment IDs thuộc speaker này
    segment_ids: List[int] = field(default_factory=list)
    
    # Embeddings (optional, cho speaker verification)
    embedding: Optional[List[float]] = None
    
    def add_segment(self, segment_id: int, duration: float):
        """Thêm segment vào speaker profile"""
        self.segment_ids.append(segment_id)
        self.segment_count += 1
        self.total_duration += duration
    
    def __repr__(self):
        return (
            f"Speaker(id='{self.id}', segments={self.segment_count}, "
            f"duration={self.total_duration:.1f}s)"
        )
