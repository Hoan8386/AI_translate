"""
Segment Data Model
===================
Data object trung tâm của toàn bộ hệ thống.
Mọi module đều đọc và ghi object này.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json


@dataclass
class Segment:
    """
    Đại diện cho 1 đoạn audio/video trong pipeline.
    
    Đây là data object trung tâm - mọi module đều đọc và ghi object này.
    
    Ví dụ:
        Segment(
            id=1,
            speaker="speaker_1",
            start=0.0,
            end=5.0,
            source_audio="segment_001.wav",
            reference_audio="speaker_1_reference.wav",
            zh_text="大家好",
            vi_text="Xin chào mọi người",
            generated_audio="speaker_1_vi.wav"
        )
    """
    
    # --- Identification ---
    id: int = 0
    speaker: str = ""
    
    # --- Timing ---
    start: float = 0.0
    end: float = 0.0
    
    # --- Audio paths ---
    source_audio: str = ""          # Audio gốc (segment_001.wav)
    reference_audio: str = ""       # Reference cho voice clone
    generated_audio: str = ""       # Audio tiếng Việt sinh ra
    aligned_audio: str = ""         # Audio sau khi align thời lượng
    
    # --- Text ---
    zh_text: str = ""               # Text tiếng Trung (ASR output)
    vi_text: str = ""               # Text tiếng Việt (Translation output)
    
    # --- Metadata ---
    confidence: float = 0.0         # ASR confidence
    speed_factor: float = 1.0       # Hệ số tốc độ (audio alignment)
    
    @property
    def duration(self) -> float:
        """Thời lượng segment (giây)"""
        return self.end - self.start
    
    @property
    def duration_ms(self) -> int:
        """Thời lượng segment (milliseconds)"""
        return int((self.end - self.start) * 1000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển sang dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Chuyển sang JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Segment':
        """Tạo Segment từ dictionary"""
        # Chỉ lấy fields hợp lệ
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Segment':
        """Tạo Segment từ JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __repr__(self):
        return (
            f"Segment(id={self.id}, speaker='{self.speaker}', "
            f"time={self.start:.1f}-{self.end:.1f}s, "
            f"zh='{self.zh_text[:20]}...', vi='{self.vi_text[:20]}...')"
        )
