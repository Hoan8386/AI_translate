"""
Job Data Model
================
Tracking trạng thái của mỗi lần chạy pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

from models_data.segment import Segment
from models_data.speaker import Speaker


class JobStatus(Enum):
    """Trạng thái job"""
    PENDING = "pending"
    PROCESSING = "processing"
    STAGE_1_VIDEO = "stage_1_video_processor"
    STAGE_2_AUDIO = "stage_2_audio_extractor"
    STAGE_3_SPEAKER = "stage_3_speaker_detector"
    STAGE_4_SEGMENT = "stage_4_segment_creator"
    STAGE_5_ASR = "stage_5_asr"
    STAGE_6_TRANSLATE = "stage_6_translation"
    STAGE_7_VOICE = "stage_7_voice_clone"
    STAGE_8_ALIGN = "stage_8_audio_alignment"
    STAGE_9_LIPSYNC = "stage_9_lipsync"
    STAGE_10_RENDER = "stage_10_renderer"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """
    Đại diện cho 1 lần chạy pipeline.
    
    Chứa toàn bộ thông tin và trạng thái của quá trình dubbing.
    """
    
    # --- Identification ---
    id: str = ""
    
    # --- Input/Output ---
    input_video: str = ""
    output_video: str = ""
    
    # --- Status ---
    status: JobStatus = JobStatus.PENDING
    current_stage: int = 0
    progress: float = 0.0  # 0-100
    error_message: str = ""
    
    # --- Timestamps ---
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str = ""
    completed_at: str = ""
    
    # --- Paths (set during processing) ---
    normalized_video: str = ""
    extracted_audio: str = ""
    merged_audio: str = ""
    lipsync_video: str = ""
    
    # --- Data ---
    segments: List[Segment] = field(default_factory=list)
    speakers: Dict[str, Speaker] = field(default_factory=dict)
    
    # --- Timing ---
    stage_times: Dict[str, float] = field(default_factory=dict)
    
    def update_status(self, status: JobStatus, stage: int = 0):
        """Cập nhật trạng thái job"""
        self.status = status
        if stage > 0:
            self.current_stage = stage
            self.progress = (stage / 10) * 100
        
        if status == JobStatus.PROCESSING and not self.started_at:
            self.started_at = datetime.now().isoformat()
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
            self.completed_at = datetime.now().isoformat()
    
    def add_segment(self, segment: Segment):
        """Thêm segment vào job"""
        self.segments.append(segment)
    
    def add_speaker(self, speaker: Speaker):
        """Thêm speaker vào job"""
        self.speakers[speaker.id] = speaker
    
    def get_segments_by_speaker(self, speaker_id: str) -> List[Segment]:
        """Lấy tất cả segments của 1 speaker"""
        return [s for s in self.segments if s.speaker == speaker_id]
    
    def save(self, path: Path):
        """Lưu job state ra file JSON"""
        data = {
            "id": self.id,
            "input_video": self.input_video,
            "output_video": self.output_video,
            "status": self.status.value,
            "current_stage": self.current_stage,
            "progress": self.progress,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "normalized_video": self.normalized_video,
            "extracted_audio": self.extracted_audio,
            "merged_audio": self.merged_audio,
            "lipsync_video": self.lipsync_video,
            "segments": [s.to_dict() for s in self.segments],
            "speakers": {
                k: {
                    "id": v.id,
                    "total_duration": v.total_duration,
                    "segment_count": v.segment_count,
                    "reference_audio": v.reference_audio,
                    "segment_ids": v.segment_ids,
                }
                for k, v in self.speakers.items()
            },
            "stage_times": self.stage_times,
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'Job':
        """Load job state từ file JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        job = cls(
            id=data["id"],
            input_video=data["input_video"],
            output_video=data.get("output_video", ""),
            status=JobStatus(data["status"]),
            current_stage=data.get("current_stage", 0),
            progress=data.get("progress", 0.0),
            error_message=data.get("error_message", ""),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            normalized_video=data.get("normalized_video", ""),
            extracted_audio=data.get("extracted_audio", ""),
            merged_audio=data.get("merged_audio", ""),
            lipsync_video=data.get("lipsync_video", ""),
        )
        
        # Load segments
        for seg_data in data.get("segments", []):
            job.segments.append(Segment.from_dict(seg_data))
        
        # Load speakers
        for spk_id, spk_data in data.get("speakers", {}).items():
            speaker = Speaker(
                id=spk_data["id"],
                total_duration=spk_data.get("total_duration", 0),
                segment_count=spk_data.get("segment_count", 0),
                reference_audio=spk_data.get("reference_audio", ""),
                segment_ids=spk_data.get("segment_ids", []),
            )
            job.speakers[spk_id] = speaker
        
        job.stage_times = data.get("stage_times", {})
        return job
    
    def __repr__(self):
        return (
            f"Job(id='{self.id}', status={self.status.value}, "
            f"stage={self.current_stage}/10, "
            f"segments={len(self.segments)}, "
            f"speakers={len(self.speakers)})"
        )
