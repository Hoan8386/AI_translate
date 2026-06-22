"""
Configuration & Settings
=========================
Cấu hình trung tâm cho toàn bộ hệ thống dubbing.
Tối ưu cho RTX 5060 8GB.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Load .env nếu có
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class VideoSettings:
    """Chuẩn video output"""
    codec: str = "libx264"
    fps: int = 30
    audio_codec: str = "aac"
    preset: str = "medium"
    crf: int = 23  # Chất lượng (0-51, thấp = chất lượng cao)


@dataclass
class AudioSettings:
    """Chuẩn audio processing"""
    sample_rate: int = 16000
    channels: int = 1  # Mono
    format: str = "wav"
    bit_depth: int = 16


@dataclass
class GPUSettings:
    """Cấu hình GPU - Tối ưu RTX 5060 8GB"""
    # Nguyên tắc vàng: chỉ 1 model trên GPU tại 1 thời điểm
    max_models_on_gpu: int = 1
    max_vram_mb: int = 8192  # 8GB
    # Ngưỡng cảnh báo VRAM (MB)
    vram_warning_threshold: int = 7000
    # Device
    device: str = "cuda"
    # Batch size nhỏ để tiết kiệm VRAM
    batch_size: int = 1


@dataclass
class TranslationSettings:
    """Cấu hình dịch thuật"""
    # Provider: "gemini" hoặc "openai"
    provider: str = os.getenv("TRANSLATION_PROVIDER", "gemini")
    # API Keys
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    # Model settings
    gemini_model: str = "gemini-2.0-flash"
    openai_model: str = "gpt-4o-mini"
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    # Rate limiting
    requests_per_minute: int = 30


@dataclass
class SpeakerSettings:
    """Cấu hình speaker detection"""
    # Hugging Face token cho pyannote
    hf_token: str = os.getenv("HF_TOKEN", "")
    # pyannote model
    model_name: str = "pyannote/speaker-diarization-3.1"
    # Số speaker tối đa (None = auto detect)
    max_speakers: Optional[int] = None
    # Thời lượng tối thiểu segment (giây)
    min_segment_duration: float = 0.5


@dataclass
class ASRSettings:
    """Cấu hình ASR (SenseVoice)"""
    model_name: str = "iic/SenseVoiceSmall"
    language: str = "zh"
    # Batch size cho ASR
    batch_size: int = 1


@dataclass
class VoiceCloneSettings:
    """Cấu hình Voice Cloning (Fish Speech)"""
    # Fish Speech API server URL
    server_url: str = os.getenv("FISH_SPEECH_URL", "http://127.0.0.1:8080")
    # Timeout cho mỗi request (giây)
    request_timeout: int = 120
    # Thời lượng tối thiểu reference audio (giây)
    min_reference_duration: float = 3.0
    # Thời lượng tối đa reference audio (giây)
    max_reference_duration: float = 30.0
    # Target language
    target_language: str = "VI"


@dataclass
class LipSyncSettings:
    """Cấu hình Lip Sync (Wav2Lip)"""
    # Model checkpoint
    checkpoint_path: str = ""  # Sẽ được set tự động
    # Wav2Lip settings
    face_det_batch_size: int = 1
    wav2lip_batch_size: int = 1
    # Resize factor để giảm VRAM
    resize_factor: int = 1
    # Padding
    pads: list = field(default_factory=lambda: [0, 10, 0, 0])


@dataclass
class Settings:
    """Cấu hình trung tâm"""

    # === Đường dẫn ===
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # Sub-directories
    @property
    def input_dir(self) -> Path:
        return self.project_root / "input"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output"

    @property
    def temp_dir(self) -> Path:
        return self.project_root / "temp"

    @property
    def cache_dir(self) -> Path:
        return self.project_root / "cache"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def third_party_dir(self) -> Path:
        return self.project_root / "third_party"

    # === Component settings ===
    video: VideoSettings = field(default_factory=VideoSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    gpu: GPUSettings = field(default_factory=GPUSettings)
    translation: TranslationSettings = field(default_factory=TranslationSettings)
    speaker: SpeakerSettings = field(default_factory=SpeakerSettings)
    asr: ASRSettings = field(default_factory=ASRSettings)
    voice_clone: VoiceCloneSettings = field(default_factory=VoiceCloneSettings)
    lipsync: LipSyncSettings = field(default_factory=LipSyncSettings)

    # === Logging ===
    log_level: str = "INFO"
    log_to_file: bool = True

    def __post_init__(self):
        """Tạo các thư mục cần thiết"""
        for dir_path in [self.input_dir, self.output_dir, self.temp_dir,
                         self.cache_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Set Wav2Lip checkpoint path
        if not self.lipsync.checkpoint_path:
            self.lipsync.checkpoint_path = str(
                self.models_dir / "wav2lip" / "wav2lip_gan.pth"
            )


# Singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Lấy cấu hình singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
