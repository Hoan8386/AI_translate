"""
Stage 3: Speaker Detector
=========================
Nhận diện và phân biệt nhiều người nói trong audio.

Input:
    audio.wav

Output:
    [
        {"speaker": "speaker_1", "start": 0.0, "end": 5.0},
        {"speaker": "speaker_2", "start": 5.0, "end": 11.0}
    ]

Technology:
    pyannote.audio
"""

from pathlib import Path
from typing import List, Dict

import torch

from utils.logger import get_logger, log_stage
from utils.timer import Timer
from utils.gpu_manager import GPUManager
from config.settings import get_settings

logger = get_logger(__name__)


class SpeakerDetector:

    STAGE_NUM = 3
    STAGE_NAME = "Speaker Detector"

    # Shared giữa tất cả instance
    _shared_pipeline = None

    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()

        # dùng pipeline đã load nếu có
        self._pipeline = SpeakerDetector._shared_pipeline

    def _load_model(self):
        """
        Load pyannote pipeline (chỉ load 1 lần)
        """

        if self._pipeline is not None:
            return

        from pyannote.audio import Pipeline

        hf_token = self.settings.speaker.hf_token

        if not hf_token:
            raise ValueError(
                "HF_TOKEN chưa được thiết lập.\n"
                "1. Đăng ký HuggingFace\n"
                "2. Accept license của pyannote/speaker-diarization-3.1\n"
                "3. Set HF_TOKEN trong file .env"
            )

        logger.info(
            f"Loading pyannote model: {self.settings.speaker.model_name}"
        )

        pipeline = Pipeline.from_pretrained(
            self.settings.speaker.model_name,
            use_auth_token=hf_token,
        )

        if pipeline is None:
            raise RuntimeError(
                "Không load được pyannote pipeline."
            )

        # chuyển lên GPU
        if self.gpu.device == "cuda":
            logger.info("Moving pyannote model to CUDA...")
            pipeline.to(torch.device("cuda"))

        logger.info(f"Pipeline type: {type(pipeline)}")

        self._pipeline = pipeline
        SpeakerDetector._shared_pipeline = pipeline

        logger.info("pyannote model loaded")

    def process(self, audio_path: str) -> List[Dict]:
        """
        Speaker diarization

        Args:
            audio_path: file wav

        Returns:
            [
                {
                    "speaker": "speaker_1",
                    "start": 0.0,
                    "end": 3.2
                }
            ]
        """

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")

        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):

            audio_file = Path(audio_path)

            if not audio_file.exists():
                raise FileNotFoundError(
                    f"Audio không tồn tại: {audio_path}"
                )

            self.gpu.ensure_free(2000)

            self._load_model()

            logger.info("Running speaker diarization...")

            diarization_params = {}

            if self.settings.speaker.max_speakers:
                diarization_params["max_speakers"] = (
                    self.settings.speaker.max_speakers
                )

            diarization = self._pipeline(
                str(audio_file),
                **diarization_params
            )

            results = []

            speaker_map = {}
            speaker_counter = 0

            for turn, _, speaker in diarization.itertracks(
                    yield_label=True):

                if speaker not in speaker_map:
                    speaker_counter += 1
                    speaker_map[speaker] = (
                        f"speaker_{speaker_counter}"
                    )

                duration = turn.end - turn.start

                if duration < self.settings.speaker.min_segment_duration:
                    continue

                results.append(
                    {
                        "speaker": speaker_map[speaker],
                        "start": round(turn.start, 3),
                        "end": round(turn.end, 3),
                    }
                )

            results = self._merge_adjacent(results)

            unique_speakers = {
                r["speaker"]
                for r in results
            }

            logger.info(
                f"Detected {len(unique_speakers)} speakers "
                f"with {len(results)} segments"
            )

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")

        return results

    def _merge_adjacent(
            self,
            segments: List[Dict],
            gap_threshold: float = 0.3
    ) -> List[Dict]:

        if not segments:
            return segments

        merged = [segments[0].copy()]

        for seg in segments[1:]:

            last = merged[-1]

            gap = seg["start"] - last["end"]

            if (
                    seg["speaker"] == last["speaker"]
                    and gap < gap_threshold
            ):
                last["end"] = seg["end"]

            else:
                merged.append(seg.copy())

        if len(merged) < len(segments):
            logger.info(
                f"Merged {len(segments)} → {len(merged)} segments"
            )

        return merged

    @classmethod
    def unload_model(cls):
        """
        Giải phóng model khi shutdown server
        """

        if cls._shared_pipeline is not None:
            logger.info("Unloading pyannote model...")

            del cls._shared_pipeline
            cls._shared_pipeline = None

            import gc

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("pyannote model unloaded")