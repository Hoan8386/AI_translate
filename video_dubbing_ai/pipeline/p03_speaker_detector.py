"""
Stage 3: Speaker Detector
Optimized for RTX 5060 + Torch 2.8 + pyannote.audio 3.3.1
"""

from pathlib import Path
from typing import List, Dict
import json
import gc

import torch
import torchaudio

from utils.logger import get_logger, log_stage
from utils.timer import Timer
from utils.gpu_manager import GPUManager
from config.settings import get_settings

logger = get_logger(__name__)


class SpeakerDetector:

    STAGE_NUM = 3
    STAGE_NAME = "Speaker Detector"

    _shared_pipeline = None

    def __init__(self):

        self.gpu = GPUManager()

        self.settings = get_settings()

        self._pipeline = SpeakerDetector._shared_pipeline

    def _load_model(self):

        if self._pipeline is not None:
            return

        # ===== FIX CHO TORCH >=2.6 =====

        from torch.torch_version import TorchVersion

        torch.serialization.add_safe_globals(
            [
                TorchVersion,
            ]
        )

        # ==============================

        from pyannote.audio import Pipeline

        hf_token = self.settings.speaker.hf_token

        if not hf_token:

            raise ValueError(
                "HF_TOKEN chưa được cấu hình.\n"
                "Accept license của pyannote trước."
            )

        logger.info(
            f"Loading {self.settings.speaker.model_name}"
        )

        pipeline = Pipeline.from_pretrained(
            self.settings.speaker.model_name,
            use_auth_token=hf_token,
        )

        if pipeline is None:

            raise RuntimeError(
                "Không thể load pyannote pipeline."
            )

        if self.gpu.device == "cuda":

            logger.info(
                "Moving pyannote model to CUDA..."
            )

            pipeline.to(
                torch.device("cuda")
            )

        self._pipeline = pipeline

        SpeakerDetector._shared_pipeline = pipeline

        logger.info(
            "pyannote loaded."
        )

    def process(
        self,
        audio_path: str,
    ) -> List[Dict]:

        log_stage(
            self.STAGE_NUM,
            self.STAGE_NAME,
            "START",
        )

        with Timer(
            f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"
        ):

            audio_file = Path(audio_path)

            if not audio_file.exists():

                raise FileNotFoundError(
                    audio_path
                )

            self.gpu.ensure_free(
                2000
            )

            self._load_model()

            try:

                waveform, sample_rate = torchaudio.load(
                    str(audio_file)
                )

                if waveform.shape[0] > 1:

                    waveform = waveform.mean(
                        dim=0,
                        keepdim=True,
                    )

                audio_input = {
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                }

            except Exception:

                audio_input = str(
                    audio_file
                )

            params = {}

            if getattr(
                self.settings.speaker,
                "min_speakers",
                None,
            ):

                params[
                    "min_speakers"
                ] = (
                    self.settings
                    .speaker
                    .min_speakers
                )

            if getattr(
                self.settings.speaker,
                "max_speakers",
                None,
            ):

                params[
                    "max_speakers"
                ] = (
                    self.settings
                    .speaker
                    .max_speakers
                )

            if getattr(
                self.settings.speaker,
                "num_speakers",
                None,
            ):

                params[
                    "num_speakers"
                ] = (
                    self.settings
                    .speaker
                    .num_speakers
                )

            logger.info(
                "Running diarization..."
            )

            diarization = self._pipeline(
                audio_input,
                **params,
            )

            results = []

            speaker_map = {}

            counter = 0

            min_duration = getattr(
                self.settings.speaker,
                "min_segment_duration",
                0.2,
            )

            for turn, _, speaker in diarization.itertracks(
                yield_label=True
            ):

                if speaker not in speaker_map:

                    counter += 1

                    speaker_map[
                        speaker
                    ] = (
                        f"speaker_{counter}"
                    )

                duration = (
                    turn.end
                    - turn.start
                )

                if duration < min_duration:

                    continue

                results.append(
                    {
                        "speaker":
                        speaker_map[speaker],

                        "start":
                        round(
                            turn.start,
                            3,
                        ),

                        "end":
                        round(
                            turn.end,
                            3,
                        ),
                    }
                )

            results = self._merge_adjacent(
                results
            )

            return results

    def _merge_adjacent(
        self,
        segments,
        gap_threshold=0.3,
    ):

        if not segments:

            return []

        merged = [
            segments[0].copy()
        ]

        for seg in segments[1:]:

            last = merged[-1]

            gap = (
                seg["start"]
                - last["end"]
            )

            if (
                seg["speaker"]
                == last["speaker"]
                and gap < gap_threshold
            ):

                last["end"] = seg["end"]

            else:

                merged.append(
                    seg.copy()
                )

        return merged

    @classmethod
    def unload_model(cls):

        if cls._shared_pipeline:

            del cls._shared_pipeline

            cls._shared_pipeline = None

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()