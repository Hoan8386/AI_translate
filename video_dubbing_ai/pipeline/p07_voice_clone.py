"""
Stage 7: Voice Clone (Fish Speech - Local Inference)
===================================================
Sinh giọng nói tiếng Việt từ text (vi_text) theo từng speaker.

Input:
    List[Segment] (đã có vi_text)

Output:
    List[Segment] (thêm cloned_audio_path)
"""

from pathlib import Path
from typing import List, Optional
import torch
import gc
import soundfile as sf
import numpy as np

from models_data.segment import Segment
from utils.logger import get_logger, log_stage
from utils.timer import Timer
from utils.gpu_manager import GPUManager
from config.settings import get_settings

logger = get_logger(__name__)


class VoiceCloner:

    STAGE_NUM = 7
    STAGE_NAME = "Voice Clone (Fish Speech)"

    _shared_llama_model = None
    _shared_decoder_model = None

    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()

        self.llama_model = VoiceCloner._shared_llama_model
        self.decoder_model = VoiceCloner._shared_decoder_model

    # =========================
    # LOAD MODEL
    # =========================
    def _load_model(self):
        if self.llama_model is not None and self.decoder_model is not None:
            return

        logger.info("Loading Fish Speech models...")

        self.gpu.ensure_free(4000)

        try:
            from tools.llama.inference import load_model as load_llama
            from tools.vqvq.inference import load_model as load_decoder

            base_dir = self.settings.third_party_dir / "fish-speech"

            # LLaMA (speech tokenizer / generator)
            llama_path = base_dir / self.settings.voice_clone.llama_checkpoint_path
            self.llama_model = load_llama(
                checkpoint_path=str(llama_path),
                device=self.settings.gpu.device
            )

            # Decoder (waveform generator)
            decoder_path = base_dir / self.settings.voice_clone.decoder_checkpoint_path
            self.decoder_model = load_decoder(
                config_name=self.settings.voice_clone.decoder_config_name,
                checkpoint_path=str(decoder_path),
                device=self.settings.gpu.device
            )

            VoiceCloner._shared_llama_model = self.llama_model
            VoiceCloner._shared_decoder_model = self.decoder_model

            logger.info("Fish Speech models loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load Fish Speech: {e}")
            raise

    # =========================
    # SPEAKER REF AUDIO
    # =========================
    def _get_ref_audio(self, speaker: str) -> Optional[str]:
        ref_path = self.settings.cache_dir / f"ref_{speaker}.wav"
        return str(ref_path) if Path(ref_path).exists() else None

    # =========================
    # INFERENCE (SAFE VERSION)
    # =========================
    def _infer(self, text: str, ref_audio: Optional[str]):
        """
        NOTE: Đây là wrapper an toàn.
        Bạn cần thay bằng Fish Speech real API khi connect full.
        """

        # ⚠️ PLACEHOLDER inference (tránh crash pipeline)
        sr = 22050
        duration = max(1, len(text) * 0.05)

        t = np.linspace(0, duration, int(sr * duration))
        wav = 0.01 * np.sin(2 * np.pi * 220 * t)

        return wav, sr

        # =========================
        # REAL FISH SPEECH (KHI BẬT)
        # =========================
        # latent = self.llama_model.generate(
        #     text=text,
        #     reference_audio=ref_audio
        # )
        #
        # wav, sr = self.decoder_model.decode(latent)
        # return wav, sr

    # =========================
    # MAIN PROCESS
    # =========================
    def process(self, segments: List[Segment], speakers=None, output_dir=None) -> List[Segment]:
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")

        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):

            self._load_model()

            output_dir = Path(output_dir or (self.settings.temp_dir / "cloned_segments"))
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Voice cloning {len(segments)} segments...")

            for i, segment in enumerate(segments):

                if not segment.vi_text or not segment.vi_text.strip():
                    continue

                try:
                    ref_audio = self._get_ref_audio(segment.speaker)

                    if not ref_audio:
                        logger.warning(f"No reference audio for speaker: {segment.speaker}")

                    logger.info(
                        f"[{i+1}/{len(segments)}] "
                        f"{segment.speaker}: {segment.vi_text[:50]}"
                    )

                    wav, sr = self._infer(segment.vi_text, ref_audio)

                    out_path = output_dir / f"seg_{i}_{segment.speaker}.wav"

                    sf.write(str(out_path), wav, sr)

                    segment.cloned_audio_path = str(out_path)

                except Exception as e:
                    logger.error(f"Voice clone failed at segment {i}: {e}")
                    segment.cloned_audio_path = ""

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return segments

    # =========================
    # CLEANUP
    # =========================
    @classmethod
    def unload_model(cls):
        logger.info("Unloading Fish Speech models...")

        cls._shared_llama_model = None
        cls._shared_decoder_model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("GPU memory cleared.")