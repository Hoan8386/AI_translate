"""
Stage 5: Chinese ASR
====================
Nhận dạng giọng nói tiếng Trung (Speech-to-Text).

Input:  segment_001.wav
Output: Segment đã có zh_text

Model: SenseVoice Small (FunASR)
"""

import gc
from pathlib import Path
from typing import List

import torch
from funasr import AutoModel

from models_data.segment import Segment
from utils.logger import get_logger, log_stage
from utils.timer import Timer
from utils.gpu_manager import GPUManager
from config.settings import get_settings

logger = get_logger(__name__)


class ChineseASR:
    """
    Nhận dạng tiếng Trung bằng SenseVoice Small.
    """

    STAGE_NUM = 5
    STAGE_NAME = "Chinese ASR (SenseVoice)"

    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()
        self._model = None

    def _load_model(self):
        """Load SenseVoice model"""

        if self._model is not None:
            return

        logger.info("Loading ASR model: iic/SenseVoiceSmall")

        try:
            self._model = AutoModel(
                model="iic/SenseVoiceSmall",
                trust_remote_code=True,
                disable_update=True,
                device=self.gpu.device,
            )

            logger.info(f"Model type: {type(self._model)}")

            if self._model is None:
                raise RuntimeError("Không load được SenseVoice model.")

            logger.info("SenseVoice model loaded")

        except Exception as e:
            logger.exception("Load SenseVoice thất bại")
            raise RuntimeError(f"Load SenseVoice thất bại: {e}")

    def _unload_model(self):
        """Giải phóng VRAM"""

        if self._model is not None:
            del self._model
            self._model = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("SenseVoice model unloaded")

    def process(self, segments: List[Segment]) -> List[Segment]:
        """
        Chạy ASR cho tất cả segment.
        """

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")

        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):

            try:
                self.gpu.ensure_free(2000)

                self._load_model()

                total = len(segments)

                for i, segment in enumerate(segments):

                    audio_path = segment.source_audio

                    if not audio_path:
                        logger.warning(
                            f"Segment {segment.id} không có source_audio"
                        )
                        continue

                    if not Path(audio_path).exists():
                        logger.warning(
                            f"Audio không tồn tại: {audio_path}"
                        )
                        continue

                    logger.info(
                        f"ASR [{i+1}/{total}] "
                        f"Segment {segment.id} ({segment.speaker})"
                    )

                    try:

                        result = self._model.generate(
                            input=audio_path,
                            language="zh",
                        )

                        if result and len(result) > 0:

                            text = result[0].get("text", "")
                            text = self._clean_text(text)

                            segment.zh_text = text
                            segment.confidence = result[0].get(
                                "confidence", 0.0
                            )

                            logger.info(
                                f'  → "{text}" '
                                f"(conf={segment.confidence:.2f})"
                            )

                        else:

                            segment.zh_text = ""
                            segment.confidence = 0.0

                            logger.warning(
                                f"Segment {segment.id}: không nhận dạng được"
                            )

                    except Exception as e:

                        logger.exception(
                            f"ASR error ở segment {segment.id}"
                        )

                        segment.zh_text = ""
                        segment.confidence = 0.0

                recognized = sum(
                    1 for s in segments
                    if getattr(s, "zh_text", "")
                )

                logger.info(
                    f"ASR hoàn thành: "
                    f"{recognized}/{len(segments)} segments"
                )

            finally:
                self._unload_model()

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")

        return segments

    def _clean_text(self, text: str) -> str:
        """
        Loại bỏ tag đặc biệt của SenseVoice.

        Ví dụ:
        <|zh|><|NEUTRAL|><|Speech|>大家好
        ->
        大家好
        """

        import re

        text = re.sub(r"<\|.*?\|>", "", text)

        return text.strip()