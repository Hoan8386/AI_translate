"""
Stage 6: Translation
====================
Dịch tiếng Trung → tiếng Việt.

Input:  大家好
Output: Xin chào mọi người

Engine: Google Translate (deep-translator)
"""

from typing import List

from deep_translator import GoogleTranslator

from models_data.segment import Segment
from utils.logger import get_logger, log_stage
from utils.timer import Timer

logger = get_logger(__name__)


class Translator:
    """
    Dịch tiếng Trung → tiếng Việt bằng Google Translate.
    """

    STAGE_NUM = 6
    STAGE_NAME = "Translation (ZH → VI)"

    def __init__(self):
        self._translator = GoogleTranslator(
            source="zh-CN",
            target="vi"
        )

    def _translate_single(self, text: str) -> str:
        """
        Dịch một câu.
        """

        if not text or not text.strip():
            return ""

        try:
            translated = self._translator.translate(text)

            if translated is None:
                return ""

            return translated.strip()

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return f"[TRANSLATION ERROR: {text}]"

    def process(self, segments: List[Segment]) -> List[Segment]:
        """
        Dịch tất cả segments.
        """

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")

        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):

            to_translate = [
                s for s in segments
                if s.zh_text and s.zh_text.strip()
            ]

            if len(to_translate) == 0:
                logger.warning("Không có text cần dịch.")
                log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
                return segments

            logger.info(
                f"Dịch {len(to_translate)} segments (ZH → VI)"
            )

            for i, segment in enumerate(to_translate):

                logger.info(
                    f'Translating [{i+1}/{len(to_translate)}]: "{segment.zh_text}"'
                )

                segment.vi_text = self._translate_single(
                    segment.zh_text
                )

                logger.info(
                    f'  → "{segment.vi_text}"'
                )

            translated = sum(
                1 for s in segments
                if s.vi_text and s.vi_text.strip()
            )

            logger.info(
                f"Translation hoàn thành: "
                f"{translated}/{len(segments)} segments"
            )

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")

        return segments