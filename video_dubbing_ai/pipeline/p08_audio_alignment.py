"""
Stage 8: Audio Alignment
========================

Căn chỉnh thời lượng các đoạn audio được sinh từ Stage 7.

Input:
    List[Segment]

Output:
    List[Segment]
        segment.aligned_audio

Hiện tại chỉ copy audio sang thư mục aligned.
Sau này có thể bổ sung:
    - time stretch
    - silence padding
    - loudness normalize
"""

from pathlib import Path
from typing import List
import shutil

from models_data.segment import Segment
from utils.logger import get_logger, log_stage
from utils.timer import Timer

logger = get_logger(__name__)


class AudioAligner:

    STAGE_NUM = 8
    STAGE_NAME = "Audio Alignment"

    def __init__(self):
        pass

    def process(
        self,
        segments: List[Segment],
        output_dir: str,
    ) -> List[Segment]:
        """
        Align toàn bộ audio của các segment.

        Parameters
        ----------
        segments
            Danh sách Segment.

        output_dir
            Thư mục lưu audio sau khi align.

        Returns
        -------
        List[Segment]
        """

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):

            logger.info(
                f"Aligning {len(segments)} audio segments..."
            )

            aligned_count = 0

            for seg in segments:

                ####################################################
                # Lấy audio sinh từ Stage 7
                ####################################################

                input_audio = None

                if hasattr(seg, "generated_audio"):
                    input_audio = seg.generated_audio

                if input_audio is None and hasattr(seg, "audio_path"):
                    input_audio = seg.audio_path

                if input_audio is None:
                    logger.warning(
                        f"Segment {seg.id}: chưa có generated audio."
                    )
                    continue

                input_audio = Path(input_audio)

                if not input_audio.exists():
                    logger.warning(
                        f"Segment {seg.id}: file không tồn tại {input_audio}"
                    )
                    continue

                ####################################################
                # Output
                ####################################################

                output_audio = output_dir / f"segment_{seg.id}.wav"

                ####################################################
                # Hiện tại chỉ copy.
                # Sau này thay bằng ffmpeg/time-stretch nếu cần.
                ####################################################

                shutil.copy2(input_audio, output_audio)

                ####################################################
                # Update Segment
                ####################################################

                seg.aligned_audio = str(output_audio)

                aligned_count += 1

                logger.debug(
                    f"Segment {seg.id} aligned -> {output_audio.name}"
                )

            logger.info(
                f"Aligned {aligned_count}/{len(segments)} segments."
            )

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")

        return segments