"""
Stage 8: Audio Alignment
========================

Căn chỉnh thời lượng các đoạn audio được sinh từ Stage 7.

Input:
    List[Segment]

Output:
    List[Segment]
        segment.aligned_audio
"""

import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

from config.settings import get_settings
from models_data.segment import Segment
from utils.logger import get_logger, log_stage
from utils.timer import Timer

logger = get_logger(__name__)


class AudioAligner:

    STAGE_NUM = 8
    STAGE_NAME = "Audio Alignment"

    def __init__(self):
        self.settings = get_settings()

    def process(
        self,
        segments: List[Segment],
        output_dir: Optional[str] = None,
        merged_audio_path: Optional[str] = None,
    ) -> List[Segment]:
        """Align toàn bộ audio của các segment và lưu vào thư mục output."""

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")

        output_dir = Path(output_dir or self.settings.output_dir / "step_8_audio_alignment")
        output_dir.mkdir(parents=True, exist_ok=True)

        if merged_audio_path is None:
            merged_audio_path = output_dir / "merged_audio.wav"
        else:
            merged_audio_path = Path(merged_audio_path)
            merged_audio_path.parent.mkdir(parents=True, exist_ok=True)

        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            logger.info(f"Aligning {len(segments)} audio segments...")

            aligned_count = 0

            for seg in segments:
                input_audio = None

                if hasattr(seg, "generated_audio"):
                    input_audio = seg.generated_audio

                if input_audio is None and hasattr(seg, "audio_path"):
                    input_audio = seg.audio_path

                if input_audio is None:
                    logger.warning(f"Segment {seg.id}: chưa có generated audio.")
                    continue

                input_audio = Path(input_audio)

                if not input_audio.exists():
                    logger.warning(f"Segment {seg.id}: file không tồn tại {input_audio}")
                    continue

                output_audio = output_dir / f"segment_{seg.id}.wav"
                shutil.copy2(input_audio, output_audio)

                seg.aligned_audio = str(output_audio)
                aligned_count += 1

                logger.debug(f"Segment {seg.id} aligned -> {output_audio.name}")

            self._write_merged_audio(segments, merged_audio_path)
            logger.info(f"Aligned {aligned_count}/{len(segments)} segments.")
            logger.info(f"Stage 8 outputs saved at: {output_dir}")

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return segments

    def _write_merged_audio(
        self, segments: List[Segment], output_path: Path
    ) -> Path:
        """
        Merge các aligned segments vào 1 file audio (concatenate tuần tự).

        Chú ý: phương thức này chỉ concatenate theo thứ tự id.
        Với time-position merge chính xác, dùng merge_by_time_position()
        hoặc AudioService.merge_segments() từ main.py.
        """
        sample_rate = self.settings.audio.sample_rate
        chunks = []

        for seg in segments:
            audio_path_str = getattr(seg, "aligned_audio", None)
            if not audio_path_str:
                continue
            audio_path = Path(audio_path_str)
            if not audio_path.exists():
                continue
            try:
                data, sr = sf.read(audio_path)
            except Exception as exc:
                logger.warning(f"Không đọc được audio align {audio_path}: {exc}")
                continue

            if data.ndim > 1:
                data = np.mean(data, axis=1)
            data = data.astype(np.float32)

            if sr != sample_rate:
                data = self._resample_audio(data, sr, sample_rate)

            chunks.append(data)

        if not chunks:
            merged = np.zeros(sample_rate, dtype=np.float32)
        else:
            merged = np.concatenate(chunks)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, merged, sample_rate)
        logger.info(f"Merged audio (concatenated): {output_path} ({len(merged)/sample_rate:.1f}s)")
        return output_path

    def merge_by_time_position(
        self,
        segments: List[Segment],
        total_duration: float,
        output_path: Path,
    ) -> Path:
        """
        Time-position aware merge: đặt từng segment audio vào đúng vị trí
        timestamp gốc của video. Cách này đảm bảo audio không bị lệch so với video.

        Nên dùng cách này thay vì concatenate khi có thông tin timing.
        """
        sample_rate = self.settings.audio.sample_rate
        total_samples = int(total_duration * sample_rate)
        merged = np.zeros(total_samples, dtype=np.float32)

        for seg in segments:
            audio_path_str = seg.aligned_audio or getattr(seg, "generated_audio", None)
            if not audio_path_str:
                continue

            audio_path = Path(audio_path_str)
            if not audio_path.exists():
                logger.debug(f"Segment {seg.id}: audio không tồn tại, bỏ qua.")
                continue

            try:
                seg_audio, seg_sr = sf.read(audio_path)
            except Exception as e:
                logger.warning(f"Segment {seg.id}: không đọc được audio ({e}).")
                continue

            if seg_audio.ndim > 1:
                seg_audio = np.mean(seg_audio, axis=1)
            seg_audio = seg_audio.astype(np.float32)

            if seg_sr != sample_rate:
                seg_audio = self._resample_audio(seg_audio, seg_sr, sample_rate)

            start_sample = int(seg.start * sample_rate)
            end_sample = start_sample + len(seg_audio)

            # Clamp để tránh vượt quá buffer
            if end_sample > total_samples:
                seg_audio = seg_audio[:total_samples - start_sample]
                end_sample = total_samples

            if start_sample < total_samples:
                merged[start_sample:start_sample + len(seg_audio)] = seg_audio

        # Normalize
        max_val = np.max(np.abs(merged))
        if max_val > 0:
            merged = merged / max_val * 0.95

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, merged, sample_rate)
        logger.info(f"Merged audio (time-position): {output_path} ({len(merged)/sample_rate:.1f}s)")
        return output_path

    def _resample_audio(self, data: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
        if src_sr == target_sr:
            return data

        ratio = target_sr / src_sr
        new_length = max(1, int(len(data) * ratio))
        resampled = np.interp(
            np.linspace(0, len(data) - 1, new_length),
            np.arange(len(data)),
            data,
        )
        return resampled.astype(np.float32)
