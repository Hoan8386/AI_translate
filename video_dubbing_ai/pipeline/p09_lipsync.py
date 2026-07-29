"""
Stage 9: Lip Sync (Wav2Lip GAN)
================================
Khớp khẩu hình môi nhân vật trong video theo audio tiếng Việt.

Input:
    - normalized_video.mp4
    - merged_audio.wav (từ Stage 8)

Output:
    - lipsync_video.mp4

Gọi Wav2Lip inference.py qua subprocess để tránh xung đột
với argparse globals và import paths.
"""

import subprocess
import sys
from pathlib import Path

import torch
import gc

from utils.logger import get_logger, log_stage
from utils.timer import Timer
from utils.gpu_manager import GPUManager
from config.settings import get_settings

logger = get_logger(__name__)

WAV2LIP_DIR = Path(__file__).parent.parent / "third_party" / "Wav2Lip"


class LipSyncProcessor:

    STAGE_NUM = 9
    STAGE_NAME = "Lip Sync (Wav2Lip)"

    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()

    def _get_checkpoint_path(self) -> Path:
        """Trả về đường dẫn checkpoint Wav2Lip."""
        path = Path(self.settings.lipsync.checkpoint_path)
        if path.exists():
            return path
        # Thử tìm trong thư mục mặc định
        alt = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
        if alt.exists():
            return alt
        return path  # Trả về path gốc (sẽ raise error nếu không tồn tại)

    def process(self, video_path: str, audio_path: str, output_path: str) -> str:
        """
        Thực hiện lip sync bằng Wav2Lip.

        Parameters
        ----------
        video_path : str
            Video gốc (normalized)
        audio_path : str
            Audio tiếng Việt (merged)
        output_path : str
            Đường dẫn output video

        Returns
        -------
        str : đường dẫn output
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")

        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):

            video_path = str(video_path)
            audio_path = str(audio_path)
            output_path = str(output_path)

            # Kiểm tra input files
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video không tồn tại: {video_path}")
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio không tồn tại: {audio_path}")

            # Kiểm tra checkpoint
            checkpoint = self._get_checkpoint_path()
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Wav2Lip checkpoint không tồn tại: {checkpoint}\n"
                    f"Tải từ: https://github.com/Rudrabha/Wav2Lip#getting-the-weights"
                )

            # Đảm bảo output dir tồn tại
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Đảm bảo đủ VRAM
            self.gpu.ensure_free(2000)

            # Chạy Wav2Lip inference qua subprocess
            logger.info("Khởi chạy Wav2Lip inference...")

            lipsync_settings = self.settings.lipsync

            cmd = [
                sys.executable,
                str(WAV2LIP_DIR / "inference.py"),
                "--checkpoint_path", str(checkpoint),
                "--face", video_path,
                "--audio", audio_path,
                "--outfile", output_path,
                "--face_det_batch_size", str(lipsync_settings.face_det_batch_size),
                "--wav2lip_batch_size", str(lipsync_settings.wav2lip_batch_size),
                "--resize_factor", str(lipsync_settings.resize_factor),
                "--pads", *[str(p) for p in lipsync_settings.pads],
                "--nosmooth",
            ]

            logger.info(f"Wav2Lip command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(WAV2LIP_DIR),
                timeout=600,  # 10 phút timeout
            )

            if result.returncode != 0:
                logger.error(f"Wav2Lip stderr: {result.stderr[:1000]}")
                raise RuntimeError(
                    f"Wav2Lip inference thất bại (return code {result.returncode})"
                )

            if not Path(output_path).exists():
                raise RuntimeError(
                    "Wav2Lip không tạo được file output"
                )

            logger.info(f"Lip sync hoàn tất: {output_path}")

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return output_path

    @classmethod
    def unload_model(cls):
        """Giải phóng VRAM (nếu có model trong memory)."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Đã dọn sạch bộ nhớ GPU của bước Lip Sync.")