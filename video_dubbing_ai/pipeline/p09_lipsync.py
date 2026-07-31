"""
Stage 9: Lip Sync (Wav2Lip GAN)
================================
Khớp khẩu hình môi nhân vật trong video theo audio tiếng Việt.

Input:
    - normalized_video.mp4
    - merged_audio.wav (từ Stage 8)

Output:
    - lipsync_video.mp4
"""

import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from config.settings import get_settings
from utils.gpu_manager import GPUManager
from utils.logger import get_logger, log_stage
from utils.timer import Timer

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
        alt = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
        if alt.exists():
            return alt
        return path

    def process(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Thực hiện lip sync bằng Wav2Lip và lưu output vào thư mục output."""
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")

        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            video_path = str(video_path)
            audio_path = str(audio_path)
            output_path = str(output_path)

            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video không tồn tại: {video_path}")
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio không tồn tại: {audio_path}")

            checkpoint = self._get_checkpoint_path()
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Wav2Lip checkpoint không tồn tại: {checkpoint}\n"
                    f"Tải từ: https://github.com/Rudrabha/Wav2Lip#getting-the-weights"
                )

            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(video_path, output_dir / f"{Path(video_path).stem}_input.mp4")
            shutil.copy2(audio_path, output_dir / f"{Path(audio_path).stem}_input.wav")

            self.gpu.ensure_free(2000)
            logger.info("Khởi chạy Wav2Lip inference trên GPU/CPU nhẹ hơn để giảm áp lực RAM...")

            lipsync_settings = self.settings.lipsync
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "2")
            env.setdefault("MKL_NUM_THREADS", "2")
            env.setdefault("OPENBLAS_NUM_THREADS", "2")
            env.setdefault("NUMEXPR_NUM_THREADS", "2")
            if self.settings.gpu.device == "cuda":
                env.setdefault("CUDA_VISIBLE_DEVICES", "0")
                env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

            face_det_batch_size = max(1, int(lipsync_settings.face_det_batch_size))
            wav2lip_batch_size = max(1, int(lipsync_settings.wav2lip_batch_size))
            resize_factor = max(2, int(lipsync_settings.resize_factor))

            cmd = [
                sys.executable,
                str(WAV2LIP_DIR / "inference.py"),
                "--checkpoint_path", str(checkpoint),
                "--face", video_path,
                "--audio", audio_path,
                "--outfile", output_path,
                "--face_det_batch_size", str(face_det_batch_size),
                "--wav2lip_batch_size", str(wav2lip_batch_size),
                "--resize_factor", str(resize_factor),
                "--pads", *[str(p) for p in lipsync_settings.pads],
                "--nosmooth",
            ]

            logger.info(f"Wav2Lip command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                cwd=str(WAV2LIP_DIR),
                timeout=1800,
                env=env,
            )

            if result.returncode != 0:
                logger.error(
                    "Wav2Lip stdout:\n%s\nWav2Lip stderr:\n%s",
                    result.stdout[:1000],
                    result.stderr[:1000],
                )
                raise RuntimeError(
                    f"Wav2Lip inference thất bại (return code {result.returncode})"
                )

            if not Path(output_path).exists():
                raise RuntimeError("Wav2Lip không tạo được file output")

            manifest = {
                "stage": self.STAGE_NAME,
                "device": self.settings.gpu.device,
                "gpu_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "face_det_batch_size": face_det_batch_size,
                "wav2lip_batch_size": wav2lip_batch_size,
                "resize_factor": resize_factor,
                "input_video": video_path,
                "input_audio": audio_path,
                "output_video": output_path,
            }
            with (output_dir / "stage9_manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)

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