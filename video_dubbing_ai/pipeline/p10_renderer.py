"""
Stage 10: Video Renderer
==========================
Ghép tất cả lại thành video cuối cùng.

Input:  video_lipsync.mp4 + final_audio.wav
Output: output.mp4
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from utils.logger import get_logger, log_stage
from utils.timer import Timer

logger = get_logger(__name__)


class VideoRenderer:
    """Render video cuối cùng bằng ffmpeg, ưu tiên encoder GPU khi có sẵn."""

    STAGE_NUM = 10
    STAGE_NAME = "Video Renderer"

    def process(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        original_video: Optional[str] = None,
    ) -> str:
        """Render video cuối cùng và lưu output vào thư mục output."""
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")

        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video không tồn tại: {video_path}")
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio không tồn tại: {audio_path}")

            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(video_path, output_dir / f"{Path(video_path).stem}_input.mp4")
            shutil.copy2(audio_path, output_dir / f"{Path(audio_path).stem}_input.wav")

            normalized_audio = self._prepare_audio(audio_path, output_dir / f"{Path(audio_path).stem}_normalized.wav")
            self._render(video_path, normalized_audio, output_path)

            if not Path(output_path).exists():
                raise RuntimeError(f"Render thất bại, output không tồn tại: {output_path}")

            manifest = {
                "stage": self.STAGE_NAME,
                "input_video": video_path,
                "input_audio": audio_path,
                "normalized_audio": str(normalized_audio),
                "output_video": output_path,
            }
            with (output_dir / "stage10_manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)

            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"Output video: {output_path} ({size_mb:.1f}MB)")

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return output_path

    def _prepare_audio(self, audio_path: str, output_path: str) -> str:
        output_path = str(output_path)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(audio_path),
            "-vn",
            "-ac", "2",
            "-ar", "48000",
            "-c:a", "pcm_s16le",
            str(output_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio normalize error:\n{result.stdout}\n{result.stderr}"
            )
        return output_path

    def _render(self, video_path: str, audio_path: str, output_path: str):
        cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel", "info",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-movflags", "+faststart",
            "-shortest",
        ]

        if self._supports_encoder("h264_nvenc"):
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"])
            logger.info("Rendering final video with GPU encoder h264_nvenc...")
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "23"])
            logger.info("Rendering final video with CPU encoder libx264...")

        cmd.extend(["-c:a", "aac", "-b:a", "192k", str(output_path)])
        logger.debug(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=1800,
        )
        if result.returncode != 0:
            logger.warning("Render command failed, thử fallback copy-video command")
            logger.warning("stdout=%s", result.stdout[:1000])
            logger.warning("stderr=%s", result.stderr[:1000])
            return self._render_with_copy(video_path, audio_path, output_path)

        logger.info("Render hoàn thành!")

    def _supports_encoder(self, encoder_name: str) -> bool:
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                timeout=60,
            )
        except Exception:
            return False

        text = (result.stdout or "") + (result.stderr or "")
        return encoder_name in text

    def _render_with_copy(self, video_path: str, audio_path: str, output_path: str):
        cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel", "info",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output_path),
        ]
        logger.info("Fallback render: copy video stream + encode audio")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=1800,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg render error:\n{result.stdout}\n{result.stderr}"
            )
        logger.info("Fallback render hoàn thành!")
        return output_path

    def add_subtitles(
        self,
        video_path: str,
        srt_path: str,
        output_path: str,
    ) -> str:
        """Thêm subtitle vào video (optional, cho phiên bản sau)."""
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"subtitles={srt_path}",
            "-c:a", "copy",
            "-y",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"Subtitle error:\n{result.stderr}")

        return output_path
