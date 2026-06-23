"""
Stage 3: Speaker Detector (Bản tối ưu phối hợp Stage 2 Denoise)
==============================================================
Nhận diện và phân biệt nhiều người nói từ file âm thanh đã được cô lập Vocal sạch.

Input:
    audio.wav (Đường dẫn file từ Stage 2)

Output:
    [
        {"speaker": "speaker_1", "start": 0.0, "end": 5.0},
        {"speaker": "speaker_2", "start": 5.0, "end": 11.0}
    ]

Technology:
    pyannote.audio + torchaudio
"""

from pathlib import Path
from typing import List, Dict
import json

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

    # Shared pipeline giữa tất cả các instance (Singleton Pattern)
    _shared_pipeline = None

    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()
        self._pipeline = SpeakerDetector._shared_pipeline

    def _load_model(self):
        """Load pyannote pipeline (chỉ load 1 lần duy nhất)"""
        if self._pipeline is not None:
            return

        from pyannote.audio import Pipeline

        hf_token = self.settings.speaker.hf_token
        if not hf_token:
            raise ValueError(
                "HF_TOKEN chưa được thiết lập trong file .env!\n"
                "Vui lòng accept license tại pyannote/speaker-diarization-3.1"
            )

        logger.info(f"Loading pyannote model: {self.settings.speaker.model_name}")
        pipeline = Pipeline.from_pretrained(
            self.settings.speaker.model_name,
            use_auth_token=hf_token,
        )

        if pipeline is None:
            raise RuntimeError("Không thể khởi tạo pyannote pipeline từ HuggingFace.")

        # Chuyển mô hình lên GPU RTX 5060
        if self.gpu.device == "cuda":
            logger.info("Moving pyannote model to CUDA...")
            pipeline.to(torch.device("cuda"))

        self._pipeline = pipeline
        SpeakerDetector._shared_pipeline = pipeline
        logger.info("pyannote model loaded thành công.")

    def process(self, audio_path: str) -> List[Dict]:
        """Phân tách và nhận diện danh tính người nói (Speaker Diarization)"""
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")

        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            audio_file = Path(audio_path)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio đầu vào không tồn tại: {audio_path}")

            # Đảm bảo bộ nhớ GPU trống trước khi chạy
            self.gpu.ensure_free(2000)
            self._load_model()

            logger.info("Đang nạp dữ liệu âm thanh qua Torchaudio...")
            try:
                waveform, sample_rate = torchaudio.load(str(audio_file))
                
                # Lớp phòng thủ: Nếu Stage 2 lỡ lọt file Stereo, tự động xử lý downmix an toàn
                if waveform.shape[0] > 1:
                    logger.warning("Phát hiện audio đa kênh lọt vào Stage 3. Đang tự động Downmix...")
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                audio_input = {"waveform": waveform, "sample_rate": sample_rate}
            except Exception as e:
                logger.warning(f"Không thể load qua torchaudio ({e}). Fallback về đường dẫn file thô.")
                audio_input = str(audio_file)

            # Khởi tạo tham số cấu hình động từ settings
            diarization_params = {}
            if hasattr(self.settings.speaker, "min_speakers") and self.settings.speaker.min_speakers:
                diarization_params["min_speakers"] = self.settings.speaker.min_speakers
            if hasattr(self.settings.speaker, "max_speakers") and self.settings.speaker.max_speakers:
                diarization_params["max_speakers"] = self.settings.speaker.max_speakers
            if hasattr(self.settings.speaker, "num_speakers") and self.settings.speaker.num_speakers:
                diarization_params["num_speakers"] = self.settings.speaker.num_speakers

            logger.info("Running speaker diarization pipeline...")
            diarization = self._pipeline(audio_input, **diarization_params)

            results = []
            speaker_map = {}
            speaker_counter = 0

            # Lặp qua các phân đoạn trả về từ mô hình pyannote
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                logger.debug(f"Raw Segment: {turn.start:.2f}s - {turn.end:.2f}s | Speaker: {speaker}")

                if speaker not in speaker_map:
                    speaker_counter += 1
                    speaker_map[speaker] = f"speaker_{speaker_counter}"

                duration = turn.end - turn.start
                
                # Vì Stage 2 lọc Vocal rất sạch rồi, ta có thể tự tin hạ ngưỡng duration xuống 0.2s 
                # để không bị nuốt các câu đối thoại cực ngắn (ví dụ: "Vâng", "Được", "Ừ",...)
                min_duration = getattr(self.settings.speaker, "min_segment_duration", 0.2)
                if duration < min_duration:
                    continue

                results.append(
                    {
                        "speaker": speaker_map[speaker],
                        "start": round(turn.start, 3),
                        "end": round(turn.end, 3),
                    }
                )

            # Gộp các đoạn hội thoại liền mạch gần nhau của cùng 1 người nói
            results = self._merge_adjacent(results)

            unique_speakers = {r["speaker"] for r in results}
            logger.info(f"Kết quả Stage 3: Phát hiện {len(unique_speakers)} người nói với {len(results)} segments.")

            # Xuất log JSON lưu trữ
            try:
                job_id = audio_file.parent.parent.name if audio_file.parent.name == "audio" else audio_file.stem
                step_output_dir = self.settings.output_dir / "step_3_speaker_detector"
                step_output_dir.mkdir(parents=True, exist_ok=True)
                
                dest_path = step_output_dir / f"{job_id}_speakers.json"
                with open(dest_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                logger.info(f"Đã xuất kết quả Stage 3 ra file: {dest_path}")
            except Exception as e:
                logger.warning(f"Lỗi khi ghi file log Stage 3: {e}")

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return results

    def _merge_adjacent(self, segments: List[Dict], gap_threshold: float = 0.3) -> List[Dict]:
        """Gộp các phân đoạn kế cận nếu cùng một người nói và khoảng nghỉ nhỏ hơn ngưỡng"""
        if not segments:
            return segments

        merged = [segments[0].copy()]
        for seg in segments[1:]:
            last = merged[-1]
            gap = seg["start"] - last["end"]

            if seg["speaker"] == last["speaker"] and gap < gap_threshold:
                last["end"] = seg["end"]
            else:
                merged.append(seg.copy())

        if len(merged) < len(segments):
            logger.info(f"Tối ưu hóa phân đoạn: Đã gộp {len(segments)} → {len(merged)} segments")

        return merged

    @classmethod
    def unload_model(cls):
        """Giải phóng hoàn toàn mô hình Pyannote khỏi VRAM GPU"""
        if cls._shared_pipeline is not None:
            logger.info("Unloading pyannote model khỏi VRAM...")
            del cls._shared_pipeline
            cls._shared_pipeline = None

            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Đã giải phóng hoàn toàn bộ nhớ GPU của Stage 3.")