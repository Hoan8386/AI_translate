"""
Stage 7: Voice Cloning (Fish Speech - Local Inference)
======================================================

Sinh giọng nói tiếng Việt cho từng Segment sử dụng Fish Speech local.

Input:
    - segments (List[Segment] với vi_text)
    - speakers  (Dict[str, Speaker])

Output:
    - segments (đã cập nhật generated_audio)

Flow:
    1. Load Fish Speech models (LLAMA + DAC decoder) lần đầu tiên
    2. Encode reference audio từ segment gốc
    3. Gọi TTSInferenceEngine.inference(ServeTTSRequest) để sinh audio
    4. Lưu kết quả WAV
    5. Unload models khỏi VRAM sau khi xong
"""

import gc
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("dubbing")

# Thêm fish-speech vào path
FISH_SPEECH_DIR = Path(__file__).parent.parent / "third_party" / "fish-speech"


class FishSpeechManager:
    """
    Singleton quản lý Fish Speech models.
    Load một lần, dùng cho toàn bộ segments, rồi unload.
    """

    _instance = None
    _engine = None
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_models(self):
        """Load LLAMA + DAC decoder models vào GPU."""
        if self._loaded:
            logger.info("[Fish Speech] Models đã được load sẵn.")
            return

        settings = get_settings()
        vc = settings.voice_clone

        # Thêm fish-speech vào sys.path
        fish_path = str(FISH_SPEECH_DIR)
        if fish_path not in sys.path:
            sys.path.insert(0, fish_path)

        # Resolve absolute paths
        llama_path = str(FISH_SPEECH_DIR / vc.llama_checkpoint_path)
        decoder_path = str(FISH_SPEECH_DIR / vc.decoder_checkpoint_path)

        if not Path(llama_path).exists():
            raise FileNotFoundError(
                f"LLAMA checkpoint không tồn tại: {llama_path}"
            )
        if not Path(decoder_path).exists():
            raise FileNotFoundError(
                f"Decoder checkpoint không tồn tại: {decoder_path}"
            )

        logger.info(f"[Fish Speech] Loading LLAMA model từ {llama_path}...")

        # Import Fish Speech modules
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder
        from fish_speech.models.text2semantic.inference import (
            launch_thread_safe_queue,
        )

        precision = torch.half if vc.half else torch.bfloat16

        # Load LLAMA queue
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=llama_path,
            device=vc.device,
            precision=precision,
            compile=False,
        )
        logger.info("[Fish Speech] LLAMA model loaded.")

        # Load DAC decoder
        logger.info(f"[Fish Speech] Loading decoder từ {decoder_path}...")
        decoder_model = load_decoder(
            config_name=vc.decoder_config_name,
            checkpoint_path=decoder_path,
            device=vc.device,
        )
        logger.info("[Fish Speech] Decoder model loaded.")

        # TTSInferenceEngine
        self._engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=False,
        )
        self._llama_queue = llama_queue
        self._decoder_model = decoder_model
        self._loaded = True

        logger.info("[Fish Speech] Tất cả models đã sẵn sàng.")

    @property
    def engine(self):
        if not self._loaded:
            self.load_models()
        return self._engine

    def unload(self):
        """Giải phóng VRAM."""
        if not self._loaded:
            return

        logger.info("[Fish Speech] Unloading models...")

        # Gửi None để stop worker thread
        if hasattr(self, "_llama_queue") and self._llama_queue:
            try:
                self._llama_queue.put(None)
            except Exception:
                pass

        # Delete references
        if hasattr(self, "_decoder_model") and self._decoder_model:
            del self._decoder_model
        if hasattr(self, "_engine") and self._engine:
            del self._engine

        self._engine = None
        self._decoder_model = None
        self._llama_queue = None
        self._loaded = False
        FishSpeechManager._engine = None
        FishSpeechManager._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("[Fish Speech] Đã dọn sạch bộ nhớ GPU.")


class VoiceCloner:

    def __init__(self):
        self.manager = FishSpeechManager()

    def _read_audio_bytes(self, audio_path: str) -> bytes:
        """Đọc file audio thành bytes để dùng làm reference."""
        with open(audio_path, "rb") as f:
            return f.read()

    def _generate_with_fish_speech(
        self, text: str, reference_audio_path: str, reference_text: str,
        output_path: str
    ) -> bool:
        """
        Sinh audio bằng Fish Speech.

        Returns:
            True nếu thành công, False nếu thất bại.
        """
        try:
            sys_path_backup = sys.path.copy()
            fish_path = str(FISH_SPEECH_DIR)
            if fish_path not in sys.path:
                sys.path.insert(0, fish_path)

            from fish_speech.utils.schema import (
                ServeReferenceAudio,
                ServeTTSRequest,
            )

            # Tạo request
            references = []
            if reference_audio_path and Path(reference_audio_path).exists():
                audio_bytes = self._read_audio_bytes(reference_audio_path)
                if len(audio_bytes) > 0:
                    references.append(
                        ServeReferenceAudio(
                            audio=audio_bytes,
                            text=reference_text or "",
                        )
                    )

            request = ServeTTSRequest(
                text=text,
                references=references,
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.8,
                repetition_penalty=1.1,
                temperature=0.8,
                format="wav",
                streaming=False,
                normalize=True,
            )

            # Gọi inference
            engine = self.manager.engine
            sample_rate = None
            audio_data = None

            for result in engine.inference(request):
                if result.code == "error":
                    logger.error(
                        f"[Fish Speech] Lỗi inference: {result.error}"
                    )
                    return False
                elif result.code == "final":
                    if isinstance(result.audio, tuple):
                        sample_rate, audio_data = result.audio
                    break

            if audio_data is None or sample_rate is None:
                logger.warning("[Fish Speech] Không sinh được audio.")
                return False

            # Lưu file WAV
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio_data, sample_rate)

            duration = len(audio_data) / sample_rate
            logger.info(
                f"[Fish Speech] Đã sinh {duration:.2f}s audio → {Path(output_path).name}"
            )
            return True

        except Exception as e:
            logger.error(f"[Fish Speech] Lỗi: {e}")
            return False

    def process(self, segments, speakers, generated_dir):
        """
        Sinh giọng nói tiếng Việt cho toàn bộ segments.

        Parameters
        ----------
        segments : List[Segment]
        speakers : Dict[str, Speaker]
        generated_dir : str

        Returns
        -------
        List[Segment]
        """
        logger.info(
            f"[Voice Clone] Bắt đầu xử lý {len(segments)} segments..."
        )

        os.makedirs(generated_dir, exist_ok=True)

        # Thử load Fish Speech
        fish_speech_available = False
        try:
            self.manager.load_models()
            fish_speech_available = True
            logger.info("[Voice Clone] Fish Speech đã sẵn sàng.")
        except Exception as e:
            logger.warning(
                f"[Voice Clone] Fish Speech không khả dụng: {e}"
            )
            logger.info(
                "[Voice Clone] Sẽ tạo WAV silence thay thế."
            )

        try:
            for seg in segments:

                # Lấy thông tin segment
                if isinstance(seg, dict):
                    text = seg.get("vi_text", "")
                    seg_id = seg.get("id", 0)
                    speaker = seg.get("speaker", "unknown")
                    source_audio = seg.get("source_audio", "")
                    zh_text = seg.get("zh_text", "")
                    seg_start = seg.get("start", 0.0)
                    seg_end = seg.get("end", 1.0)
                else:
                    text = getattr(seg, "vi_text", "")
                    seg_id = getattr(seg, "id", 0)
                    speaker = getattr(seg, "speaker", "unknown")
                    source_audio = getattr(seg, "source_audio", "")
                    zh_text = getattr(seg, "zh_text", "")
                    seg_start = getattr(seg, "start", 0.0)
                    seg_end = getattr(seg, "end", 1.0)

                if not text:
                    logger.warning(f"Segment {seg_id} không có vi_text.")
                    continue

                logger.info(
                    f"[Voice Clone] Segment {seg_id} ({speaker}): '{text[:40]}...'"
                )

                output_path = os.path.join(
                    generated_dir,
                    f"segment_{seg_id}.wav",
                )

                # Thử Fish Speech
                generated = False
                if fish_speech_available:
                    generated = self._generate_with_fish_speech(
                        text=text,
                        reference_audio_path=source_audio,
                        reference_text=zh_text,
                        output_path=output_path,
                    )

                # Fallback: tạo WAV silence
                if not generated:
                    need_silence = (
                        not os.path.exists(output_path)
                        or os.path.getsize(output_path) == 0
                    )
                    if need_silence:
                        duration = seg_end - seg_start
                        if duration <= 0:
                            duration = 1.0
                        sr = 16000
                        silence = np.zeros(
                            int(duration * sr), dtype=np.float32
                        )
                        sf.write(output_path, silence, sr)
                        logger.info(
                            f"Segment {seg_id}: fallback WAV silence ({duration:.2f}s)"
                        )

                # Update Segment
                if isinstance(seg, dict):
                    seg["generated_audio"] = output_path
                else:
                    seg.generated_audio = output_path

            logger.info("[Voice Clone] Hoàn thành toàn bộ segments.")
            return segments

        except Exception as e:
            logger.exception(e)
            raise

        finally:
            self.manager.unload()

    def clone(self, segments, speakers, generated_dir):
        """Alias cho code cũ."""
        return self.process(segments, speakers, generated_dir)