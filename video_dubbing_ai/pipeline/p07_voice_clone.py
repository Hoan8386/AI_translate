"""
Stage 7: Voice Cloning (Local Fish Speech)
============================================
Sinh giọng Việt giữ đặc trưng từng người.
Load model Fish Speech trực tiếp (không cần server).

Bước 7.1: Load Fish Speech model (LLAMA + Decoder)
Bước 7.2: Sinh tiếng Việt với giọng clone

Input:  speaker_1_reference.wav + "Xin chào mọi người"
Output: speaker_1_vi.wav

Model: Fish Speech 1.5 (Local Inference)
"""

import gc
import os
import sys
import subprocess
import queue
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from typing import List, Dict, Optional

from models_data.segment import Segment
from models_data.speaker import Speaker
from utils.logger import get_logger, log_stage
from utils.timer import Timer
from utils.gpu_manager import GPUManager
from config.settings import get_settings

logger = get_logger(__name__)


class VoiceCloner:
    """
    Clone giọng nói với Fish Speech (Local Inference).
    
    Quy trình:
    1. Load LLAMA model + Decoder model lên GPU
    2. Đọc reference audio của speaker → encode VQ tokens
    3. Gọi TTSInferenceEngine.inference() trực tiếp
    4. Decode audio → lưu WAV
    
    Fallback: Nếu load model thất bại → dùng edge-tts
    """
    
    STAGE_NUM = 7
    STAGE_NAME = "Voice Cloning (Fish Speech)"
    
    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()
        self._model_loaded = False
        self._tts_engine = None
        self._decoder_model = None
        self._llama_queue = None
        
        # Fish Speech paths
        self._fish_speech_dir = (
            self.settings.project_root / "third_party" / "fish-speech"
        )
    
    def _load_model(self) -> bool:
        """
        Load Fish Speech models (LLAMA + Decoder) lên GPU.
        
        Returns:
            True nếu load thành công, False nếu thất bại
        """
        if self._model_loaded and self._tts_engine is not None:
            logger.info("✅ Fish Speech model đã được load sẵn")
            return True
        
        try:
            # Thêm fish-speech vào sys.path để import được
            fish_speech_path = str(self._fish_speech_dir)
            if fish_speech_path not in sys.path:
                sys.path.insert(0, fish_speech_path)
            
            # Import Fish Speech modules
            from fish_speech.models.text2semantic.inference import (
                launch_thread_safe_queue,
            )
            from fish_speech.models.dac.inference import (
                load_model as load_decoder_model,
            )
            from fish_speech.inference_engine import TTSInferenceEngine
            
            cfg = self.settings.voice_clone
            
            # Resolve checkpoint paths
            llama_path = self._fish_speech_dir / cfg.llama_checkpoint_path
            decoder_path = self._fish_speech_dir / cfg.decoder_checkpoint_path
            
            # Kiểm tra checkpoints có tồn tại không
            if not llama_path.exists():
                logger.error(
                    f"❌ Không tìm thấy LLAMA checkpoint: {llama_path}\n"
                    f"  Hãy tải model về:\n"
                    f"  huggingface-cli download fishaudio/fish-speech-1.5 "
                    f"--local-dir {self._fish_speech_dir / 'checkpoints/s2-pro'}"
                )
                return False
            
            if not decoder_path.exists():
                logger.error(
                    f"❌ Không tìm thấy Decoder checkpoint: {decoder_path}"
                )
                return False
            
            # Xác định device và precision
            device = cfg.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                logger.warning("CUDA không khả dụng, chuyển sang CPU")
            
            precision = torch.half if cfg.half else torch.bfloat16
            
            logger.info(f"  Loading LLAMA model từ {llama_path}...")
            self._llama_queue = launch_thread_safe_queue(
                checkpoint_path=str(llama_path),
                device=device,
                precision=precision,
                compile=False,
            )
            logger.info("  ✅ LLAMA model loaded")
            
            logger.info(f"  Loading Decoder model từ {decoder_path}...")
            self._decoder_model = load_decoder_model(
                config_name=cfg.decoder_config_name,
                checkpoint_path=str(decoder_path),
                device=device,
            )
            logger.info("  ✅ Decoder model loaded")
            
            # Tạo TTS inference engine
            self._tts_engine = TTSInferenceEngine(
                llama_queue=self._llama_queue,
                decoder_model=self._decoder_model,
                precision=precision,
                compile=False,
            )
            
            self._model_loaded = True
            logger.info("✅ Fish Speech model sẵn sàng (local inference)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi load Fish Speech model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._model_loaded = False
            return False
    
    def _unload_model(self):
        """Giải phóng VRAM sau khi xong."""
        try:
            if self._llama_queue is not None:
                self._llama_queue.put(None)  # Signal worker thread to stop
                self._llama_queue = None
            
            if self._decoder_model is not None:
                del self._decoder_model
                self._decoder_model = None
            
            if self._tts_engine is not None:
                del self._tts_engine
                self._tts_engine = None
            
            self._model_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("🗑 Fish Speech model đã được giải phóng khỏi VRAM")
        except Exception as e:
            logger.warning(f"Lỗi khi unload model: {e}")
    
    def _generate_with_fish_speech(
        self,
        text: str,
        reference_audio: Optional[str],
        output_path: str,
    ) -> bool:
        """
        Sinh audio bằng Fish Speech local inference.
        
        Args:
            text: Nội dung tiếng Việt cần sinh
            reference_audio: Đường dẫn reference audio (WAV) để clone giọng
            output_path: Đường dẫn lưu output
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        if self._tts_engine is None:
            logger.error("Fish Speech engine chưa được load")
            return False
        
        try:
            # Import schema
            fish_speech_path = str(self._fish_speech_dir)
            if fish_speech_path not in sys.path:
                sys.path.insert(0, fish_speech_path)
            
            from fish_speech.utils.schema import (
                ServeTTSRequest,
                ServeReferenceAudio,
            )
            
            # Chuẩn bị references
            references = []
            if reference_audio and Path(reference_audio).exists():
                with open(reference_audio, "rb") as f:
                    audio_bytes = f.read()
                
                # Reference audio cần kèm text mô tả (có thể để trống)
                references.append(
                    ServeReferenceAudio(
                        audio=audio_bytes,
                        text="",  # Không cần text mô tả cho reference
                    )
                )
            
            # Tạo TTS request
            request = ServeTTSRequest(
                text=text,
                references=references,
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.2,
                temperature=0.7,
                format="wav",
                streaming=False,
            )
            
            # Chạy inference
            final_audio = None
            sample_rate = None
            
            for result in self._tts_engine.inference(request):
                if result.code == "error":
                    logger.error(f"Fish Speech inference lỗi: {result.error}")
                    return False
                elif result.code == "final":
                    if isinstance(result.audio, tuple):
                        sample_rate, final_audio = result.audio
            
            if final_audio is None:
                logger.error("Không nhận được audio output từ Fish Speech")
                return False
            
            # Lưu audio ra file WAV
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            sf.write(output_path, final_audio, sample_rate)
            
            # Kiểm tra file hợp lệ
            file_size = os.path.getsize(output_path)
            if file_size < 1000:
                logger.warning(
                    f"File output quá nhỏ ({file_size} bytes), "
                    f"có thể không hợp lệ"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi Fish Speech inference: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _fallback_tts(self, text: str, output_path: str) -> bool:
        """
        Fallback TTS khi Fish Speech không khả dụng.
        Sử dụng edge-tts (miễn phí, chất lượng tốt nhưng không clone giọng).
        
        Args:
            text: Nội dung cần sinh audio
            output_path: Đường dẫn lưu output
            
        Returns:
            True nếu thành công
        """
        try:
            logger.info("  Dùng edge-tts fallback (không clone giọng)...")
            
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                "edge-tts",
                "--voice", "vi-VN-HoaiMyNeural",
                "--text", text,
                "--write-media", output_path,
            ]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                timeout=30,
                text=True,
            )
            
            if result.returncode != 0:
                logger.error(f"edge-tts lỗi: {result.stderr}")
                return False
            
            return Path(output_path).exists()
            
        except FileNotFoundError:
            logger.error(
                "edge-tts chưa được cài đặt. "
                "Chạy: pip install edge-tts"
            )
            return False
        except subprocess.TimeoutExpired:
            logger.error("edge-tts timeout")
            return False
        except Exception as e:
            logger.error(f"Fallback TTS lỗi: {e}")
            return False
    
    def process(
        self,
        segments: List[Segment],
        speakers: Dict[str, Speaker],
        output_dir: str,
    ) -> List[Segment]:
        """
        Voice clone cho tất cả segments.
        
        Args:
            segments: Danh sách Segment đã có vi_text
            speakers: Dict speakers với reference_audio
            output_dir: Thư mục lưu generated audio
            
        Returns:
            Danh sách Segment đã có generated_audio
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")
        
        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Bước 7.1: Load Fish Speech model
            logger.info("Bước 7.1: Load Fish Speech model (local)...")
            model_available = self._load_model()
            
            if not model_available:
                logger.warning(
                    "⚠ Fish Speech model không khả dụng!\n"
                    "  → Sẽ dùng edge-tts làm fallback (không clone giọng)\n"
                    "  → Để có voice cloning, hãy tải model:\n"
                    "     huggingface-cli download fishaudio/fish-speech-1.5 "
                    f"--local-dir {self._fish_speech_dir / 'checkpoints/s2-pro'}"
                )
            
            # Bước 7.2: Sinh giọng Việt cho từng segment
            logger.info("Bước 7.2: Sinh giọng Việt với Fish Speech...")
            
            cloned_count = 0
            fallback_count = 0
            error_count = 0
            
            for i, segment in enumerate(segments):
                if not segment.vi_text.strip():
                    logger.debug(
                        f"  Segment {segment.id}: không có vi_text, bỏ qua"
                    )
                    continue
                
                logger.info(
                    f"  Voice clone [{i+1}/{len(segments)}]: "
                    f"Segment {segment.id} ({segment.speaker}) "
                    f"\"{segment.vi_text[:30]}...\""
                )
                
                # Xác định reference audio cho speaker
                reference_audio = None
                if segment.speaker in speakers:
                    spk = speakers[segment.speaker]
                    if spk.reference_audio and Path(spk.reference_audio).exists():
                        reference_audio = spk.reference_audio
                
                # Tạo đường dẫn output
                generated_filename = (
                    f"{segment.speaker}_seg{segment.id:03d}_vi.wav"
                )
                generated_path = str(output_path / generated_filename)
                
                success = False
                
                # Thử Fish Speech trước
                if model_available:
                    success = self._generate_with_fish_speech(
                        text=segment.vi_text,
                        reference_audio=reference_audio,
                        output_path=generated_path,
                    )
                    if success:
                        cloned_count += 1
                        clone_type = (
                            "voice cloned" if reference_audio 
                            else "default voice"
                        )
                        logger.info(
                            f"    → {generated_filename} ({clone_type})"
                        )
                
                # Fallback nếu Fish Speech thất bại
                if not success:
                    success = self._fallback_tts(
                        segment.vi_text, generated_path
                    )
                    if success:
                        fallback_count += 1
                        logger.info(
                            f"    → {generated_filename} (edge-tts fallback)"
                        )
                    else:
                        error_count += 1
                        logger.error(
                            f"    ✗ Không thể sinh audio cho "
                            f"segment {segment.id}"
                        )
                
                if success:
                    segment.generated_audio = generated_path
            
            # Giải phóng VRAM sau khi xong
            if model_available:
                self._unload_model()
            
            # Thống kê
            total = cloned_count + fallback_count
            logger.info(
                f"\nVoice cloning hoàn thành:\n"
                f"  ✓ Fish Speech (local): {cloned_count} segments\n"
                f"  ⚡ Fallback (edge-tts): {fallback_count} segments\n"
                f"  ✗ Lỗi: {error_count} segments\n"
                f"  Tổng: {total}/{len(segments)} segments"
            )
        
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return segments
