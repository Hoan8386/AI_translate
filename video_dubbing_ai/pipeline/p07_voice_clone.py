"""
Stage 7: Voice Cloning
========================
Phần QUAN TRỌNG NHẤT - Sinh giọng Việt giữ đặc trưng từng người.

Bước 7.1: Tạo hồ sơ giọng nói (reference audio)
Bước 7.2: Sinh tiếng Việt với giọng clone

Input:  speaker_1_reference.wav + "Xin chào mọi người"
Output: speaker_1_vi.wav

Model: OpenVoice V2
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict

from models_data.segment import Segment
from models_data.speaker import Speaker
from utils.logger import get_logger, log_stage
from utils.timer import Timer
from utils.gpu_manager import GPUManager
from config.settings import get_settings

logger = get_logger(__name__)


class VoiceCloner:
    """
    Clone giọng nói với OpenVoice V2.
    
    Quy trình:
    1. Load reference audio của speaker
    2. Extract speaker embedding (tone color)  
    3. Sinh giọng Việt với TTS
    4. Apply tone color conversion để giữ đặc trưng giọng
    """
    
    STAGE_NUM = 7
    STAGE_NAME = "Voice Cloning (OpenVoice V2)"
    
    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()
        self._tts_model = None
        self._tone_color_converter = None
    
    def _add_openvoice_to_path(self):
        """Thêm OpenVoice vào Python path"""
        openvoice_path = str(self.settings.third_party_dir / "OpenVoice")
        if openvoice_path not in sys.path:
            sys.path.insert(0, openvoice_path)
    
    def _load_models(self):
        """Load OpenVoice V2 models lên GPU"""
        self._add_openvoice_to_path()
        
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
        
        logger.info("Loading OpenVoice V2 models...")
        
        device = self.gpu.device
        
        # Load Tone Color Converter
        ckpt_converter = os.path.join(
            str(self.settings.third_party_dir / "OpenVoice"),
            "checkpoints_v2", "converter"
        )
        
        # Nếu không có checkpoint local, thử dùng từ cache
        if not os.path.exists(ckpt_converter):
            # Tải từ Hugging Face
            logger.info("Downloading OpenVoice V2 checkpoints...")
            ckpt_converter = "myshell-ai/OpenVoiceV2"
        
        self._tone_color_converter = ToneColorConverter(
            f"{ckpt_converter}/config.json",
            device=device
        )
        self._tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
        
        logger.info("OpenVoice V2 models loaded")
        self.gpu.log_vram("After OpenVoice load")
    
    def _unload_models(self):
        """Unload tất cả OpenVoice models"""
        if self._tone_color_converter is not None:
            del self._tone_color_converter
            self._tone_color_converter = None
        
        if self._tts_model is not None:
            del self._tts_model
            self._tts_model = None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("OpenVoice models unloaded")
    
    def _extract_speaker_embedding(self, reference_audio: str):
        """
        Extract tone color embedding từ reference audio.
        
        Args:
            reference_audio: Đường dẫn reference audio
            
        Returns:
            Speaker embedding (tone color)
        """
        self._add_openvoice_to_path()
        from openvoice import se_extractor
        
        target_se, audio_name = se_extractor.get_se(
            reference_audio,
            self._tone_color_converter,
            vad=True,
        )
        
        return target_se
    
    def _generate_base_tts(self, text: str, output_path: str):
        """
        Sinh audio base bằng TTS (trước khi apply voice cloning).
        
        Dùng MeloTTS (đi kèm OpenVoice V2) cho tiếng Việt.
        """
        try:
            from melo.api import TTS
            
            if self._tts_model is None:
                # Load MeloTTS
                device = self.gpu.device
                self._tts_model = TTS(language="VI", device=device)
                logger.info("MeloTTS loaded for Vietnamese")
            
            # Sinh audio
            speaker_ids = self._tts_model.hps.data.spk2id
            # Lấy speaker đầu tiên
            default_speaker = list(speaker_ids.keys())[0]
            
            self._tts_model.tts_to_file(
                text,
                speaker_ids[default_speaker],
                output_path,
                speed=1.0,
            )
            
        except ImportError:
            # Fallback: dùng edge-tts hoặc gtts
            logger.warning("MeloTTS không khả dụng, dùng fallback TTS")
            self._fallback_tts(text, output_path)
    
    def _fallback_tts(self, text: str, output_path: str):
        """Fallback TTS nếu MeloTTS không khả dụng"""
        try:
            import subprocess
            # Dùng edge-tts (free, chất lượng tốt)
            cmd = [
                "edge-tts",
                "--voice", "vi-VN-HoaiMyNeural",
                "--text", text,
                "--write-media", output_path,
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
        except Exception as e:
            logger.error(f"Fallback TTS cũng lỗi: {e}")
            raise RuntimeError(f"Không thể sinh TTS cho: {text}")
    
    def _apply_voice_conversion(
        self,
        source_audio: str,
        target_se,
        output_path: str,
    ):
        """
        Apply tone color conversion.
        
        Chuyển giọng base TTS → giọng target speaker.
        """
        self._add_openvoice_to_path()
        from openvoice import se_extractor
        
        # Extract source speaker embedding
        source_se, _ = se_extractor.get_se(
            source_audio,
            self._tone_color_converter,
            vad=True,
        )
        
        # Apply conversion
        self._tone_color_converter.convert(
            audio_src_path=source_audio,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
        )
    
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
            
            # Thư mục tạm cho base TTS
            base_tts_dir = output_path.parent / "base_tts"
            base_tts_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Load models
                self.gpu.ensure_free(3000)
                self._load_models()
                
                # Bước 7.1: Extract speaker embeddings
                logger.info("Bước 7.1: Extracting speaker embeddings...")
                speaker_embeddings = {}
                
                for speaker_id, speaker in speakers.items():
                    if speaker.reference_audio and Path(speaker.reference_audio).exists():
                        logger.info(f"  Extracting embedding: {speaker_id}")
                        se = self._extract_speaker_embedding(speaker.reference_audio)
                        speaker_embeddings[speaker_id] = se
                    else:
                        logger.warning(
                            f"  {speaker_id}: không có reference audio, "
                            f"sẽ dùng giọng mặc định"
                        )
                
                # Bước 7.2: Sinh giọng Việt cho từng segment
                logger.info("Bước 7.2: Generating Vietnamese voice...")
                
                for i, segment in enumerate(segments):
                    if not segment.vi_text.strip():
                        logger.debug(f"  Segment {segment.id}: không có vi_text, bỏ qua")
                        continue
                    
                    logger.info(
                        f"  Voice clone [{i+1}/{len(segments)}]: "
                        f"Segment {segment.id} ({segment.speaker}) "
                        f"\"{segment.vi_text[:30]}...\""
                    )
                    
                    try:
                        # Sinh base TTS
                        base_audio = str(base_tts_dir / f"base_{segment.id:03d}.wav")
                        self._generate_base_tts(segment.vi_text, base_audio)
                        
                        # Apply voice conversion nếu có embedding
                        generated_filename = f"{segment.speaker}_seg{segment.id:03d}_vi.wav"
                        generated_path = str(output_path / generated_filename)
                        
                        if segment.speaker in speaker_embeddings:
                            self._apply_voice_conversion(
                                base_audio,
                                speaker_embeddings[segment.speaker],
                                generated_path,
                            )
                            logger.info(f"    → {generated_filename} (voice cloned)")
                        else:
                            # Không có embedding → copy base TTS
                            import shutil
                            shutil.copy2(base_audio, generated_path)
                            logger.info(f"    → {generated_filename} (default voice)")
                        
                        segment.generated_audio = generated_path
                        
                    except Exception as e:
                        logger.error(
                            f"    Voice clone error cho segment {segment.id}: {e}"
                        )
                
                # Thống kê
                cloned = sum(1 for s in segments if s.generated_audio)
                logger.info(
                    f"Voice cloning hoàn thành: {cloned}/{len(segments)} segments"
                )
                
            finally:
                # LUÔN unload models
                self._unload_models()
        
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return segments
