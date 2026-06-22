"""
Stage 7: Voice Cloning
========================
Phần QUAN TRỌNG NHẤT - Sinh giọng Việt giữ đặc trưng từng người.

Bước 7.1: Kiểm tra Fish Speech server
Bước 7.2: Sinh tiếng Việt với giọng clone qua Fish Speech API

Input:  speaker_1_reference.wav + "Xin chào mọi người"
Output: speaker_1_vi.wav

Model: Fish Speech (Local HTTP API Server)
"""

import os
import subprocess
import requests
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
    Clone giọng nói với Fish Speech (Local API Server).
    
    Quy trình:
    1. Kiểm tra Fish Speech server đang chạy (GET /v1/health)
    2. Đọc reference audio của speaker
    3. Gửi POST /v1/tts với text + reference audio
    4. Nhận audio output (WAV) và lưu file
    
    Fallback: Nếu Fish Speech server không chạy → dùng edge-tts
    """
    
    STAGE_NUM = 7
    STAGE_NAME = "Voice Cloning (Fish Speech)"
    
    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()
        self._server_url = self.settings.voice_clone.server_url
        self._timeout = self.settings.voice_clone.request_timeout
        self._server_available = False
    
    def _check_server(self) -> bool:
        """
        Kiểm tra Fish Speech server có đang chạy không.
        
        Returns:
            True nếu server sẵn sàng, False nếu không
        """
        try:
            url = f"{self._server_url}/v1/health"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    logger.info(f"✅ Fish Speech server sẵn sàng: {self._server_url}")
                    return True
            logger.warning(
                f"Fish Speech server phản hồi không hợp lệ: "
                f"status={response.status_code}"
            )
            return False
        except requests.ConnectionError:
            logger.warning(
                f"❌ Không thể kết nối Fish Speech server tại {self._server_url}. "
                f"Hãy chắc chắn server đang chạy:\n"
                f"  cd third_party/fish-speech\n"
                f"  python tools/api_server.py --listen 0.0.0.0:8080"
            )
            return False
        except Exception as e:
            logger.warning(f"Lỗi kiểm tra Fish Speech server: {e}")
            return False
    
    def _generate_with_fish_speech(
        self,
        text: str,
        reference_audio: Optional[str],
        output_path: str,
    ) -> bool:
        """
        Sinh audio bằng Fish Speech API.
        
        Gửi POST /v1/tts với text + reference audio (nếu có).
        Fish Speech sẽ tự động clone giọng từ reference audio.
        
        Args:
            text: Nội dung tiếng Việt cần sinh
            reference_audio: Đường dẫn reference audio (WAV) để clone giọng
            output_path: Đường dẫn lưu output
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        url = f"{self._server_url}/v1/tts"
        
        try:
            # Chuẩn bị request body (multipart/form-data)
            # Fish Speech API nhận JSON body với references
            payload = {
                "text": text,
                "format": "wav",
                "streaming": False,
            }
            
            files = {}
            
            if reference_audio and Path(reference_audio).exists():
                # Gửi reference audio để voice cloning
                with open(reference_audio, "rb") as ref_f:
                    ref_data = ref_f.read()
                
                # Fish Speech API: gửi qua multipart form
                files = {
                    "reference_audio": (
                        Path(reference_audio).name,
                        ref_data,
                        "audio/wav",
                    )
                }
                
                # Gửi request với reference audio
                response = requests.post(
                    url,
                    data=payload,
                    files=files,
                    timeout=self._timeout,
                )
            else:
                # Không có reference → TTS thông thường (random voice)
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self._timeout,
                )
            
            if response.status_code == 200:
                # Lưu audio output
                output_dir = Path(output_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                # Kiểm tra file hợp lệ
                file_size = os.path.getsize(output_path)
                if file_size < 1000:
                    logger.warning(
                        f"File output quá nhỏ ({file_size} bytes), "
                        f"có thể không hợp lệ"
                    )
                    return False
                
                return True
            else:
                error_msg = response.text[:200] if response.text else "Unknown error"
                logger.error(
                    f"Fish Speech API lỗi: status={response.status_code}, "
                    f"error={error_msg}"
                )
                return False
                
        except requests.Timeout:
            logger.error(
                f"Fish Speech API timeout ({self._timeout}s) cho text: "
                f"\"{text[:50]}...\""
            )
            return False
        except Exception as e:
            logger.error(f"Lỗi gọi Fish Speech API: {e}")
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
            
            # Bước 7.1: Kiểm tra Fish Speech server
            logger.info("Bước 7.1: Kiểm tra Fish Speech server...")
            self._server_available = self._check_server()
            
            if not self._server_available:
                logger.warning(
                    "⚠ Fish Speech server không khả dụng!\n"
                    "  → Sẽ dùng edge-tts làm fallback (không clone giọng)\n"
                    "  → Để có voice cloning, hãy khởi động Fish Speech server:\n"
                    "     cd third_party/fish-speech\n"
                    "     python tools/api_server.py --listen 0.0.0.0:8080"
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
                if self._server_available:
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
            
            # Thống kê
            total = cloned_count + fallback_count
            logger.info(
                f"\nVoice cloning hoàn thành:\n"
                f"  ✓ Fish Speech: {cloned_count} segments\n"
                f"  ⚡ Fallback (edge-tts): {fallback_count} segments\n"
                f"  ✗ Lỗi: {error_count} segments\n"
                f"  Tổng: {total}/{len(segments)} segments"
            )
        
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return segments
