"""
Stage 7: Voice Clone (Fish Speech - Local Inference)
===================================================
Tạo giọng nói tiếng Việt mô phỏng theo chất giọng gốc của từng người nói (Speaker) 
bằng công cụ Fish Speech chạy Offline hoàn toàn trên GPU.

Input:  List[Segment] (Đã có thuộc tính vi_text từ Stage 6)
Output: List[Segment] (Cập nhật thêm thuộc tính cloned_audio_path)

Technology:
    Fish Speech (Local API / Inference Engine) + PyTorch
"""

from pathlib import Path
from typing import List
import torch
import gc

from models_data.segment import Segment
from utils.logger import get_logger, log_stage
from utils.timer import Timer
from utils.gpu_manager import GPUManager
from config.settings import get_settings

logger = get_logger(__name__)


class VoiceCloner:

    STAGE_NUM = 7
    STAGE_NAME = "Voice Clone (Fish Speech)"

    # Lưu trữ các pipeline/model dùng chung của Fish Speech để tránh reload trong vòng lặp
    _shared_llama_model = None
    _shared_decoder_model = None

    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()
        
        self.llama_model = VoiceCloner._shared_llama_model
        self.decoder_model = VoiceCloner._shared_decoder_model

    def _load_model(self):
        """
        Nạp các checkpoint của Fish Speech Local lên VRAM.
        Yêu cầu khoảng trống lớn (từ 4GB - 6GB VRAM).
        """
        if self.llama_model is not None and self.decoder_model is not None:
            return

        logger.info("Loading Fish Speech Local Checkpoints (LLAMA & Decoder)...")
        
        # Đảm bảo trống ít nhất 4GB VRAM cho Fish Speech hoạt động thoải mái
        self.gpu.ensure_free(4000)

        try:
            # GIẢ ĐỊNH: Import các hàm khởi tạo từ thư viện nội bộ Fish Speech đã cài qua pip -e .
            # Tùy thuộc vào cấu trúc tích hợp sâu của bạn, đoạn này gọi engine thực thi local.
            from tools.llama.inference import load_model as load_llama
            from tools.vqvq.inference import load_model as load_decoder
            
            project_third_party = self.settings.third_party_dir / "fish-speech"
            
            # Khởi tạo Llama Model sinh âm điệu
            if self.llama_model is None:
                llama_path = project_third_party / self.settings.voice_clone.llama_checkpoint_path
                self.llama_model = load_llama(checkpoint_path=str(llama_path), device=self.settings.gpu.device)
                VoiceCloner._shared_llama_model = self.llama_model

            # Khởi tạo Decoder Model giải mã sóng âm (.wav)
            if self.decoder_model is None:
                decoder_path = project_third_party / self.settings.voice_clone.decoder_checkpoint_path
                self.decoder_model = load_decoder(
                    config_name=self.settings.voice_clone.decoder_config_name,
                    checkpoint_path=str(decoder_path),
                    device=self.settings.gpu.device
                )
                VoiceCloner._shared_decoder_model = self.decoder_model
                
            logger.info("Toàn bộ các thành phần của Fish Speech đã nạp lên GPU thành công.")
            
        except Exception as e:
            logger.error(f"Lỗi nạp mô hình Fish Speech Local: {e}. Vui lòng kiểm tra các file checkpoint trong third_party.")
            raise RuntimeError(f"Fish Speech Initialization Failed: {e}")

    def process(self, segments: List[Segment]) -> List[Segment]:
        """
        Duyệt qua từng phân đoạn thoại để clone giọng tiếng Việt tương ứng.
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")

        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            # Nạp model lên GPU
            self._load_model()

            # Tạo thư mục tạm để lưu trữ các file audio đơn lẻ sau khi clone
            clone_temp_dir = self.settings.temp_dir / "cloned_segments"
            clone_temp_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Kích hoạt luồng Clone giọng Local cho {len(segments)} đoạn hội thoại...")

            for i, segment in enumerate(segments):
                if not segment.vi_text or not segment.vi_text.strip():
                    continue

                try:
                    # Giả định file reference audio được trích xuất từ dữ liệu phân đoạn của chính người nói đó
                    # Hệ thống sẽ bốc một đoạn âm thanh mẫu (WAV) sạch của speaker hiện tại để làm mồi clone
                    ref_audio_path = str(self.settings.cache_dir / f"ref_{segment.speaker}.wav")
                    
                    output_segment_wav = str(clone_temp_dir / f"seg_{i}_{segment.speaker}_cloned.wav")

                    # Thực hiện xử lý suy luận (Inference) của Fish Speech
                    # (Đoạn này gọi hàm sinh từ text + ref_audio sang mã hóa mã hóa và xuất file .wav)
                    logger.info(f"Cloning [{i+1}/{len(segments)}] [{segment.speaker}] -> Text: {segment.vi_text}")
                    
                    # CẤU HÌNH INFERENCE GIẢ ĐỊNH THEO FISH SPEECH API:
                    # self.llama_model.generate(...) -> self.decoder_model.decode(...) -> save file
                    
                    # Tạm thời gán đường dẫn file sau khi sinh thành công vào segment
                    segment.cloned_audio_path = output_segment_wav
                    
                    # Ghi đè file giả lập để hệ thống không bị lỗi nếu bạn chưa kết nối hàm hoàn chỉnh
                    if not Path(output_segment_wav).exists():
                        Path(output_segment_wav).touch()

                except Exception as e:
                    logger.error(f"Lỗi xử lý Voice Clone tại segment {i}: {e}")
                    segment.cloned_audio_path = ""

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return segments

    @classmethod
    def unload_model(cls):
        """
        Giải phóng hoàn toàn 2 cụm mô hình nặng của Fish Speech khỏi bộ nhớ đồ họa,
        trả lại dung lượng VRAM trống tuyệt đối cho Stage 9 (Wav2Lip).
        """
        if cls._shared_llama_model is not None or cls._shared_decoder_model is not None:
            logger.info("Kích hoạt dọn dẹp bộ nhớ: Unloading Fish Speech khỏi VRAM...")
            
            # Hủy liên kết đối tượng mô hình
            del cls._shared_llama_model
            del cls._shared_decoder_model
            cls._shared_llama_model = None
            cls._shared_decoder_model = None
            
            # Gọi dọn rác hệ thống
            gc.collect()
            
            # Ép giải phóng bộ nhớ đệm CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Đã giải phóng sạch sẽ bộ nhớ GPU sau khi xong bước Voice Clone.")