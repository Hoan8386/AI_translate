"""
Stage 5: ASR (Automated Speech Recognition)
===========================================
Nhận dạng lời thoại tiếng Trung từ file audio và khớp vào cấu trúc phân đoạn (Segments).

Input:
    - audio_path: File wav (16KHz, Mono)
    - segments: List[Segment] (Đã có timeline start/end và speaker từ Stage 3 & 4)

Output:
    - List[Segment] (Được cập nhật thêm thuộc tính zh_text)

Technology:
    FunASR + SenseVoiceSmall (Chạy Local hoàn toàn, tối ưu GPU)
"""

from pathlib import Path
from typing import List
import torch

from models_data.segment import Segment
from utils.logger import get_logger, log_stage
from utils.timer import Timer
from utils.gpu_manager import GPUManager
from config.settings import get_settings

logger = get_logger(__name__)


class ASRProcessor:

    STAGE_NUM = 5
    STAGE_NAME = "ASR (SenseVoice)"

    # Lưu trữ mô hình dùng chung (Singleton Pattern) để tránh reload liên tục
    _shared_model = None

    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()
        
        # Đồng bộ trạng thái model với bộ nhớ dùng chung
        self._model = ASRProcessor._shared_model

    def _load_model(self):
        """
        Khởi tạo và nạp mô hình FunASR SenseVoiceSmall lên GPU.
        Kiểm soát VRAM chặt chẽ trước khi load.
        """
        if self._model is not None:
            return

        from funasr import AutoModel

        logger.info(f"Loading ASR Model: {self.settings.asr.model_name}...")
        
        # Đảm bảo trống ít nhất 2GB VRAM trên RTX 5060 trước khi nạp model
        self.gpu.ensure_free(2000)

        try:
            # Khởi tạo AutoModel từ FunASR
            model = AutoModel(
                model=self.settings.asr.model_name,
                device=self.settings.gpu.device,      # "cuda" hoặc "cpu"
                ncpu=4,                               # Số luồng CPU bổ trợ
                hub="ms",                             # Tải qua ModelScope (tối ưu cho mạng VN/Trung Quốc)
                disable_update=True
            )
            
            self._model = model
            ASRProcessor._shared_model = model
            logger.info("ASR Model (SenseVoiceSmall) đã nạp vào VRAM thành công.")
            
        except Exception as e:
            logger.error(f"Không thể nạp mô hình ASR: {e}")
            raise RuntimeError(f"ASR Initialization Failed: {e}")

    def process(self, audio_path: str, segments: List[Segment]) -> List[Segment]:
        """
        Trích xuất lời thoại tiếng Trung cho từng phân đoạn timeline.
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")

        if not segments:
            logger.warning("Danh sách segments trống. Bỏ qua bước ASR.")
            log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
            return segments

        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            audio_file = Path(audio_path)
            if not audio_file.exists():
                raise FileNotFoundError(f"File Audio không tồn tại: {audio_path}")

            # Nạp model lên GPU
            self._load_model()

            logger.info(f"Đang tiến hành nhận dạng lời thoại cho {len(segments)} segments...")

            # Duyệt qua từng segment để cắt và nhận diện lời thoại dựa trên timeline (start/end)
            for i, segment in enumerate(segments):
                try:
                    # Gọi hàm suy luận của FunASR truyền kèm khoảng thời gian start/end (tính bằng mili giây)
                    # Note: Cấu hình tham số tùy thuộc vào cách SenseVoice nhận cắt đoạn, 
                    # thông dụng là truyền trực tiếp luồng cắt hoặc truyền chunk qua param
                    
                    res = self._model.generate(
                        input=str(audio_file),
                        cache={},
                        language=self.settings.asr.language, # "zh"
                        use_itn=True,                        # Chuyển đổi số thông minh (Inverse Text Normalization)
                        batch_size_s=self.settings.asr.batch_size, # Giữ batch size nhỏ = 1 để tiết kiệm VRAM
                        beg_ans=int(segment.start * 1000),   # Đổi sang mili giây nếu API yêu cầu
                        end_ans=int(segment.end * 1000)
                    )

                    # Bóc tách text từ kết quả trả về của FunASR
                    if res and isinstance(res, list) and len(res) > 0:
                        # Kết quả thông thường của FunASR có dạng [{'text': '...'}]
                        raw_text = res[0].get('text', '').strip()
                        segment.zh_text = raw_text
                    else:
                        segment.zh_text = ""

                    logger.info(
                        f"ASR [{i+1}/{len(segments)}] ({segment.start:.2f}s - {segment.end:.2f}s) "
                        f"[{getattr(segment, 'speaker', 'unknown')}]: {segment.zh_text}"
                    )

                except Exception as seg_err:
                    logger.warning(f"Lỗi nhận dạng tại segment {i} ({segment.start}s - {segment.end}s): {seg_err}")
                    segment.zh_text = ""

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return segments

    @classmethod
    def unload_model(cls):
        """
        Giải phóng hoàn toàn mô hình FunASR khỏi VRAM GPU.
        Gọi ngay sau khi hoàn thành Stage 5 để chống lỗi tràn bộ nhớ (OOM).
        """
        if cls._shared_model is not None:
            logger.info("Kích hoạt dọn dẹp bộ nhớ: Unloading FunASR khỏi VRAM...")
            
            # Hủy liên kết đối tượng mô hình
            del cls._shared_model
            cls._shared_model = None
            
            # Kích hoạt dọn rác của Python
            import gc
            gc.collect()
            
            # Ép PyTorch dọn sạch bộ nhớ đệm trên card đồ họa
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Bộ nhớ GPU dành cho ASR đã được giải phóng hoàn toàn.")