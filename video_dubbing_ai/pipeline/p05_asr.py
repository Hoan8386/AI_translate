import os
import gc
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from utils.logger import get_logger

# Khởi tạo logger chuẩn theo cấu trúc gốc của dự án
logger = get_logger("dubbing")

class ASRModelManager:
    """Singleton Manager giúp nạp Whisper trực tiếp vào VRAM và quản lý giải phóng bộ nhớ"""
    _instance = None
    _model = None
    _processor = None
    _model_id = "openai/whisper-large-v3"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ASRModelManager, cls).__new__(cls)
        return cls._instance

    def get_model_and_processor(self):
        if self._model is None or self._processor is None:
            logger.info(f"[*] Khởi tạo và nạp thẳng ASR Model ({self._model_id}) vào VRAM...")
            try:
                # Nạp processor xử lý text/tokenizer
                self._processor = AutoProcessor.from_pretrained(self._model_id)

                # Nạp ma trận trọng số thẳng vào GPU, bỏ qua RAM nền ảo
                self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self._model_id,
                    torch_dtype=torch.float16,     # Giảm 50% dung lượng VRAM
                    low_cpu_mem_usage=True,        # Không tạo bản sao rỗng trên System RAM
                    device_map="cuda"              # Đọc file tới đâu, nhồi thẳng vào VRAM tới đó
                )
                logger.info("[+] Nạp ASR Model thành công và an toàn trên GPU.")
            except Exception as e:
                logger.error(f"[-] Lỗi khi nạp ASR Model trực tiếp: {str(e)}")
                raise e
        else:
            logger.debug("[+] Sử dụng ASR Model từ VRAM Cache.")
            
        return self._model, self._processor

    def unload_model(self):
        """Xóa sạch Whisper khỏi cả bộ nhớ RAM và VRAM ngay sau khi xử lý xong"""
        if self._model is not None or self._processor is not None:
            logger.info("[*] Đang tiến hành giải phóng (Unload) ASR Model...")
            
            # Cắt liên kết tham chiếu Object
            self._model = None
            self._processor = None
            
            # Thu gom rác trên RAM và dọn cache VRAM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("[+] Đã dọn sạch bộ nhớ của ASR Model.")

class ASRProcessor:
    """Class bọc ngoài chuẩn để khớp với pipeline/__init__.py và asr_service.py của dự án"""
    
    def __init__(self):
        # Khởi tạo manager tối ưu bộ nhớ
        self.manager = ASRModelManager()

    def process(self, audio_path, segments):
        """
        HÀM CHÍNH: Được gọi từ services/asr_service.py bằng lệnh self.asr.process()
        """
        if not os.path.exists(audio_path):
            logger.error(f"File audio không tồn tại: {audio_path}")
            return segments

        logger.info(f"[ASR] Bắt đầu xử lý nhận diện cho file: {audio_path}")
        
        # Gọi nạp model trực tiếp vào GPU an toàn
        model, processor = self.manager.get_model_and_processor()

        try:
            # -----------------------------------------------------------------
            # KHÚC NÀY: Dán logic chạy inference thực tế sinh ra text của bạn vào đây.
            # Sử dụng object `model` và `processor` lấy từ manager ở trên.
            # Tuyệt đối KHÔNG dùng lệnh: model.to("cuda") nữa.
            # -----------------------------------------------------------------
            
            # Giả lập cập nhật thuộc tính text cho các segments đầu ra
            for seg in segments:
                # Kiểm tra xem seg là object hay dict để cập nhật text tiếng Trung (zh_text)
                if isinstance(seg, dict):
                    if not seg.get('zh_text'):
                        seg['zh_text'] = "Dữ liệu tiếng Trung nhận diện mẫu"
                else:
                    if not hasattr(seg, 'zh_text') or not seg.zh_text:
                        seg.zh_text = "Dữ liệu tiếng Trung nhận diện mẫu"
            
            logger.info("[ASR] Hoàn thành nhận diện giọng nói.")
            return segments

        except Exception as e:
            logger.error(f"[ASR] Lỗi trong quá trình inference: {str(e)}")
            return segments
            
        finally:
            # Chạy xong (hoặc lỗi) đều giải phóng ngay lập tức để lấy lại không gian VRAM
            self.manager.unload_model()

    def transcribe(self, audio_path, segments):
        """Hàm dự phòng nếu file pipeline/__init__.py gọi qua tên transcribe"""
        return self.process(audio_path, segments)