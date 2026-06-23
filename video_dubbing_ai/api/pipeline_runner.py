"""
Pipeline Runner - Tối ưu hóa VRAM cho RTX 5060 8GB
==================================================
Bộ điều phối trung tâm quản lý vòng đời của các mô hình AI trên GPU.
Đảm bảo tuân thủ nguyên tắc: CHỈ 1 MODEL TRÊN VRAM TẠI 1 THỜI ĐIỂM.

Technology:
    Python 3.10 + PyTorch (CUDA 12.1)
"""

from pathlib import Path
import torch
import gc

from utils.logger import get_logger, log_stage
from utils.timer import Timer
from config.settings import get_settings

# Import các Stage từ các package trong hệ thống của bạn
from pipeline.p01_video_processor import VideoProcessor
from pipeline.p02_audio_extractor import AudioExtractor
from pipeline.p04_segment_creator import SegmentCreator
from pipeline.p03_speaker_detector import SpeakerDetector
from pipeline.p05_asr import ASRProcessor
from pipeline.p06_translation import Translator
from pipeline.p07_voice_clone import VoiceCloner
from pipeline.p08_audio_alignment import AudioAligner
from pipeline.p09_lipsync import LipSyncProcessor
from pipeline.p10_renderer import Renderer

logger = get_logger(__name__)


class PipelineRunner:
    """
    Quản lý và thực thi tuần tự 10 bước của hệ thống Auto Dubbing,
    chủ động giải phóng VRAM ngay sau khi kết thúc từng bước AI.
    """

    def __init__(self):
        self.settings = get_settings()

    def run(self, input_video_path: str) -> str:
        """
        Kích hoạt luồng chạy toàn bộ Pipeline xử lý video.
        
        Args:
            input_video_path: Đường dẫn tới file video gốc (.mp4)
            
        Returns:
            Đường dẫn tới video output đã được dubbing và render hoàn chỉnh
        """
        logger.info("=======================================================")
        logger.info("   BẮT ĐẦU HỆ THỐNG DUBBING AI CHUYÊN NGHIỆP (LOCAL)   ")
        logger.info("=======================================================")

        video_p = Path(input_video_path)
        if not video_p.exists():
            raise FileNotFoundError(f"Video đầu vào không tồn tại: {input_video_path}")

        # Khởi tạo các biến lưu trữ dữ liệu trung gian qua các Stage
        normalized_video = None
        extracted_audio = None
        speaker_segments = None
        structured_segments = None
        asr_segments = None
        translated_segments = None
        cloned_audio_segments = None
        aligned_audio_path = None
        synced_video_path = None
        final_output_path = None

        try:
            # -----------------------------------------------------------------
            # Stage 1: Video Processor
            # -----------------------------------------------------------------
            # Chuẩn hóa video đầu vào (FPS, Resolution)
            p01 = VideoProcessor()
            normalized_video = p01.process(str(video_p))

            # -----------------------------------------------------------------
            # Stage 2: Audio Extractor
            # -----------------------------------------------------------------
            # Trích xuất file WAV từ video đã chuẩn hóa (Khớp cấu hình 16KHz)
            p02 = AudioExtractor()
            extracted_audio_name = f"{video_p.stem}_extracted.wav"
            target_audio_path = str(self.settings.temp_dir / extracted_audio_name)
            extracted_audio = p02.process(normalized_video, target_audio_path)

            # -----------------------------------------------------------------
            # Stage 3: Speaker Detector (Cần ~2GB VRAM - Kéo mô hình Pyannote)
            # -----------------------------------------------------------------
            p03 = SpeakerDetector()
            speaker_segments = p03.process(extracted_audio)
            
            # GIẢI PHÓNG VRAM NGAY LẬP TỨC
            SpeakerDetector.unload_model()

            # -----------------------------------------------------------------
            # Stage 4: Segment Creator
            # -----------------------------------------------------------------
            # Đóng gói danh sách Segment theo chuẩn dữ liệu models_data
            p04 = SegmentCreator()
            structured_segments = p04.process(speaker_segments)

            # -----------------------------------------------------------------
            # Stage 5: ASR - SenseVoice (Cần ~1.5GB VRAM - Trích xuất tiếng Trung)
            # -----------------------------------------------------------------
            p05 = ASRProcessor()
            asr_segments = p05.process(extracted_audio, structured_segments)
            
            # GIẢI PHÓNG VRAM NGAY LẬP TỨC
            ASRProcessor.unload_model()

            # -----------------------------------------------------------------
            # Stage 6: Translation Local - Qwen2.5 (Cần ~2GB - 3GB VRAM)
            # -----------------------------------------------------------------
            p06 = Translator()
            translated_segments = p06.process(asr_segments)
            
            # GIẢI PHÓNG VRAM NGAY LẬP TỨC
            Translator.unload_model()

            # -----------------------------------------------------------------
            # Stage 7: Voice Clone - Fish Speech Local (Cần ~4GB - 6GB VRAM)
            # -----------------------------------------------------------------
            # VRAM lúc này trống hoàn toàn (~7.5GB khả dụng), Fish Speech tự do bung lụa
            p07 = VoiceCloner()
            cloned_audio_segments = p07.process(translated_segments)
            
            # GIẢI PHÓNG VRAM NGAY LẬP TỨC
            if hasattr(VoiceCloner, 'unload_model'):
                VoiceCloner.unload_model()

            # -----------------------------------------------------------------
            # Stage 8: Audio Alignment
            # -----------------------------------------------------------------
            # Khớp thời lượng và trộn audio nền, xử lý Time Stretching
            p08 = AudioAligner()
            aligned_audio_path = p08.process(cloned_audio_segments)

            # -----------------------------------------------------------------
            # Stage 9: Lip Sync - Wav2Lip Local (Cần ~3GB - 4GB VRAM)
            # -----------------------------------------------------------------
            p09 = LipSyncProcessor()
            synced_video_path = p09.process(normalized_video, aligned_audio_path)
            
            # GIẢI PHÓNG VRAM NGAY LẬP TỨC
            if hasattr(LipSyncProcessor, 'unload_model'):
                LipSyncProcessor.unload_model()

            # -----------------------------------------------------------------
            # Stage 10: Renderer
            # -----------------------------------------------------------------
            # Đóng gói luồng video, luồng audio và xuất file sản phẩm chuẩn hóa
            p10 = Renderer()
            final_output_name = f"{video_p.stem}_dubbed_final.mp4"
            target_output_path = str(self.settings.output_dir / final_output_name)
            final_output_path = p10.process(synced_video_path, aligned_audio_path, target_output_path)

            logger.info("=======================================================")
            logger.info(f"    PIPELINE HOÀN THÀNH XUẤT SẮC: {final_output_path}   ")
            logger.info("=======================================================")
            
            return final_output_path

        except Exception as e:
            logger.error(f"[CRITICAL ERROR] Pipeline dừng hoạt động đột ngột tại một Stage: {e}")
            # Thực hiện cứu hộ dọn dẹp bộ nhớ khẩn cấp phòng trừ dính treo tài nguyên GPU
            self._emergency_clean_gpu()
            raise e

    def _emergency_clean_gpu(self):
        """
        Dọn dẹp VRAM cưỡng bức trong trường hợp xảy ra ngoại lệ (Exception) 
        để tránh GPU bị khóa bộ nhớ ở các tiến trình chạy sau.
        """
        logger.warning("Đang chạy tiến trình dọn dẹp VRAM khẩn cấp (Emergency Clean)...")
        try:
            SpeakerDetector.unload_model()
        except: pass
        try:
            ASRProcessor.unload_model()
        except: pass
        try:
            Translator.unload_model()
        except: pass
        try:
            if hasattr(VoiceCloner, 'unload_model'): VoiceCloner.unload_model()
        except: pass
        try:
            if hasattr(LipSyncProcessor, 'unload_model'): LipSyncProcessor.unload_model()
        except: pass

        # Thu dọn rác hệ thống
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Dọn dẹp khẩn cấp GPU hoàn tất.")