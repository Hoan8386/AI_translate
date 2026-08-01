"""
╔══════════════════════════════════════════════════════════╗
║  Chinese Video To Vietnamese AI Dubbing System          ║
║  ──────────────────────────────────────────────────────  ║
║  Input:  Video tiếng Trung                              ║
║  Output: Video tiếng Việt                               ║
║                                                         ║
║  Features:                                              ║
║  ✓ Tự nhận diện nhiều người nói                         ║
║  ✓ Giữ đặc trưng giọng từng người                      ║
║  ✓ Đồng bộ thời lượng                                  ║
║  ✓ Đồng bộ khẩu hình (Wav2Lip)                         ║
║  ✓ Tối ưu RTX 5060 8GB (1 model/GPU)                   ║
╚══════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from models_data.job import Job, JobStatus
from utils.logger import setup_logger, get_logger, log_stage
from utils.timer import Timer
from utils.file_manager import FileManager
from utils.gpu_manager import GPUManager

# Services
from services.video_service import VideoService
from services.audio_service import AudioService
from services.speaker_service import SpeakerService
from services.asr_service import ASRService
from services.translation_service import TranslationService
from services.voice_service import VoiceService
from services.lipsync_service import LipSyncService
from services.renderer_service import RendererService


def process_video(input_path: str, output_name: str = None, skip_lipsync: bool = False,
                   progress_callback=None):
    """
    ╔════════════════════════════════════════╗
    ║  HÀM ĐIỀU PHỐI TRUNG TÂM            ║
    ║  Chạy tuần tự 10 giai đoạn pipeline  ║
    ╚════════════════════════════════════════╝
    
    Args:
        input_path: Đường dẫn video input (tiếng Trung)
        output_name: Tên file output (mặc định: input_name_vi.mp4)
        skip_lipsync: Bỏ qua lip sync (nhanh hơn, chỉ thay audio)
        progress_callback: Hàm callback để báo tiến trình (dùng cho web API)
            Signature: callback(stage, stage_name, progress, message, log_type,
                                segments, vram_used, vram_model)
    """
    # Helper để gọi callback (nếu có)
    def notify(stage, stage_name, progress, message="", log_type="info",
               segments=None, vram_used=0, vram_model="",
               stage_data=None, timing=None):
        if progress_callback:
            progress_callback(
                stage=stage, stage_name=stage_name, progress=progress,
                message=message, log_type=log_type, segments=segments,
                vram_used=vram_used, vram_model=vram_model,
                stage_data=stage_data, timing=timing,
            )
    # === SETUP ===
    settings = get_settings()
    logger = setup_logger("dubbing", settings.log_level)
    file_mgr = FileManager(settings.project_root)
    gpu = GPUManager()
    global_timer = Timer()
    
    # Tạo Job ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file = Path(input_path)
    job_id = f"{input_file.stem}_{timestamp}"
    
    # Tạo Job
    job = Job(
        id=job_id,
        input_video=str(input_path),
    )
    
    logger.info(
        f"\n{'='*60}\n"
        f"  🎬 CHINESE → VIETNAMESE DUBBING PIPELINE\n"
        f"  Job ID: {job_id}\n"
        f"  Input:  {input_path}\n"
        f"  GPU:    {gpu}\n"
        f"{'='*60}\n"
    )
    
    try:
        # Tạo job directory
        job_dir = file_mgr.create_job_dir(job_id)
        job.update_status(JobStatus.PROCESSING)
        
        # ─────────────────────────────────────────
        # STAGE 1: Video Processor
        # ─────────────────────────────────────────
        global_timer.start("Stage 01: Video Processor")
        job.update_status(JobStatus.STAGE_1_VIDEO, 1)
        notify(1, "Video Processor", 5, "Chuẩn hóa H264/30FPS/AAC...")
        
        video_svc = VideoService()
        stage_01_dir = file_mgr.get_stage_dir(job_id, "stage_01_video_processor")
        normalized_path = str(stage_01_dir / "normalized.mp4")
        video_svc.normalize(input_path, normalized_path)
        job.normalized_video = normalized_path
        
        # Lấy thông tin video
        video_info = video_svc.get_info(normalized_path)
        total_duration = video_info["duration"]
        
        global_timer.stop("Stage 01: Video Processor")
        elapsed_01 = global_timer.get_elapsed("Stage 01: Video Processor")
        notify(1, "Video Processor", 10, "✅ Video chuẩn hóa xong", "success",
               stage_data={
                   "video_info": video_info,
                   "normalized_path": normalized_path,
                   "output_dir": str(stage_01_dir),
               },
               timing={"duration_s": round(elapsed_01, 2)})
        
        # ─────────────────────────────────────────
        # STAGE 2: Audio Extractor
        # ─────────────────────────────────────────
        global_timer.start("Stage 02: Audio Extractor")
        job.update_status(JobStatus.STAGE_2_AUDIO, 2)
        notify(2, "Audio Extractor", 15, "Trích xuất audio 16KHz Mono WAV...")
        
        audio_svc = AudioService()
        stage_02_dir = file_mgr.get_stage_dir(job_id, "stage_02_audio_extractor")
        audio_path = str(stage_02_dir / "audio.wav")
        audio_svc.extract(normalized_path, audio_path)
        job.extracted_audio = audio_path
        
        global_timer.stop("Stage 02: Audio Extractor")
        elapsed_02 = global_timer.get_elapsed("Stage 02: Audio Extractor")
        notify(2, "Audio Extractor", 20, "✅ Trích xuất audio xong", "success",
               stage_data={
                   "audio_path": audio_path,
                   "sample_rate": 16000,
                   "channels": "mono",
                   "output_dir": str(stage_02_dir),
               },
               timing={"duration_s": round(elapsed_02, 2)})
        
        # ─────────────────────────────────────────
        # STAGE 3: Speaker Detector
        # ─────────────────────────────────────────
        global_timer.start("Stage 03: Speaker Detector")
        job.update_status(JobStatus.STAGE_3_SPEAKER, 3)
        notify(3, "Speaker Detector", 25, "🔧 GPU: Loading pyannote...",
               vram_used=2200, vram_model="pyannote")
        
        speaker_svc = SpeakerService()
        stage_03_dir = file_mgr.get_stage_dir(job_id, "stage_03_speaker_detector")
        diarization = speaker_svc.detect(audio_path)
        diarization_path = stage_03_dir / "diarization.json"
        with diarization_path.open("w", encoding="utf-8") as handle:
            json.dump(diarization, handle, ensure_ascii=False, indent=2)
        
        global_timer.stop("Stage 03: Speaker Detector")
        elapsed_03 = global_timer.get_elapsed("Stage 03: Speaker Detector")
        notify(3, "Speaker Detector", 30, "✅ Phát hiện speakers xong", "success",
               vram_used=0, vram_model="",
               stage_data={
                   "diarization": diarization,
                   "num_speakers": len(set(s.get("speaker", "") for s in diarization)) if isinstance(diarization, list) else 0,
                   "output_dir": str(stage_03_dir),
               },
               timing={"duration_s": round(elapsed_03, 2)})
        
        # ─────────────────────────────────────────
        # STAGE 4: Segment Creator
        # ─────────────────────────────────────────
        global_timer.start("Stage 04: Segment Creator")
        job.update_status(JobStatus.STAGE_4_SEGMENT, 4)
        notify(4, "Segment Creator", 35, "Cắt audio theo speaker...")
        
        stage_04_dir = file_mgr.get_stage_dir(job_id, "stage_04_segment_creator")
        segments_dir = str(stage_04_dir / "segments")
        segments, speakers = audio_svc.create_segments(
            audio_path, diarization, segments_dir
        )
        job.segments = segments
        for spk in speakers.values():
            job.add_speaker(spk)
        
        global_timer.stop("Stage 04: Segment Creator")
        elapsed_04 = global_timer.get_elapsed("Stage 04: Segment Creator")
        seg_preview = [{"id": s.id, "speaker": s.speaker, "start": s.start, "end": s.end}
                       for s in segments[:20]]
        notify(4, "Segment Creator", 40,
               f"✅ Tạo {len(segments)} segments, {len(speakers)} speakers", "success",
               stage_data={
                   "num_segments": len(segments),
                   "num_speakers": len(speakers),
                   "speakers": list(speakers.keys()),
                   "segments_preview": seg_preview,
                   "output_dir": str(stage_04_dir),
               },
               timing={"duration_s": round(elapsed_04, 2)})
        
        # ─────────────────────────────────────────
        # STAGE 5: ASR (Chinese Speech-to-Text)
        # ─────────────────────────────────────────
        global_timer.start("Stage 05: Chinese ASR")
        job.update_status(JobStatus.STAGE_5_ASR, 5)
        notify(5, "Chinese ASR", 45, "🔧 GPU: Loading SenseVoice...",
               vram_used=1800, vram_model="SenseVoice")
        
        asr_svc = ASRService()
        stage_05_dir = file_mgr.get_stage_dir(job_id, "stage_05_asr")
        segments = asr_svc.transcribe(audio_path, segments)
        job.segments = segments
        asr_output_path = stage_05_dir / "asr_segments.json"
        with asr_output_path.open("w", encoding="utf-8") as handle:
            json.dump([
                {"id": s.id, "speaker": s.speaker, "start": s.start,
                 "end": s.end, "zh_text": s.zh_text, "vi_text": s.vi_text}
                for s in segments
            ], handle, ensure_ascii=False, indent=2)
        
        # Gửi kết quả ASR cho frontend
        seg_data = [{"id": s.id, "speaker": s.speaker,
                     "start": s.start, "end": s.end,
                     "zh_text": s.zh_text, "vi_text": s.vi_text}
                    for s in segments if s.zh_text]
        
        global_timer.stop("Stage 05: Chinese ASR")
        elapsed_05 = global_timer.get_elapsed("Stage 05: Chinese ASR")
        notify(5, "Chinese ASR", 50, "✅ Nhận dạng tiếng Trung xong", "success",
               segments=seg_data, vram_used=0, vram_model="",
               stage_data={
                   "num_segments": len(seg_data),
                   "segments": seg_data,
                   "asr_json": str(asr_output_path),
                   "output_dir": str(stage_05_dir),
               },
               timing={"duration_s": round(elapsed_05, 2)})
        
        # ─────────────────────────────────────────
        # STAGE 6: Translation (ZH → VI)
        # ─────────────────────────────────────────
        global_timer.start("Stage 06: Translation")
        job.update_status(JobStatus.STAGE_6_TRANSLATE, 6)
        notify(6, "Translation", 55, "Dịch ZH → VI...")
        
        trans_svc = TranslationService()
        stage_06_dir = file_mgr.get_stage_dir(job_id, "stage_06_translation")
        segments = trans_svc.translate(segments)
        job.segments = segments
        translation_output_path = stage_06_dir / "translated_segments.json"
        with translation_output_path.open("w", encoding="utf-8") as handle:
            json.dump([
                {"id": s.id, "speaker": s.speaker, "start": s.start,
                 "end": s.end, "zh_text": s.zh_text, "vi_text": s.vi_text}
                for s in segments
            ], handle, ensure_ascii=False, indent=2)
        
        # Gửi kết quả dịch cho frontend
        seg_data = [{"id": s.id, "speaker": s.speaker,
                     "start": s.start, "end": s.end,
                     "zh_text": s.zh_text, "vi_text": s.vi_text}
                    for s in segments if s.vi_text]
        
        global_timer.stop("Stage 06: Translation")
        elapsed_06 = global_timer.get_elapsed("Stage 06: Translation")
        notify(6, "Translation", 60, "✅ Dịch thuật xong", "success",
               segments=seg_data,
               stage_data={
                   "num_segments": len(seg_data),
                   "segments": seg_data,
                   "translation_json": str(translation_output_path),
                   "output_dir": str(stage_06_dir),
               },
               timing={"duration_s": round(elapsed_06, 2)})
        
        # ─────────────────────────────────────────
        # STAGE 7: Voice Cloning
        # ─────────────────────────────────────────
        global_timer.start("Stage 07: Voice Cloning")
        job.update_status(JobStatus.STAGE_7_VOICE, 7)
        notify(7, "Voice Cloning", 65, "🔧 Fish Speech: Sinh giọng Việt...",
               vram_used=0, vram_model="Fish Speech (External)")
        
        voice_svc = VoiceService()
        stage_07_dir = file_mgr.get_stage_dir(job_id, "stage_07_voice_cloning")
        generated_dir = str(stage_07_dir / "generated")
        segments = voice_svc.clone(segments, speakers, generated_dir)
        job.segments = segments
        
        global_timer.stop("Stage 07: Voice Cloning")
        elapsed_07 = global_timer.get_elapsed("Stage 07: Voice Cloning")
        cloned_files = [s.generated_audio for s in segments if getattr(s, "generated_audio", "")]
        notify(7, "Voice Cloning", 70, "✅ Clone giọng nói xong", "success",
               vram_used=0, vram_model="",
               stage_data={
                   "num_cloned": len(cloned_files),
                   "generated_dir": generated_dir,
                   "output_dir": str(stage_07_dir),
               },
               timing={"duration_s": round(elapsed_07, 2)})
        
        # ─────────────────────────────────────────
        # STAGE 8: Audio Alignment
        # ─────────────────────────────────────────
        global_timer.start("Stage 08: Audio Alignment")
        job.update_status(JobStatus.STAGE_8_ALIGN, 8)
        notify(8, "Audio Alignment", 75, "Đồng bộ thời lượng audio...")
        
        stage_08_dir = file_mgr.get_stage_dir(job_id, "stage_08_audio_alignment")
        aligned_dir = str(stage_08_dir / "aligned")
        segments = audio_svc.align(segments, aligned_dir)
        job.segments = segments
        
        # Merge tất cả segments
        merged_audio_path = str(stage_08_dir / "merged_audio.wav")
        audio_svc.merge_segments(segments, total_duration, merged_audio_path)
        job.merged_audio = merged_audio_path
        
        global_timer.stop("Stage 08: Audio Alignment")
        elapsed_08 = global_timer.get_elapsed("Stage 08: Audio Alignment")
        notify(8, "Audio Alignment", 80, "✅ Đồng bộ audio xong", "success",
               stage_data={
                   "merged_audio": merged_audio_path,
                   "aligned_dir": aligned_dir,
                   "num_segments": len(segments),
                   "output_dir": str(stage_08_dir),
               },
               timing={"duration_s": round(elapsed_08, 2)})
        
        # ─────────────────────────────────────────
        # STAGE 9: Lip Sync
        # ─────────────────────────────────────────
        global_timer.start("Stage 09: Lip Sync")
        job.update_status(JobStatus.STAGE_9_LIPSYNC, 9)
        
        stage_09_dir = file_mgr.get_stage_dir(job_id, "stage_09_lipsync")
        lipsync_svc = LipSyncService()
        lipsync_path = str(stage_09_dir / "lipsync.mp4")
        
        if skip_lipsync:
            notify(9, "Lip Sync", 85, "Lip sync bị bỏ qua, thay audio trực tiếp...")
            logger.info("Lip sync bị bỏ qua (--skip-lipsync)")
            lipsync_svc.simple_replace(normalized_path, merged_audio_path, lipsync_path)
        else:
            notify(9, "Lip Sync", 85, "🔧 GPU: Loading Wav2Lip...",
                   vram_used=4100, vram_model="Wav2Lip")
            lipsync_svc.sync(normalized_path, merged_audio_path, lipsync_path)
        
        job.lipsync_video = lipsync_path
        
        global_timer.stop("Stage 09: Lip Sync")
        elapsed_09 = global_timer.get_elapsed("Stage 09: Lip Sync")
        notify(9, "Lip Sync", 90, "✅ Lip sync xong", "success",
               vram_used=0, vram_model="",
               stage_data={
                   "lipsync_path": lipsync_path,
                   "skip_lipsync": skip_lipsync,
                   "output_dir": str(stage_09_dir),
               },
               timing={"duration_s": round(elapsed_09, 2)})
        
        # ─────────────────────────────────────────
        # STAGE 10: Final Render
        # ─────────────────────────────────────────
        global_timer.start("Stage 10: Renderer")
        job.update_status(JobStatus.STAGE_10_RENDER, 10)
        notify(10, "Video Renderer", 95, "Ghép video + audio → output.mp4...")
        
        renderer_svc = RendererService()
        stage_10_dir = file_mgr.get_stage_dir(job_id, "stage_10_renderer")
        
        # Xác định output path
        if output_name is None:
            output_name = f"{input_file.stem}_vi.mp4"
        output_path = str(settings.output_dir / output_name)
        stage_output_path = str(stage_10_dir / output_name)
        
        renderer_svc.render(lipsync_path, merged_audio_path, stage_output_path)
        
        # Copy final render sang output chính
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage_output_path, output_path)
        
        job.output_video = output_path
        
        global_timer.stop("Stage 10: Renderer")
        elapsed_10 = global_timer.get_elapsed("Stage 10: Renderer")
        notify(10, "Video Renderer", 100, "✅ Render video xong", "success",
               stage_data={
                   "output_path": output_path,
                   "output_name": output_name,
                   "output_dir": str(stage_10_dir),
               },
               timing={"duration_s": round(elapsed_10, 2)})
        
        # ═════════════════════════════════════════
        # HOÀN THÀNH
        # ═════════════════════════════════════════
        job.update_status(JobStatus.COMPLETED)
        
        # Lưu job state
        job_state_path = job_dir / "job_state.json"
        job.save(job_state_path)
        
        # In báo cáo
        global_timer.report()
        
        logger.info(
            f"\n{'='*60}\n"
            f"  ✅ DUBBING HOÀN THÀNH!\n"
            f"  Output: {output_path}\n"
            f"  Segments: {len(segments)}\n"
            f"  Speakers: {len(speakers)}\n"
            f"{'='*60}\n"
        )
        
        return output_path
        
    except Exception as e:
        job.update_status(JobStatus.FAILED)
        job.error_message = str(e)
        
        # Lưu job state khi lỗi
        try:
            job_state_path = job_dir / "job_state.json"
            job.save(job_state_path)
        except Exception:
            pass
        
        logger.error(f"\n{'='*60}\n  ❌ PIPELINE FAILED: {e}\n{'='*60}\n")
        raise


def main():
    """CLI Entry Point"""
    parser = argparse.ArgumentParser(
        description="🎬 Chinese Video To Vietnamese AI Dubbing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python main.py input/video.mp4
  python main.py input/video.mp4 --output my_video_vi.mp4
  python main.py input/video.mp4 --skip-lipsync
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Đường dẫn video tiếng Trung (input)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Tên file output (mặc định: input_vi.mp4)"
    )
    
    parser.add_argument(
        "--skip-lipsync",
        action="store_true",
        help="Bỏ qua lip sync (nhanh hơn, chỉ thay audio)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (mặc định: INFO)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ File không tồn tại: {args.input}")
        sys.exit(1)
    
    if not input_path.suffix.lower() in ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv'):
        print(f"⚠ File có thể không phải video: {args.input}")
    
    # Set log level
    settings = get_settings()
    settings.log_level = args.log_level
    
    # Run pipeline
    try:
        output = process_video(
            str(input_path),
            output_name=args.output,
            skip_lipsync=args.skip_lipsync,
        )
        print(f"\n✅ Output: {output}")
        
    except KeyboardInterrupt:
        print("\n⚠ Pipeline bị hủy bởi người dùng")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
