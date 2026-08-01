# 🎬 AI Dubbing System — Architecture Documentation

> **Chinese Video → Vietnamese AI Dubbing**  
> Phiên bản: V3 | GPU Target: RTX 5060 8GB VRAM | Python 3.10+

---

## 📁 Cấu trúc thư mục

```
video_dubbing_ai/
│
├── 📄 main.py                  ← CLI Entry Point + Orchestrator (10 stages)
├── 📄 run_web.py               ← Khởi động FastAPI Web Server
├── 📄 check_cuda.py            ← Kiểm tra GPU & CUDA
├── 📄 requirements.txt         ← Python dependencies
├── 📄 setup_env.bat            ← Windows setup script
├── 📄 ARCHITECTURE.md          ← Tài liệu này
│
├── 📂 api/                     ← FastAPI Backend
│   ├── server.py               ← FastAPI app factory (CORS, static, WS)
│   ├── routes.py               ← REST API endpoints
│   ├── websocket.py            ← WebSocket connection manager
│   └── pipeline_runner.py     ← Job management + background threading
│
├── 📂 pipeline/                ← 10 Pipeline Stages
│   ├── p01_video_processor.py  ← Stage 1: Normalize video H264/30FPS/AAC
│   ├── p02_audio_extractor.py  ← Stage 2: Extract audio 16KHz Mono WAV
│   ├── p03_speaker_detector.py ← Stage 3: Speaker diarization (pyannote)
│   ├── p04_segment_creator.py  ← Stage 4: Create audio segments per speaker
│   ├── p05_asr.py              ← Stage 5: Chinese ASR (SenseVoice)
│   ├── p06_translation.py      ← Stage 6: ZH→VI Translation (Qwen2.5 / API)
│   ├── p07_voice_clone.py      ← Stage 7: Vietnamese TTS (Fish Speech)
│   ├── p08_audio_alignment.py  ← Stage 8: Align duration + merge audio
│   ├── p09_lipsync.py          ← Stage 9: Lip sync (Wav2Lip)
│   └── p10_renderer.py         ← Stage 10: Final video render (ffmpeg)
│
├── 📂 services/                ← High-level Service Layer
│   ├── video_service.py        ← Video normalize + info
│   ├── audio_service.py        ← Audio extract, segment, align, merge
│   ├── speaker_service.py      ← Speaker detection wrapper
│   ├── asr_service.py          ← ASR transcription wrapper
│   ├── translation_service.py  ← Translation API/local wrapper
│   ├── voice_service.py        ← Voice cloning wrapper
│   ├── lipsync_service.py      ← Lip sync wrapper
│   └── renderer_service.py     ← Video render wrapper
│
├── 📂 config/
│   └── settings.py             ← Centralized config (dataclass singleton)
│
├── 📂 models_data/             ← Python data models (NOT AI models)
│   ├── job.py                  ← Job dataclass + JobStatus enum
│   ├── segment.py              ← Segment dataclass
│   └── speaker.py              ← Speaker dataclass
│
├── 📂 utils/
│   ├── file_manager.py         ← Job directory management
│   ├── gpu_manager.py          ← VRAM monitoring + GPU utilities
│   ├── logger.py               ← Logging setup
│   └── timer.py                ← Pipeline timing (with get_elapsed)
│
├── 📂 web/                     ← Frontend (Vanilla HTML/CSS/JS)
│   ├── index.html              ← Single-page app (4 tabs)
│   ├── style.css               ← Dark mode design system + Step Viewer CSS
│   └── script.js               ← Tab switching, WebSocket, StepViewer class
│
├── 📂 input/                   ← Upload videos vào đây
├── 📂 output/                  ← Output videos (final results)
│   └── stages/
│       └── <job_id>/
│           ├── stage_01_video_processor/
│           │   ├── manifest.json
│           │   ├── data.json
│           │   └── normalized.mp4
│           ├── stage_02_audio_extractor/
│           │   └── audio.wav
│           ├── stage_03_speaker_detector/
│           │   └── diarization.json
│           ├── stage_05_asr/
│           │   └── asr_segments.json
│           ├── stage_06_translation/
│           │   └── translated_segments.json
│           ├── stage_08_audio_alignment/
│           │   └── merged_audio.wav
│           ├── stage_09_lipsync/
│           │   └── lipsync.mp4
│           └── stage_10_renderer/
│               └── <video>_vi.mp4
│
├── 📂 temp/                    ← Temporary working files (per job)
├── 📂 cache/                   ← Model cache (HuggingFace)
│
└── C:/models/                  ← AI Models directory
    ├── wav2lip/
    │   └── wav2lip_gan.pth
    ├── pyannote/
    ├── sensevoice/
    ├── fish-speech/
    └── qwen/
```

---

## 🤖 AI Models & VRAM

| Stage | Model | VRAM | Framework |
|-------|-------|------|-----------|
| 03 | pyannote/speaker-diarization-3.1 | ~2.2 GB | PyTorch |
| 05 | SenseVoiceSmall | ~1.8 GB | FunASR |
| 06 | Qwen2.5 / Gemini API / GPT | ~2-3 GB | Transformers |
| 07 | Fish Speech S2-Pro | ~4-6 GB | PyTorch |
| 09 | Wav2Lip GAN | ~4.1 GB | PyTorch |

> **Nguyên tắc vàng**: CHỈ 1 MODEL TRÊN GPU TẠI 1 THỜI ĐIỂM → unload_model() sau mỗi stage

---

## 🔄 Luồng dữ liệu

`
Video Input (Chinese .mp4)
    → Stage 01 → Normalized Video (H264/30FPS/AAC)
    → Stage 02 → Audio WAV (16KHz, Mono)
    → Stage 03 → Speaker Diarization [{speaker, start, end}]
    → Stage 04 → Structured Segments
    → Stage 05 → Segments + zh_text (Chinese ASR)
    → Stage 06 → Segments + vi_text (Vietnamese Translation)
    → Stage 07 → Segments + generated_audio (Vietnamese WAV)
    → Stage 08 → Aligned segments + merged_audio.wav
    → Stage 09 → Lip-synced video
    → Stage 10 → Final output_vi.mp4
`

---

## 🌐 REST API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | /api/upload | Upload video, tạo job |
| POST | /api/jobs/{id}/start | Bắt đầu pipeline |
| GET | /api/jobs/{id} | Trạng thái job |
| GET | /api/jobs | Danh sách jobs |
| GET | /api/jobs/{id}/download | Tải video kết quả |
| DELETE | /api/jobs/{id} | Xóa job |
| GET | /api/jobs/{id}/stages | Danh sách stages + data |
| GET | /api/jobs/{id}/stages/{n} | Chi tiết stage N |
| GET | /api/jobs/{id}/output-files | Liệt kê output files |
| WS | /ws/jobs/{id} | WebSocket realtime updates |

---

## ⚙️ Cấu hình (config/settings.py)

| Setting | Mặc định | Env Var |
|---------|----------|---------|
| models_dir | C:/models | MODELS_DIR |
| output_stages_dir | output/stages/ | - |
| translation.provider | gemini | TRANSLATION_PROVIDER |
| translation.gemini_api_key | - | GEMINI_API_KEY |
| speaker.hf_token | - | HF_TOKEN |

---

## 🚀 Cách chạy

`ash
# Web Interface (khuyến nghị)
python run_web.py
# Mở: http://localhost:8000

# CLI
python main.py input/video.mp4
python main.py input/video.mp4 --output my_vi.mp4 --skip-lipsync
`

---

*Tài liệu được tạo tự động — 2026-08-01*
