# 🎬 Hướng Dẫn Cài Đặt & Chạy — AI Video Dubbing System

> **Hệ thống AI tự động dịch và lồng tiếng video Trung → Việt**  
> Python 3.10 · CUDA 12.1+ · RTX GPU 6–8 GB VRAM · Windows 10/11

---

## 📋 Mục lục

1. [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
2. [Cài đặt lần đầu](#-cài-đặt-lần-đầu)
3. [Cấu hình API Keys](#-cấu-hình-api-keys)
4. [Tải AI Models](#-tải-ai-models)
5. [Chạy hệ thống](#-chạy-hệ-thống)
6. [Xử lý sự cố](#-xử-lý-sự-cố)
7. [Thư mục cần giữ / có thể xóa](#-thư-mục-cần-giữ--có-thể-xóa)
8. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)

---

## ✅ Yêu cầu hệ thống

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| **OS** | Windows 10 | Windows 11 |
| **Python** | 3.10 | 3.10.x (⚠️ chỉ 3.10) |
| **GPU** | NVIDIA 6 GB VRAM | RTX 3060 / 4060 / 5060 8 GB |
| **CUDA Toolkit** | 11.8 | 12.1+ |
| **ffmpeg** | Bất kỳ | Phiên bản mới nhất |
| **Git** | Bất kỳ | Git for Windows |
| **RAM** | 16 GB | 32 GB |
| **Ổ cứng trống** | 30 GB | 50 GB+ |

> ⚠️ **ffmpeg** phải được thêm vào biến môi trường `PATH`.  
> Tải tại: https://ffmpeg.org/download.html → Giải nén → Thêm thư mục `bin/` vào PATH.

---

## 🚀 Cài đặt lần đầu

### Bước 1 — Kiểm tra môi trường

Mở **Command Prompt** hoặc **PowerShell** và kiểm tra:

```cmd
python --version
:: Phải là Python 3.10.x

git --version
:: Phải hiển thị phiên bản git

ffmpeg -version
:: Phải hiển thị thông tin ffmpeg

nvidia-smi
:: Phải hiển thị thông tin GPU và CUDA
```

### Bước 2 — Truy cập thư mục dự án

```cmd
cd E:\HK8\AI\video_dubbing_ai
```

### Bước 3 — Chạy script cài đặt tự động

Nhấp đúp vào file `setup_env.bat` **hoặc** chạy trong cmd:

```cmd
setup_env.bat
```

Script sẽ tự động:
1. ✅ Tạo môi trường ảo `venv/`
2. ✅ Nâng cấp pip
3. ✅ Cài đặt PyTorch 2.5.1 với CUDA 12.4
4. ✅ Cài đặt tất cả thư viện từ `requirements.txt`
5. ✅ Clone Fish Speech và Wav2Lip vào `third_party/`
6. ✅ Tạo các thư mục: `input/`, `output/`, `temp/`, `cache/`, `models/`
7. ✅ Tạo file `.env` mẫu

> ⏱️ Thời gian cài đặt: **15–30 phút** tùy tốc độ mạng và CPU.

---

## 🔑 Cấu hình API Keys

Sau khi chạy `setup_env.bat`, mở file `.env` ở thư mục gốc dự án:

```env
# ===== REQUIRED =====

# HuggingFace Token - Bắt buộc cho pyannote (Speaker Diarization)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ===== TRANSLATION (chọn 1) =====

# Khuyến nghị: Gemini (miễn phí, nhanh)
GEMINI_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxx
TRANSLATION_PROVIDER=gemini

# Hoặc: OpenAI GPT
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
# TRANSLATION_PROVIDER=openai

# Hoặc: Dùng model local Qwen2.5 (không cần key, nhưng cần tải ~3GB)
# TRANSLATION_PROVIDER=local

# ===== OPTIONAL =====
MODELS_DIR=C:/models
```

### Hướng dẫn lấy API Keys:

| Key | Nguồn | Ghi chú |
|---|---|---|
| `HF_TOKEN` | https://huggingface.co/settings/tokens | Cần accept license pyannote trước |
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey | Miễn phí, khuyến nghị |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys | Trả phí |

> ⚠️ **Để dùng pyannote**, bạn phải:  
> 1. Đăng ký tài khoản HuggingFace  
> 2. Truy cập https://huggingface.co/pyannote/speaker-diarization-3.1  
> 3. Nhấn **"Agree and access repository"**  
> 4. Tạo token tại https://huggingface.co/settings/tokens

---

## 🤖 Tải AI Models

Sau khi cấu hình `.env`, kích hoạt venv và tải models:

```cmd
:: Kích hoạt môi trường ảo
venv\Scripts\activate

:: Tải TẤT CẢ models (~12-15 GB, mất 30-60 phút)
python download_models.py --hf-token hf_xxxxxxxx

:: Hoặc tải từng model riêng lẻ:
python download_models.py --model pyannote --hf-token hf_xxx
python download_models.py --model whisper
python download_models.py --model qwen
python download_models.py --model fish
python download_models.py --model wav2lip

:: Kiểm tra tất cả models đã tải chưa
python download_models.py --verify
```

### Bảng Models cần tải:

| Model | Stage | Kích thước | VRAM cần |
|---|---|---|---|
| `pyannote/speaker-diarization-3.1` | Stage 3 | ~2 GB | ~2.2 GB |
| `openai/whisper-large-v3` | Stage 5 | ~6 GB | ~3 GB |
| `Qwen/Qwen2.5-1.5B-Instruct` | Stage 6 (local) | ~3 GB | ~2–3 GB |
| `Fish Speech S2-Pro` | Stage 7 | ~4 GB | ~4–6 GB |
| `Wav2Lip GAN + s3fd` | Stage 9 | ~600 MB | ~4 GB |

> 💡 Models được lưu tại `C:/models/` (mặc định).

---

## 💻 Chạy hệ thống

### Mỗi lần chạy — Kích hoạt venv trước

```cmd
cd E:\HK8\AI\video_dubbing_ai
venv\Scripts\activate
```

### Cách 1: Web Interface (Khuyến nghị)

```cmd
python run_web.py
```

Mở trình duyệt tại: **http://localhost:8000**

**Các bước sử dụng trên Web:**
1. Kéo thả file video tiếng Trung vào khu vực **Upload Video**
2. Chọn API dịch thuật: `Gemini` / `OpenAI` / `Local`
3. Bật/tắt **Lip Sync** (tắt = nhanh hơn 3–4 lần)
4. Nhấn **"Bắt đầu Dubbing"**
5. Theo dõi tiến trình 10 bước qua WebSocket
6. Tải video kết quả khi hoàn tất

### Cách 2: CLI (Command Line)

```cmd
:: Chạy cơ bản
python main.py input/video.mp4

:: Chỉ định file đầu ra
python main.py input/video.mp4 --output output/ket_qua.mp4

:: Bỏ qua Lip Sync (nhanh hơn 3-4 lần)
python main.py input/video.mp4 --skip-lipsync

:: Xem log chi tiết
python main.py input/video.mp4 --log-level DEBUG

:: Kết hợp nhiều tùy chọn
python main.py input/video.mp4 --output output/result.mp4 --skip-lipsync --log-level INFO
```

### Kiểm tra GPU

```cmd
python check_cuda.py
```

---

## 🔧 Xử lý sự cố

### Lỗi "Out of Memory (OOM)" trên GPU
- Đóng tất cả ứng dụng đang dùng GPU
- Dùng `--skip-lipsync` để giảm tải VRAM

### Lỗi "No module named X"
```cmd
venv\Scripts\activate
pip install -r requirements.txt
```

### Lỗi pyannote "401 Unauthorized"
- Kiểm tra `HF_TOKEN` trong file `.env`
- Đảm bảo đã accept license tại HuggingFace

### ffmpeg không tìm thấy
```cmd
ffmpeg -version
:: Nếu không có, tải tại https://ffmpeg.org/download.html
```

---

## 🗂️ Thư mục cần giữ / có thể xóa

### ✅ Giữ nguyên (quan trọng)

| Thư mục / File | Mục đích |
|---|---|
| `venv/` | Môi trường Python |
| `third_party/` | Fish Speech + Wav2Lip |
| `pipeline/` | 10 giai đoạn xử lý chính |
| `services/` | Logic tích hợp |
| `api/` | FastAPI backend |
| `web/` | Giao diện web |
| `config/` | Cấu hình hệ thống |
| `models_data/` | Data models |
| `utils/` | Công cụ hỗ trợ |
| `input/` | Video đầu vào |
| `output/` | Video kết quả |
| `.env` | API Keys |
| `main.py` | Entry point CLI |
| `run_web.py` | Entry point Web |
| `requirements.txt` | Danh sách thư viện |
| `setup_env.bat` | Script cài đặt |
| `download_models.py` | Script tải models |

### 🗑️ Có thể xóa an toàn

| Thư mục / File | Lý do có thể xóa |
|---|---|
| `temp/` | File tạm — tự động tái tạo |
| `cache/` | Cache HuggingFace — tự tải lại |
| `OpenVoiceV2/` | Thư mục trống, không dùng trong pipeline |
| `scripts/` | Thư mục trống |
| `.ag-kit-backups/` | Backup tự động của ag-kit |
| `package.json` | Không cần cho Python project |
| `freeze.txt` | Chỉ tham khảo |
| `versions.txt` | Chỉ tham khảo |
| `pipeline.md` | Đã có ARCHITECTURE.md đầy đủ hơn |
| `test.py` | File test đơn giản |
| `__pycache__/` | Cache Python — tự tái tạo |
| `output/stages/` | Output trung gian — xóa sau khi done |

> ⚠️ **KHÔNG xóa `venv/`** trừ khi muốn cài lại từ đầu (mất 15–30 phút).  
> ⚠️ **KHÔNG xóa `third_party/`** trừ khi muốn tải lại Fish Speech + Wav2Lip.

---

## 📁 Cấu trúc thư mục

```
video_dubbing_ai/
│
├── main.py                  ← CLI Entry Point
├── run_web.py               ← Web Server Entry Point  
├── download_models.py       ← Script tải AI models
├── setup_env.bat            ← Script cài đặt môi trường
├── check_cuda.py            ← Kiểm tra GPU/CUDA
├── requirements.txt         ← Python dependencies
├── .env                     ← API Keys (KHÔNG commit git)
│
├── api/                     ← FastAPI Backend
├── pipeline/                ← 10 Giai đoạn Pipeline
├── services/                ← Service Layer
├── config/                  ← Cấu hình
├── models_data/             ← Data Models
├── utils/                   ← Tiện ích  
├── web/                     ← Frontend HTML/CSS/JS
│
├── venv/                    ← Môi trường ảo Python
├── third_party/             ← Fish Speech + Wav2Lip + Qwen
│   ├── fish-speech/
│   ├── Wav2Lip/
│   └── qwen2.5/
│
├── input/                   ← Đặt video cần lồng tiếng vào đây
├── output/                  ← Video kết quả xuất ra đây
├── temp/                    ← File tạm thời
└── cache/                   ← Cache HuggingFace models
```

---

## ⚡ Luồng xử lý 10 bước

```
Video tiếng Trung (.mp4)
        │
        ▼
[Stage 01] Video Processor  — Chuẩn hóa H264/30FPS
[Stage 02] Audio Extractor  — Trích xuất WAV 16KHz Mono
[Stage 03] Speaker Detector — Phân biệt người nói (pyannote)
[Stage 04] Segment Creator  — Cắt audio theo từng câu thoại
[Stage 05] Chinese ASR      — Nhận dạng giọng nói (SenseVoice)
[Stage 06] Translation      — Dịch Trung → Việt (Gemini/GPT/Qwen)
[Stage 07] Voice Cloning    — Sinh giọng tiếng Việt (Fish Speech)
[Stage 08] Audio Alignment  — Khớp thời lượng (Time-stretch)
[Stage 09] Lip Sync         — Đồng bộ khẩu hình (Wav2Lip)
[Stage 10] Video Renderer   — Ghép audio + video (ffmpeg)
        │
        ▼
Video tiếng Việt (*_vi.mp4)
```

---

*Tài liệu cập nhật: 2026-08-01*
