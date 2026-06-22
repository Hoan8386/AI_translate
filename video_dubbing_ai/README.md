# 🎬 Chinese Video To Vietnamese AI Dubbing System

<p align="center">
  <strong>Hệ thống AI tự động dịch thuật và lồng tiếng video từ tiếng Trung sang tiếng Việt</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/CUDA-12.1+-green.svg" alt="CUDA Version">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-red.svg" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-teal.svg" alt="FastAPI Version">
  <img src="https://img.shields.io/badge/Platform-Windows-lightgrey.svg" alt="Platform Windows">
</p>

---

## 🌟 Giới thiệu

**Chinese Video To Vietnamese AI Dubbing System** là một giải pháp ứng dụng trí tuệ nhân tạo toàn diện giúp tự động hóa quy trình dịch thuật và lồng tiếng (dubbing) các video tiếng Trung sang tiếng Việt. 

Hệ thống kết hợp nhiều mô hình học sâu (Deep Learning) hiện đại để thực hiện toàn bộ các bước từ phát hiện người nói, nhận dạng giọng nói (ASR), dịch thuật ngữ cảnh, tổng hợp giọng nói giữ nguyên đặc trưng (Voice Cloning), đồng bộ thời lượng cho đến đồng bộ khẩu hình nhân vật (Lip Sync).

Dự án cung cấp cả **giao diện dòng lệnh (CLI)** mạnh mẽ cho lập trình viên và **giao diện Web (Web Dashboard)** trực quan giúp dễ dàng tương tác, theo dõi tài nguyên hệ thống (VRAM GPU) và tiến độ xử lý thời gian thực qua WebSocket.

---

## ✨ Tính năng chính

| Tính năng | Chi tiết công nghệ | Mô tả |
|-----------|--------------------|-------|
| 👥 **Multi-Speaker Detection** | `pyannote.audio` | Tự động phát hiện số lượng và phân biệt danh tính của từng người nói trong video. |
| 🗣 **Voice Cloning** | `Fish Speech` | Trích xuất đặc trưng giọng nói của từng speaker gốc để sinh ra giọng lồng tiếng Việt tương đương (giữ nguyên âm sắc, giới tính, cảm xúc). |
| ⏱ **Duration Sync** | `librosa` (Time-stretch) | Tự động co giãn thời lượng file audio tiếng Việt được tạo ra để khớp hoàn toàn với khoảng thời gian nói của nhân vật trong video gốc. |
| 👄 **Lip Sync** | `Wav2Lip` | Đồng bộ khẩu hình chuyển động môi của nhân vật trong video theo file audio tiếng Việt mới, tạo cảm giác lồng tiếng tự nhiên. |
| 🚀 **Web Dashboard** | `FastAPI` + `WebSockets` | Giao diện web hiện đại hiển thị trực quan 10 bước xử lý của pipeline, log chi tiết, biểu đồ tài nguyên VRAM GPU và tải kết quả trực tiếp. |
| ⚡ **GPU VRAM Optimized** | `GPUManager` | Cơ chế dọn dẹp và quản lý VRAM động giúp hệ thống chạy mượt mà trên card đồ họa phổ thông (như RTX 5060 8GB) mà không bị tràn bộ nhớ (OOM). |

---

## 🏗 Kiến trúc hệ thống Pipeline (10 Giai đoạn)

Quy trình xử lý một video đầu vào trải qua 10 giai đoạn tuần tự khép kín:

```
                    INPUT VIDEO (.mp4)
                           │
                           ▼
             ┌─────────────────────────┐
             │  1. Video Processor     │  Chuẩn hóa video (H264 / 30 FPS)
             └─────────────────────────┘
                           │
                           ▼
             ┌─────────────────────────┐
             │  2. Audio Extractor     │  Trích xuất âm thanh (16KHz / Mono WAV)
             └─────────────────────────┘
                           │
                           ▼
             ┌─────────────────────────┐
             │  3. Speaker Detector    │  Phát hiện phân đoạn người nói (pyannote)
             └─────────────────────────┘
                           │
                           ▼
             ┌─────────────────────────┐
             │  4. Segment Creator     │  Cắt âm thanh gốc thành các đoạn nhỏ
             └─────────────────────────┘
                           │
                           ▼
             ┌─────────────────────────┐
             │  5. Chinese ASR         │  Nhận dạng giọng nói (SenseVoice Small)
             └─────────────────────────┘
                           │
                           ▼
             ┌─────────────────────────┐
             │  6. Translation         │  Dịch ZH → VI (Gemini API hoặc OpenAI GPT)
             └─────────────────────────┘
                           │
                           ▼
             ┌─────────────────────────┐
             │  7. Voice Cloning       │  Sinh giọng nói tiếng Việt (Fish Speech)
             └─────────────────────────┘
                           │
                           ▼
             ┌─────────────────────────┐
             │  8. Audio Alignment     │  Khớp thời lượng (Time-stretching)
             └─────────────────────────┘
                           │
                           ▼
             ┌─────────────────────────┐
             │  9. Lip Sync            │  Đồng bộ khẩu hình môi (Wav2Lip)
             └─────────────────────────┘
                           │
                           ▼
             ┌─────────────────────────┐
             │ 10. Video Renderer      │  Hợp nhất hình ảnh + âm thanh bằng ffmpeg
             └─────────────────────────┘
                           │
                           ▼
                    OUTPUT VIDEO (.mp4)
```

---

## 📁 Cấu trúc thư mục dự án

```
video_dubbing_ai/
├── 📄 main.py                    # Entry point chính cho giao diện CLI
├── 📄 run_web.py                 # Entry point khởi chạy FastAPI Web Server
├── 📄 requirements.txt           # Danh sách các thư viện Python cần thiết
├── 📄 setup_env.bat              # Script Windows tự động thiết lập môi trường ảo
├── 📄 .env.example               # File cấu hình mẫu cho các API Keys
│
├── 📁 config/                    # Cấu hình hệ thống
│   └── settings.py               # Các tham số cấu hình chung (Đường dẫn, thiết bị...)
│
├── 📁 pipeline/                  # Chi tiết 10 giai đoạn của Pipeline lồng tiếng
│   ├── p01_video_processor.py    # Giai đoạn 1: Chuẩn hóa định dạng video đầu vào
│   ├── p02_audio_extractor.py    # Giai đoạn 2: Trích xuất file audio từ video
│   ├── p03_speaker_detector.py   # Giai đoạn 3: Phân đoạn người nói (Speaker Diarization)
│   ├── p04_segment_creator.py    # Giai đoạn 4: Chia cắt audio theo mốc thời gian
│   ├── p05_asr.py                # Giai đoạn 5: Nhận dạng giọng nói tiếng Trung
│   ├── p06_translation.py        # Giai đoạn 6: Dịch thuật ngữ cảnh Trung -> Việt
│   ├── p07_voice_clone.py        # Giai đoạn 7: Nhân bản giọng nói giữ đặc trưng âm sắc
│   ├── p08_audio_alignment.py    # Giai đoạn 8: Căn chỉnh tốc độ nói cho khớp video gốc
│   ├── p09_lipsync.py            # Giai đoạn 9: Chạy mô hình đồng bộ khẩu hình môi
│   └── p10_renderer.py           # Giai đoạn 10: Ghép nối audio lồng tiếng vào video
│
├── 📁 services/                  # Lớp xử lý dịch vụ logic tích hợp
│   ├── video_service.py
│   ├── audio_service.py
│   ├── speaker_service.py
│   ├── asr_service.py
│   ├── translation_service.py
│   ├── voice_service.py
│   ├── lipsync_service.py
│   └── renderer_service.py
│
├── 📁 api/                       # FastAPI Backend
│   ├── server.py                 # Khởi tạo API app, mount static files
│   ├── routes.py                 # Các RESTful API Endpoints (Upload, Start, Monitor)
│   ├── websocket.py              # WebSocket phục vụ gửi tiến độ realtime
│   └── pipeline_runner.py        # Điều phối luồng chạy Pipeline trong background
│
├── 📁 web/                       # Giao diện Dashboard Frontend
│   ├── index.html                # Trang giao diện chính
│   ├── style.css                 # Thiết kế giao diện (Glassmorphism, Dark mode)
│   └── script.js                 # Logic kết nối API & WebSocket realtime
│
├── 📁 models_data/               # Lớp thực thể dữ liệu (Data models)
│   ├── segment.py                # Object Segment chứa thông tin mốc thời gian, text, audio
│   ├── speaker.py                # Thông tin đặc trưng giọng nói của từng Speaker
│   └── job.py                    # Thông tin trạng thái tiến trình (Job tracking)
│
├── 📁 utils/                     # Công cụ hỗ trợ
│   ├── gpu_manager.py            # Quản lý nạp/giải phóng bộ nhớ GPU động (VRAM)
│   ├── logger.py                 # Định dạng hiển thị log log có màu trực quan
│   ├── file_manager.py           # Quản lý dọn dẹp các tệp tin tạm thời
│   └── timer.py                  # Đo đạc thời gian thực thi của từng giai đoạn
│
├── 📁 input/                     # Thư mục chứa các video đầu vào cần lồng tiếng
├── 📁 output/                    # Thư mục lưu trữ video thành phẩm sau khi lồng tiếng
├── 📁 temp/                      # Thư mục lưu trữ các file trung gian (audio cắt nhỏ, v.v.)
├── 📁 cache/                     # Thư mục lưu cache của mô hình AI
└── 📁 models/                    # Thư mục lưu trữ trọng số (weights) của các model AI
```

---

## 🔧 Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|------------|-------------------|-------------|
| **Hệ điều hành** | Windows 10 / 11 | Windows 11 |
| **GPU** | NVIDIA GPU 6GB VRAM (hỗ trợ CUDA) | NVIDIA RTX 3060/4060/5060 8GB+ VRAM |
| **CUDA Toolkit** | CUDA 11.8 | CUDA 12.1+ (đã test mượt mà trên CUDA 12.1) |
| **Python** | 3.10 | 3.10.x |
| **Công cụ bổ trợ** | **ffmpeg** phải được cài đặt và thêm vào biến môi trường `PATH` | Phiên bản ffmpeg mới nhất |
| **Kết nối mạng** | Cần internet trong lần chạy đầu để tự động tải trọng số các mô hình | Internet băng thông rộng |

---

## 🚀 Hướng dẫn Cài đặt chi tiết

Vui lòng thực hiện theo đúng thứ tự các bước dưới đây để thiết lập môi trường chạy dự án:

### Bước 1: Chuẩn bị mã nguồn
Tải mã nguồn dự án về máy tính của bạn và truy cập vào thư mục dự án:
```bash
cd video_dubbing_ai
```

### Bước 2: Thiết lập môi trường tự động
Dự án cung cấp sẵn tệp `setup_env.bat` để tự động hóa toàn bộ quá trình cài đặt trên hệ điều hành Windows. Hãy nhấp đúp vào file hoặc chạy lệnh sau trong cmd:
```cmd
setup_env.bat
```
**Kịch bản setup_env.bat sẽ tự động thực hiện:**
1. Tạo môi trường ảo Python (`venv`) bên trong thư mục dự án để tránh xung đột thư viện hệ thống.
2. Nâng cấp trình quản lý gói `pip` lên bản mới nhất.
3. Cài đặt phiên bản **PyTorch có hỗ trợ CUDA 12.1** phù hợp cho xử lý AI trên GPU NVIDIA.
4. Cài đặt các thư viện phụ thuộc liệt kê trong [requirements.txt](file:///e:/HK8/AI/video_dubbing_ai/requirements.txt).
5. Tải về và cài đặt các thư viện đặc thù từ mã nguồn Git của bên thứ ba trong thư mục `third_party/`:
   - **Fish Speech** (Dùng cho Voice Cloning)
   - **Wav2Lip** (Dùng cho Lip Sync)
6. Tự động tạo cấu trúc các thư mục lưu trữ cần thiết (`input/`, `output/`, `temp/`, `cache/`, `models/`).
7. Tạo sẵn một file cấu hình mẫu `.env` từ `.env.example`.

### Bước 3: Cấu hình API Keys (Môi trường)
Mở file `.env` vừa được tạo ra ở thư mục gốc của dự án và điền các API Keys của bạn:

```env
# API Keys của các dịch vụ LLM / Speech
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here

# Lựa chọn công cụ dịch thuật: 'gemini' hoặc 'openai'
TRANSLATION_PROVIDER=gemini
```

**Mục đích sử dụng của các khóa:**
- **`GEMINI_API_KEY`**: Dùng để dịch các câu hội thoại tiếng Trung sang tiếng Việt bằng mô hình Gemini của Google AI Studio (Khuyên dùng - tốc độ nhanh và miễn phí/chi phí thấp). Nhận key tại: [Google AI Studio](https://aistudio.google.com/apikey).
- **`HF_TOKEN`**: Hugging Face Token dùng để tải mô hình phân tách người nói `pyannote/speaker-diarization-3.1`. Bạn cần đăng ký tài khoản Hugging Face, chấp nhận điều khoản sử dụng của model pyannote và tạo token tại: [Hugging Face Settings](https://huggingface.co/settings/tokens).
- **`OPENAI_API_KEY`**: Cung cấp nếu bạn muốn chọn `TRANSLATION_PROVIDER=openai` để dịch bằng GPT-4o/GPT-3.5 thay thế cho Gemini.

---

## 💻 Hướng dẫn chạy dự án

### Cách 1: Sử dụng Giao diện Web (Khuyên dùng)
Giao diện Web giúp bạn dễ dàng tải lên video, tuỳ chỉnh tham số và giám sát quá trình hoạt động của từng model AI một cách trực quan.

1. Kích hoạt môi trường ảo:
   ```cmd
   venv\Scripts\activate
   ```
2. Chạy tệp khởi động Web server:
   ```bash
   python run_web.py
   ```
3. Trình duyệt web của bạn sẽ tự động mở trang Dashboard tại địa chỉ: **[http://localhost:8000](http://localhost:8000)** (Nếu không tự mở, hãy sao chép địa chỉ này dán vào trình duyệt).
4. **Các bước thao tác trên Web:**
   - Kéo thả hoặc chọn một file video tiếng Trung cần dịch tại khu vực **Upload Video**.
   - Thiết lập các tùy chọn: chọn API dịch thuật (`Gemini` hoặc `OpenAI`), bật/tắt tính năng **Lip Sync** (Đồng bộ khẩu hình môi).
   - Nhấn nút **Bắt đầu Dubbing**.
   - Theo dõi tiến trình 10 bước qua WebSocket trực quan ở phần **Pipeline Monitor**, xem dung lượng VRAM thực tế đang được tiêu hao trên GPU.
   - Khi hoàn tất, một khung hiển thị video kết quả sẽ xuất hiện cho phép bạn xem trực tiếp hoặc nhấn nút **Tải Video Kết Quả** về máy.

---

### Cách 2: Sử dụng dòng lệnh CLI (Command Line)
Phù hợp khi bạn muốn xử lý tự động hàng loạt video thông qua script hoặc terminal.

1. Kích hoạt môi trường ảo:
   ```cmd
   venv\Scripts\activate
   ```
2. Đặt video cần xử lý vào thư mục `input/` (Ví dụ: `input/sample_video.mp4`).
3. Chạy lệnh thực thi cơ bản:
   ```bash
   python main.py input/sample_video.mp4
   ```
4. **Các tham số dòng lệnh tùy chọn hỗ trợ:**
   - `--output` hoặc `-o`: Chỉ định đường dẫn hoặc tên file đầu ra tùy ý (mặc định lưu tại `output/<tên_gốc>_vi.mp4`).
     ```bash
     python main.py input/sample_video.mp4 --output output/dubbed_result.mp4
     ```
   - `--skip-lipsync`: Bỏ qua bước đồng bộ khẩu hình Wav2Lip (giúp tăng tốc độ xử lý lên gấp 3-4 lần nếu bạn không quá cần chuyển động môi chính xác).
     ```bash
     python main.py input/sample_video.mp4 --skip-lipsync
     ```
   - `--log-level`: Thay đổi mức độ hiển thị log trong Terminal (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
     ```bash
     python main.py input/sample_video.mp4 --log-level DEBUG
     ```

---

## ⚡ Nguyên lý tối ưu hóa bộ nhớ GPU (VRAM 8GB)

Các mô hình học sâu như `pyannote.audio` (Speaker Detection), `SenseVoice` (ASR), `Fish Speech` (Voice Cloning) và `Wav2Lip` (Lip Sync) ngốn rất nhiều bộ nhớ đồ họa nếu chạy song song. Đối với các card đồ họa phổ thông 8GB VRAM (như RTX 3060, 4060, 5060):

Hệ thống sử dụng module `GPUManager` áp dụng nguyên tắc: **Chỉ cho phép duy nhất một mô hình AI hoạt động trên GPU tại một thời điểm.**

```
[Mô hình A chạy xong] ──> [Unload khỏi GPU] ──> [empty_cache()] ──> [Mô hình B được nạp vào GPU]
```

- Trước khi một giai đoạn bắt đầu, `GPUManager` sẽ tự động unload các mô hình cũ ra khỏi bộ nhớ VRAM.
- Gọi lệnh dọn dẹp bộ nhớ đệm `torch.cuda.empty_cache()` và `gc.collect()`.
- Nạp mô hình của giai đoạn hiện tại vào GPU.
- Cách làm này đảm bảo tiến trình hoạt động liên tục không bao giờ gặp lỗi **Out Of Memory (OOM)** trên các dòng card đồ họa thông thường.

---

## 📊 Cấu trúc thực thể dữ liệu Segment

Trong suốt chu trình chạy của pipeline, thông tin chi tiết của mỗi phân đoạn câu thoại sẽ được lưu trữ và cập nhật liên tục vào một danh sách các đối tượng `Segment`:

```python
class Segment:
    id: int                     # Số thứ tự phân đoạn (câu thoại)
    speaker: str                # Định danh người nói (ví dụ: SPEAKER_00)
    start: float                # Thời điểm bắt đầu câu nói (giây)
    end: float                  # Thời điểm kết thúc câu nói (giây)
    source_audio: str           # Đường dẫn file audio tiếng Trung đã cắt nhỏ
    reference_audio: str        # Đường dẫn file audio mẫu giọng của speaker để clone
    zh_text: str                # Nội dung tiếng Trung nhận dạng được từ ASR
    vi_text: str                # Nội dung tiếng Việt sau dịch thuật
    generated_audio: str        # Đường dẫn file audio tiếng Việt sau khi clone giọng
```

---

## 🤝 Hỗ trợ và đóng góp ý kiến

1. Tạo một bản Fork của dự án.
2. Tạo nhánh tính năng mới (`git checkout -b feature/AmazingFeature`).
3. Commit các thay đổi của bạn (`git commit -m 'Add some AmazingFeature'`).
4. Push nhánh của bạn lên Github (`git push origin feature/AmazingFeature`).
5. Tạo một yêu cầu Pull Request mới để nhóm phát triển phê duyệt.

---

## 📜 Giấy phép bản quyền (License)

Dự án này tuân theo giấy phép mã nguồn mở **MIT License**. Vui lòng tham khảo tệp `LICENSE` để biết thêm chi tiết.
