"""
╔══════════════════════════════════════════════════════════════╗
║  download_models.py                                          ║
║  ──────────────────────────────────────────────────────────  ║
║  Tải tất cả AI models cần thiết cho hệ thống AI Dubbing     ║
║                                                              ║
║  Models cần tải:                                             ║
║  1. pyannote/speaker-diarization-3.1  (Stage 3 - ~2GB)      ║
║  2. openai/whisper-large-v3           (Stage 5 - ~3GB)      ║
║  3. Qwen/Qwen2.5-1.5B-Instruct       (Stage 6 - ~3GB)      ║
║  4. Fish Speech S2-Pro checkpoints    (Stage 7 - ~4GB)      ║
║  5. Wav2Lip GAN checkpoint            (Stage 9 - ~500MB)    ║
║  6. face_detection (s3fd)             (Stage 9 - ~90MB)     ║
║                                                              ║
║  Chạy: python download_models.py                            ║
║  Hoặc: python download_models.py --model wav2lip            ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
import subprocess
import urllib.request
import shutil
from pathlib import Path
from typing import Optional

# ── Màu sắc terminal ──────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(msg):   print(f"{GREEN}  ✅ {msg}{RESET}")
def info(msg): print(f"{CYAN}  ℹ  {msg}{RESET}")
def warn(msg): print(f"{YELLOW}  ⚠  {msg}{RESET}")
def err(msg):  print(f"{RED}  ❌ {msg}{RESET}")
def step(msg): print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}  {msg}{RESET}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}\n{BOLD}  {msg}{RESET}\n{BOLD}{CYAN}{'═'*60}{RESET}")


# ── Đường dẫn gốc ──────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent
MODELS_DIR     = Path(os.getenv("MODELS_DIR", "C:/models"))
THIRD_PARTY    = PROJECT_ROOT / "third_party"

# Đường dẫn cụ thể
PYANNOTE_DIR   = MODELS_DIR / "pyannote"
WHISPER_DIR    = MODELS_DIR / "whisper-large-v3"
QWEN_DIR       = THIRD_PARTY / "qwen2.5"
FISH_DIR       = THIRD_PARTY / "fish-speech"
FISH_CKP_DIR   = FISH_DIR / "checkpoints" / "s2-pro"
WAV2LIP_DIR    = THIRD_PARTY / "Wav2Lip"
WAV2LIP_CKP    = MODELS_DIR / "wav2lip"


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_package(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def download_file(url: str, dest: Path, desc: str = ""):
    """Tải file với thanh tiến trình."""
    ensure_dir(dest.parent)
    if dest.exists():
        ok(f"Đã có: {dest.name}")
        return

    info(f"Đang tải {desc or dest.name} ...")
    info(f"  URL: {url}")

    def _progress(count, block_size, total_size):
        if total_size <= 0:
            return
        done = min(count * block_size, total_size)
        pct  = done * 100 / total_size
        bar  = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        mb_done  = done / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  [{bar}] {pct:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after progress bar
        ok(f"Tải xong: {dest}")
    except Exception as e:
        print()
        err(f"Tải thất bại: {e}")
        dest.unlink(missing_ok=True)
        raise


def hf_download(repo_id: str, dest: Path, token: Optional[str] = None,
                ignore_patterns: list = None):
    """Tải model từ HuggingFace Hub (dùng huggingface_hub)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        err("Chưa cài huggingface_hub. Chạy: pip install huggingface-hub")
        sys.exit(1)

    ensure_dir(dest)
    info(f"Tải {repo_id} → {dest}")

    kwargs = {
        "repo_id": repo_id,
        "local_dir": str(dest),
        "local_dir_use_symlinks": False,
    }
    if token:
        kwargs["token"] = token
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns

    try:
        snapshot_download(**kwargs)
        ok(f"Hoàn thành: {repo_id}")
    except Exception as e:
        err(f"Tải thất bại ({repo_id}): {e}")
        raise


def run_git_clone(url: str, dest: Path):
    """Git clone repo."""
    if dest.exists() and any(dest.iterdir()):
        ok(f"Đã có: {dest}")
        return
    ensure_dir(dest.parent)
    info(f"Git clone: {url} → {dest}")
    result = subprocess.run(
        ["git", "clone", "--depth=1", url, str(dest)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        err(f"Git clone thất bại:\n{result.stderr}")
        raise RuntimeError(result.stderr)
    ok(f"Clone xong: {dest.name}")


# ══════════════════════════════════════════════════════════════
# MODEL 1: pyannote/speaker-diarization-3.1
# ══════════════════════════════════════════════════════════════

def download_pyannote(hf_token: str):
    step("MODEL 1: pyannote/speaker-diarization-3.1")
    print(f"  {DIM}Dùng cho Stage 3 — Speaker Detection (~2.2 GB VRAM){RESET}")
    print(f"  Đích: {PYANNOTE_DIR}")

    if not hf_token:
        warn("HF_TOKEN chưa được cung cấp!")
        warn("pyannote yêu cầu bạn phải:")
        warn("  1. Tạo tài khoản https://huggingface.co")
        warn("  2. Accept license tại:")
        warn("     https://huggingface.co/pyannote/speaker-diarization-3.1")
        warn("  3. Tạo token tại https://hf.co/settings/tokens")
        warn("  4. Truyền vào: --hf-token YOUR_TOKEN")
        warn("Bỏ qua pyannote do chưa có token.")
        return

    hf_download(
        "pyannote/speaker-diarization-3.1",
        PYANNOTE_DIR,
        token=hf_token,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"]
    )


# ══════════════════════════════════════════════════════════════
# MODEL 2: openai/whisper-large-v3  (Stage 5 - ASR)
# ══════════════════════════════════════════════════════════════

def download_whisper():
    step("MODEL 2: openai/whisper-large-v3")
    print(f"  {DIM}Dùng cho Stage 5 — Chinese ASR (~6 GB disk, ~3 GB VRAM){RESET}")
    print(f"  Đích: {WHISPER_DIR}")

    hf_download(
        "openai/whisper-large-v3",
        WHISPER_DIR,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "*.ot"]
    )


# ══════════════════════════════════════════════════════════════
# MODEL 3: Qwen2.5-1.5B-Instruct  (Stage 6 - Translation)
# ══════════════════════════════════════════════════════════════

def download_qwen():
    step("MODEL 3: Qwen/Qwen2.5-1.5B-Instruct")
    print(f"  {DIM}Dùng cho Stage 6 — ZH→VI Translation Local (~3 GB){RESET}")
    print(f"  Đích: {QWEN_DIR}")

    ensure_dir(THIRD_PARTY)
    hf_download(
        "Qwen/Qwen2.5-1.5B-Instruct",
        QWEN_DIR,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"]
    )


# ══════════════════════════════════════════════════════════════
# MODEL 4: Fish Speech S2-Pro  (Stage 7 - Voice Cloning)
# ══════════════════════════════════════════════════════════════

def download_fish_speech(hf_token: Optional[str] = None):
    step("MODEL 4: Fish Speech S2-Pro (Voice Cloning)")
    print(f"  {DIM}Dùng cho Stage 7 — Voice Cloning (~4-6 GB VRAM){RESET}")

    # 4a. Clone fish-speech repo
    fish_repo_url = "https://github.com/fishaudio/fish-speech.git"
    if not (FISH_DIR / "fish_speech").exists():
        run_git_clone(fish_repo_url, FISH_DIR)
    else:
        ok(f"Fish Speech repo đã có: {FISH_DIR}")

    # 4b. Tải checkpoints từ HuggingFace
    print(f"\n  Tải Fish Speech checkpoints → {FISH_CKP_DIR}")
    ensure_dir(FISH_CKP_DIR)

    hf_download(
        "fishaudio/fish-speech-1.5",
        FISH_CKP_DIR,
        token=hf_token,
        ignore_patterns=["*.msgpack", "*.h5"]
    )

    # 4c. Đảm bảo firefly-gan decoder tồn tại
    decoder_name = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    decoder_path = FISH_CKP_DIR / decoder_name
    if not decoder_path.exists():
        warn(f"Không tìm thấy decoder: {decoder_name}")
        warn("Thử tải thủ công từ HuggingFace...")
        hf_download(
            "fishaudio/fish-speech-1.5",
            FISH_CKP_DIR,
            token=hf_token,
        )

    if decoder_path.exists():
        ok(f"Decoder tồn tại: {decoder_path.name}")
    else:
        warn(f"Decoder chưa tìm thấy, kiểm tra lại thủ công: {FISH_CKP_DIR}")


# ══════════════════════════════════════════════════════════════
# MODEL 5: Wav2Lip GAN  (Stage 9 - Lip Sync)
# ══════════════════════════════════════════════════════════════

WAV2LIP_GAN_URL = (
    "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/"
    "Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtFfbnu9A?download=1"
)
# Backup URL (Google Drive direct)
WAV2LIP_GAN_URL_BACKUP = (
    "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2Lip/wav2lip_gan.pth"
)

S3FD_URL = (
    "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
)

def download_wav2lip():
    step("MODEL 5: Wav2Lip GAN checkpoint")
    print(f"  {DIM}Dùng cho Stage 9 — Lip Sync (~4 GB VRAM){RESET}")

    # 5a. Clone Wav2Lip repo
    wav2lip_repo = "https://github.com/Rudrabha/Wav2Lip.git"
    if not (WAV2LIP_DIR / "inference.py").exists():
        run_git_clone(wav2lip_repo, WAV2LIP_DIR)
    else:
        ok(f"Wav2Lip repo đã có: {WAV2LIP_DIR}")

    # 5b. Tải wav2lip_gan.pth vào C:/models/wav2lip/
    ensure_dir(WAV2LIP_CKP)
    gan_path = WAV2LIP_CKP / "wav2lip_gan.pth"

    if not gan_path.exists():
        info("Tải wav2lip_gan.pth ...")
        # Thử URL chính từ HuggingFace (đáng tin cậy hơn)
        try:
            download_file(WAV2LIP_GAN_URL_BACKUP, gan_path, "wav2lip_gan.pth")
        except Exception:
            warn("URL chính thất bại, thử backup URL...")
            try:
                download_file(WAV2LIP_GAN_URL, gan_path, "wav2lip_gan.pth (backup)")
            except Exception as e:
                err(f"Không thể tải wav2lip_gan.pth: {e}")
                print(f"\n  {YELLOW}Tải thủ công:{RESET}")
                print(f"  URL: {WAV2LIP_GAN_URL_BACKUP}")
                print(f"  Lưu vào: {gan_path}")
    else:
        ok(f"wav2lip_gan.pth đã có ({gan_path.stat().st_size / (1024*1024):.1f} MB)")

    # 5c. Copy checkpoint vào Wav2Lip/checkpoints/ (để inference.py tìm thấy)
    local_ckp = WAV2LIP_DIR / "checkpoints"
    ensure_dir(local_ckp)
    local_gan = local_ckp / "wav2lip_gan.pth"
    if not local_gan.exists() and gan_path.exists():
        shutil.copy2(gan_path, local_gan)
        ok(f"Copy → {local_gan}")

    # 5d. Tải face detection model (s3fd)
    face_det_dir = WAV2LIP_DIR / "face_detection" / "detection" / "sfd"
    ensure_dir(face_det_dir)
    s3fd_path = face_det_dir / "s3fd.pth"

    if not s3fd_path.exists():
        try:
            download_file(S3FD_URL, s3fd_path, "s3fd face detection model")
        except Exception as e:
            err(f"Không thể tải s3fd.pth: {e}")
            print(f"  {YELLOW}Tải thủ công:{RESET} {S3FD_URL} → {s3fd_path}")
    else:
        ok(f"s3fd.pth đã có ({s3fd_path.stat().st_size / (1024*1024):.1f} MB)")


# ══════════════════════════════════════════════════════════════
# VERIFY: Kiểm tra tất cả models
# ══════════════════════════════════════════════════════════════

def verify_all():
    step("KIỂM TRA TẤT CẢ MODELS")

    checks = [
        ("pyannote (Stage 3)",
         PYANNOTE_DIR / "config.yaml"),
        ("Whisper large-v3 (Stage 5)",
         WHISPER_DIR / "config.json"),
        ("Qwen2.5 (Stage 6)",
         QWEN_DIR / "config.json"),
        ("Fish Speech config (Stage 7)",
         FISH_DIR / "fish_speech" / "__init__.py"),
        ("Fish Speech decoder (Stage 7)",
         FISH_CKP_DIR / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
        ("Wav2Lip GAN (Stage 9)",
         WAV2LIP_CKP / "wav2lip_gan.pth"),
        ("s3fd face detection (Stage 9)",
         WAV2LIP_DIR / "face_detection" / "detection" / "sfd" / "s3fd.pth"),
    ]

    all_ok = True
    for name, path in checks:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            ok(f"{name:<40} {size_mb:7.1f} MB  ← {path.parent.name}/{path.name}")
        else:
            err(f"{name:<40} THIẾU   ← {path}")
            all_ok = False

    print()
    if all_ok:
        print(f"{GREEN}{BOLD}  ✅ TẤT CẢ MODELS ĐÃ SẴN SÀNG!{RESET}")
    else:
        print(f"{RED}{BOLD}  ❌ MỘT SỐ MODELS CHƯA TẢI XONG — Chạy lại script{RESET}")

    return all_ok


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Tải tất cả AI models cho hệ thống AI Dubbing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python download_models.py                          # Tải tất cả
  python download_models.py --hf-token hf_xxxxx     # Tải tất cả với HF token
  python download_models.py --model wav2lip          # Chỉ tải Wav2Lip
  python download_models.py --model pyannote --hf-token hf_xxx
  python download_models.py --verify                 # Chỉ kiểm tra

Models:
  all        Tải tất cả (mặc định)
  pyannote   Speaker diarization model
  whisper    Chinese ASR model
  qwen       Translation model (Qwen2.5-1.5B)
  fish       Voice cloning (Fish Speech S2-Pro)
  wav2lip    Lip sync model (Wav2Lip GAN)
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="all",
        choices=["all", "pyannote", "whisper", "qwen", "fish", "wav2lip"],
        help="Model cần tải (mặc định: all)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN", ""),
        help="HuggingFace access token (hoặc set env HF_TOKEN)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Chỉ kiểm tra, không tải"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(MODELS_DIR),
        help=f"Thư mục lưu models (mặc định: {MODELS_DIR})"
    )

    args = parser.parse_args()

    # Override models dir nếu được chỉ định
    global MODELS_DIR, PYANNOTE_DIR, WHISPER_DIR, WAV2LIP_CKP
    if args.models_dir != str(MODELS_DIR):
        MODELS_DIR    = Path(args.models_dir)
        PYANNOTE_DIR  = MODELS_DIR / "pyannote"
        WHISPER_DIR   = MODELS_DIR / "whisper-large-v3"
        WAV2LIP_CKP   = MODELS_DIR / "wav2lip"
    print(f"  {BOLD}Models dir:{RESET} {MODELS_DIR}")
    print(f"  {BOLD}Third party:{RESET} {THIRD_PARTY}")
    print(f"  {BOLD}HF Token:{RESET}   {'✓ Đã cấu hình' if args.hf_token else '✗ Chưa có (pyannote sẽ bị bỏ qua)'}")

    # ── Verify only ────────────────────────────────────────────
    if args.verify:
        verify_all()
        return

    # ── Kiểm tra dependencies ──────────────────────────────────
    step("KIỂM TRA DEPENDENCIES")
    deps = {
        "huggingface_hub": "huggingface-hub",
        "torch":           "torch",
        "torchaudio":      "torchaudio",
        "transformers":    "transformers",
        "git":             None,  # system tool
    }
    for pkg, pip_name in deps.items():
        if pkg == "git":
            result = subprocess.run(["git", "--version"], capture_output=True)
            if result.returncode == 0:
                ok("git")
            else:
                err("git chưa được cài đặt! Tải từ https://git-scm.com")
                sys.exit(1)
        elif check_package(pkg):
            ok(pkg)
        else:
            warn(f"{pkg} chưa cài. Đang cài: pip install {pip_name}")
            subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=True)

    # ── Tải models ─────────────────────────────────────────────
    model = args.model
    token = args.hf_token

    try:
        if model in ("all", "pyannote"):
            download_pyannote(token)

        if model in ("all", "whisper"):
            download_whisper()

        if model in ("all", "qwen"):
            download_qwen()

        if model in ("all", "fish"):
            download_fish_speech(token)

        if model in ("all", "wav2lip"):
            download_wav2lip()

    except KeyboardInterrupt:
        print(f"\n{YELLOW}  ⚠ Bị ngắt bởi người dùng{RESET}")
        sys.exit(130)
    except Exception as e:
        err(f"Lỗi: {e}")
        sys.exit(1)

    # ── Kết quả ────────────────────────────────────────────────
    verify_all()

    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}  Bước tiếp theo:{RESET}")
    print(f"  1. Cấu hình .env file:")
    print(f"     HF_TOKEN=your_token_here")
    print(f"     GEMINI_API_KEY=your_key      (nếu dùng Gemini dịch)")
    print(f"     MODELS_DIR={MODELS_DIR}")
    print(f"  2. Chạy web server:")
    print(f"     python run_web.py")
    print(f"  3. Mở trình duyệt: http://localhost:8000")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}\n")


if __name__ == "__main__":
    main()
