@echo off
REM ============================================
REM Chinese Video To Vietnamese AI Dubbing System
REM Virtual Environment Setup Script
REM ============================================
REM Tao moi truong ao, cai dat dependencies
REM Khong luu vao may, chi trong venv
REM ============================================

echo.
echo ========================================
echo   AI Dubbing System - Environment Setup
echo ========================================
echo.

REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python khong duoc tim thay. Vui long cai dat Python 3.10+
    pause
    exit /b 1
)

REM --- Set project root ---
set PROJECT_ROOT=%~dp0
cd /d %PROJECT_ROOT%

REM --- Create virtual environment ---
echo [1/8] Tao moi truong ao (venv)...
if exist venv (
    echo   venv da ton tai, bo qua...
) else (
    python -m venv venv
    echo   Da tao venv thanh cong!
)

REM --- Activate venv ---
echo [2/8] Kich hoat venv...
call venv\Scripts\activate.bat

REM --- Upgrade pip ---
echo [3/8] Nang cap pip...
python -m pip install --upgrade pip

REM --- Install PyTorch with CUDA ---
echo [4/8] Cai dat PyTorch voi CUDA 12.1 (cho RTX 5060)...
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
REM --- Install requirements ---
echo [5/8] Cai dat cac thu vien tu requirements.txt...
pip install -r requirements.txt

REM --- Install Fish Speech from git ---
echo [6/8] Cai dat Fish Speech...
if exist third_party\fish-speech (
    echo   Fish Speech da ton tai, bo qua...
) else (
    mkdir third_party 2>nul
    cd third_party
    git clone https://github.com/fishaudio/fish-speech.git
    cd fish-speech
    pip install -e .
    cd %PROJECT_ROOT%
)

REM --- Install Wav2Lip from git ---
echo [7/8] Cai dat Wav2Lip...
if exist third_party\Wav2Lip (
    echo   Wav2Lip da ton tai, bo qua...
) else (
    cd third_party
    git clone https://github.com/Rudrabha/Wav2Lip.git
    cd Wav2Lip
    pip install -r requirements.txt 2>nul
    cd %PROJECT_ROOT%
)

REM --- Download Local Translation Model ---
echo [8/8] Dang tu dong tai Model dich thuat Local (Qwen2.5-1.5B)...
echo.
REM Tao file python tam thoi de download model an toan, chi cache vao o cung khong load len RAM
echo import os > download_models.py
echo from huggingface_hub import snapshot_download >> download_models.py
echo model_name = "Qwen/Qwen2.5-1.5B-Instruct" >> download_models.py
echo print(f"=== Bat dau tai: {model_name} (Khoang 3.5GB) ===") >> download_models.py
echo print("He thong se tu dong hiển thị thanh tien trinh download duoi day...") >> download_models.py
echo try: >> download_models.py
echo     snapshot_download(repo_id=model_name, local_files_only=False) >> download_models.py
echo     print("=== Tai Model Qwen2.5 Local ve o cung hoan tat! ===") >> download_models.py
echo except Exception as e: >> download_models.py
echo     print(f"[ERROR] Co loi xay ra khi dang tai model tu HuggingFace: {e}") >> download_models.py

REM Chay file download trong venv
python download_models.py
REM Xoa file tam thoi sau khi tai xong de sach se project
del download_models.py

REM --- Create necessary directories ---
echo.
echo Tao cac thu muc can thiet...
mkdir input 2>nul
mkdir output 2>nul
mkdir temp 2>nul
mkdir cache 2>nul
mkdir models 2>nul

REM --- Create .env template ---
if not exist .env (
    echo # API Keys> .env
    echo GEMINI_API_KEY=your_gemini_api_key_here>> .env
    echo OPENAI_API_KEY=your_openai_api_key_here>> .env
    echo HF_TOKEN=your_huggingface_token_here>> .env
    echo.>> .env
    echo # Translation provider: gemini, openai, or local>> .env
    echo TRANSLATION_PROVIDER=local>> .env
    echo.
    echo [INFO] Da tao file .env - Vui long dien API keys hoac kiem tra option!
)

echo.
echo ========================================
echo   Setup hoan tat! Model o cung da san sang.
echo ========================================
echo.
echo Buoc tiep theo:
echo   1. Chinh sua file .env (Mac dinh TRANSLATION_PROVIDER da set la local)
echo   2. Kich hoat venv: venv\Scripts\activate
echo   3. Chay thu nghiem Pipeline: python main.py input\video.mp4
echo.
pause