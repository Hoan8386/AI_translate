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
echo  AI Dubbing System - Environment Setup
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
echo [1/7] Tao moi truong ao (venv)...
if exist venv (
    echo   venv da ton tai, bo qua...
) else (
    python -m venv venv
    echo   Da tao venv thanh cong!
)

REM --- Activate venv ---
echo [2/7] Kich hoat venv...
call venv\Scripts\activate.bat

REM --- Upgrade pip ---
echo [3/7] Nang cap pip...
python -m pip install --upgrade pip

REM --- Install PyTorch with CUDA ---
echo [4/7] Cai dat PyTorch voi CUDA 12.1 (cho RTX 5060)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM --- Install requirements ---
echo [5/7] Cai dat cac thu vien tu requirements.txt...
pip install -r requirements.txt

REM --- Install OpenVoice V2 from git ---
echo [6/7] Cai dat OpenVoice V2...
if exist third_party\OpenVoice (
    echo   OpenVoice da ton tai, bo qua...
) else (
    mkdir third_party 2>nul
    cd third_party
    git clone https://github.com/myshell-ai/OpenVoice.git
    cd OpenVoice
    pip install -e .
    cd %PROJECT_ROOT%
)

REM --- Install Wav2Lip from git ---
echo [7/7] Cai dat Wav2Lip...
if exist third_party\Wav2Lip (
    echo   Wav2Lip da ton tai, bo qua...
) else (
    cd third_party
    git clone https://github.com/Rudrabha/Wav2Lip.git
    cd Wav2Lip
    pip install -r requirements.txt 2>nul
    cd %PROJECT_ROOT%
)

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
    echo # Translation provider: gemini or openai>> .env
    echo TRANSLATION_PROVIDER=gemini>> .env
    echo.
    echo [INFO] Da tao file .env - Vui long dien API keys!
)

echo.
echo ========================================
echo  Setup hoan tat!
echo ========================================
echo.
echo Buoc tiep theo:
echo   1. Chinh sua file .env voi API keys cua ban
echo   2. Kich hoat venv: venv\Scripts\activate
echo   3. Chay: python main.py input\video.mp4
echo.
pause
