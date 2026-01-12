@echo off
REM VoiceForge-Nextgen Setup Script
REM Run this to setup the environment

echo ============================================================
echo VoiceForge-Nextgen Setup
echo ============================================================
echo.

REM Check Python version
echo [1/5] Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.12.9 from python.org
    pause
    exit /b 1
)
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    echo Virtual environment created
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [5/5] Installing dependencies...
pip install -r requirements.txt

REM Check for CUDA
echo.
echo Checking CUDA availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
echo.

REM Create necessary directories
echo Creating directory structure...
if not exist "SampleVoice" mkdir SampleVoice
if not exist "logs" mkdir logs
if not exist "logs\snapshots" mkdir logs\snapshots
if not exist "native" mkdir native
echo.

REM Run diagnostic
echo ============================================================
echo Running diagnostic tool...
echo ============================================================
python diagnostic_tool.py

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Place your RVC model files (.pth, .index) in SampleVoice\ folder
echo 2. Ensure Virtual Audio Cable is installed
echo 3. Run: python app\main.py
echo.
pause