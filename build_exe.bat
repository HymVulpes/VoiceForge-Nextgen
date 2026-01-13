@echo off
REM ============================================================
REM build_exe.bat - VoiceForge-Nextgen Build Script (Minimal & Stable)
REM Version: Final Clean - No version check, no parse errors
REM ============================================================

echo Starting build of start.exe for VoiceForge-Nextgen...
echo ====================================================

REM Get script directory safely (handles spaces in path)
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
echo Project root: %PROJECT_ROOT%

REM Activate venv from project root
call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
if ERRORLEVEL 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo Virtual environment activated

REM Add ffmpeg to PATH if folder exists
set "FFMPEG_PATH=%PROJECT_ROOT%\tool\ffmpeg-8.0.1-essentials_build\bin"
if exist "%FFMPEG_PATH%" (
    set "PATH=%FFMPEG_PATH%;%PATH%"
    echo Added ffmpeg to PATH
) else (
    echo WARNING: ffmpeg folder not found - skipping
)

REM Icon option
set "ICON_OPTION="
if exist "%PROJECT_ROOT%\icon.ico" (
    set "ICON_OPTION=--icon=icon.ico"
    echo Using icon.ico
)

REM Run PyInstaller
echo Running PyInstaller...
pyinstaller launcher.py ^
    --onefile ^
    --windowed ^
    --name=start ^
    %ICON_OPTION% ^
    --add-data "app;app" ^
    --add-data "config.yml;." ^
    --hidden-import=PyQt6 ^
    --hidden-import=PyQt6.sip ^
    --hidden-import=PyQt6.QtWidgets ^
    --hidden-import=PyQt6.QtGui ^
    --hidden-import=PyQt6.QtCore ^
    --hidden-import=torch ^
    --hidden-import=numpy ^
    --hidden-import=yaml ^
    --clean

if ERRORLEVEL 1 (
    echo ERROR: PyInstaller build failed!
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo Executable: dist\start.exe
echo ====================================================
pause