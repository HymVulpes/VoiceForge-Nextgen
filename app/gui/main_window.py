"""
VoiceForge-Nextgen Launcher
File: launcher.py
Compile:
    pyinstaller launcher.py --onefile --windowed --name=Start
"""

import sys
import os
from pathlib import Path
import subprocess
import logging

# =========================
# Resolve ROOT_DIR
# =========================
if getattr(sys, "frozen", False):
    ROOT_DIR = Path(sys.executable).parent
else:
    ROOT_DIR = Path(__file__).parent


# =========================
# Logging setup
# =========================
log_dir = ROOT_DIR / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "launcher.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VoiceForgeLauncher")


# =========================
# Environment check
# =========================
def check_environment():
    logger.info("Checking environment...")

    # Prefer venv Python
    venv_python = ROOT_DIR / "venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        logger.info(f"Using venv Python: {venv_python}")
        return str(venv_python)

    # Fallback: system python
    try:
        result = subprocess.run(
            ["python", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"Using system Python: {result.stdout.strip()}")
            return "python"
    except FileNotFoundError:
        pass

    logger.error("Python not found")
    return None


# =========================
# Diagnostic runner
# =========================
def run_diagnostic(python_exe):
    logger.info("Running diagnostic tool...")

    diagnostic_script = ROOT_DIR / "diagnostic_tool.py"
    if not diagnostic_script.exists():
        logger.warning("diagnostic_tool.py not found → skipping diagnostic")
        return True

    try:
        result = subprocess.run(
            [python_exe, str(diagnostic_script)],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            logger.info("✓ Diagnostic PASSED")
        else:
            logger.warning("⚠ Diagnostic returned non-zero exit code")
            if result.stdout:
                logger.warning(result.stdout.strip())
            if result.stderr:
                logger.warning(result.stderr.strip())

        # IMPORTANT:
        # Diagnostic failure does NOT crash launcher
        return True

    except subprocess.TimeoutExpired:
        logger.error("Diagnostic timeout (60s)")
        return False

    except Exception as e:
        logger.error(f"Diagnostic execution error: {e}")
        return False


# =========================
# GUI starter
# =========================
def start_gui(python_exe):
    logger.info("Starting GUI...")

    gui_script = ROOT_DIR / "app" / "gui" / "main_window.py"
    if not gui_script.exists():
        logger.error(f"GUI script not found: {gui_script}")
        return False

    try:
        subprocess.Popen(
            [python_exe, str(gui_script)],
            cwd=ROOT_DIR
        )
        logger.info("GUI process started successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to start GUI: {e}")
        return False


# =========================
# Main
# =========================
def main():
    logger.info("=" * 60)
    logger.info("VoiceForge-Nextgen Launcher")
    logger.info("=" * 60)

    # Step 1: environment
    python_exe = check_environment()
    if not python_exe:
        input("Environment error. Press Enter to exit...")
        return 1

    # Step 2: diagnostic
    if not run_diagnostic(python_exe):
        choice = input("Diagnostic failed. Continue anyway? (y/n): ")
        if choice.lower() != "y":
            return 1

    # Step 3: GUI
    if not start_gui(python_exe):
        input("Failed to start GUI. Press Enter to exit...")
        return 1

    logger.info("VoiceForge launched successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
