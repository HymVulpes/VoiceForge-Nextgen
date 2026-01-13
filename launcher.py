"""
VoiceForge-Nextgen Launcher
File: launcher.py

Build:
    pyinstaller launcher.py --onefile --windowed --name=Start
"""

import sys
import logging
from pathlib import Path

# =========================
# Resolve ROOT_DIR
# =========================
if getattr(sys, "frozen", False):
    ROOT_DIR = Path(sys.executable).parent
else:
    ROOT_DIR = Path(__file__).parent

sys.path.insert(0, str(ROOT_DIR))

# =========================
# Logging
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
# Diagnostic
# =========================
def run_diagnostic():
    """
    Run diagnostic_tool.py
    IMPORTANT:
    - Exit code != 0 is EXPECTED
    - Diagnostic MUST NOT stop GUI
    """
    diagnostic_path = ROOT_DIR / "diagnostic_tool.py"

    if not diagnostic_path.exists():
        logger.warning("diagnostic_tool.py not found → skipping")
        return True

    try:
        import diagnostic_tool

        diagnostic_tool.main()
        logger.info("Diagnostic finished with exit code 0")
        return True

    except SystemExit as e:
        # 🔑 KEY POINT: NUỐT EXIT CODE
        if e.code == 0:
            logger.info("Diagnostic exit code 0 (OK)")
        else:
            logger.warning(
                f"Diagnostic exit code {e.code} – continuing anyway"
            )
        return False

    except Exception:
        logger.exception("Diagnostic crashed unexpectedly")
        return False


# =========================
# GUI
# =========================
def launch_gui():
    """
    Launch PyQt GUI.
    GUI controls its own event loop & sys.exit
    """
    try:
        from app.gui.main_window import main
        main()
    except Exception:
        logger.exception("Failed to launch GUI")
        sys.exit(1)


# =========================
# Main
# =========================
def main():
    logger.info("=" * 60)
    logger.info("VoiceForge-Nextgen Launcher")
    logger.info("=" * 60)

    diagnostic_ok = run_diagnostic()

    if not diagnostic_ok:
        logger.warning(
            "Running GUI in DEGRADED MODE (diagnostic failed)"
        )

    launch_gui()


if __name__ == "__main__":
    main()
