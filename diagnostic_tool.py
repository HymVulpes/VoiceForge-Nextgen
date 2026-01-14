"""
VoiceForge Diagnostic Tool
Comprehensive system check before running main application
Run this first: python diagnostic_tool.py
"""

import sys
import json
from pathlib import Path

# Add app directory
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# ============================================================
# JSON SAFETY HELPER  (ADDED)
# ============================================================


def make_json_safe(obj):
    """
    Convert non-JSON-serializable objects (numpy, Path, etc.)
    into safe Python primitives.
    """
    import numpy as np

    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return obj


# -------------------------------
# CHECK FUNCTIONS
# -------------------------------


def check_python_version():
    print("\n" + "=" * 60)
    print("1. PYTHON VERSION")
    print("=" * 60)

    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major != 3 or version.minor != 12:
        print("WARNING: Python 3.12.9 recommended")
        return False

    print("[OK] Python version OK")
    return True


def check_dependencies():
    print("\n" + "=" * 60)
    print("2. DEPENDENCIES")
    print("=" * 60)

    required = {
        "torch": "PyTorch",
        "pyaudio": "PyAudio",
        "numpy": "NumPy",
        "sqlalchemy": "SQLAlchemy",
        "colorlog": "ColorLog",
    }

    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"[OK] {name} installed")
        except ImportError:
            print(f"[MISSING] {name} NOT installed")
            all_ok = False

    return all_ok


def check_cuda():
    print("\n" + "=" * 60)
    print("3. CUDA / GPU")
    print("=" * 60)

    try:
        import torch

        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.version.cuda}")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
            return True

        print("WARNING: CUDA not available - will use CPU")
        return False

    except Exception as e:
        print(f"[ERROR] Error checking CUDA: {e}")
        return False


def check_audio_devices():
    print("\n" + "=" * 60)
    print("4. AUDIO DEVICES")
    print("=" * 60)

    try:
        import pyaudio

        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        print(f"Found {device_count} audio devices:")

        input_ok = False
        output_ok = False
        vac_found = False

        for i in range(device_count):
            info = p.get_device_info_by_index(i)
            name = info["name"]

            if info["maxInputChannels"] > 0:
                input_ok = True
                print(f"  [IN  {i}] {name}")

            if info["maxOutputChannels"] > 0:
                output_ok = True
                keywords = ["CABLE", "VoiceMeeter", "Virtual"]
                if any(k.lower() in name.lower() for k in keywords):
                    print("    [OK] Virtual Audio Cable detected!")
                    vac_found = True

        p.terminate()

        if not vac_found:
            print("WARNING: No Virtual Audio Cable found")

        return input_ok and output_ok

    except Exception as e:
        print(f"[ERROR] Error checking audio devices: {e}")
        return False


def check_ai_modules():
    print("\n" + "=" * 60)
    print("5. AI MODULES")
    print("=" * 60)

    modules = {
        "torch": "PyTorch (core)",
        "torchaudio": "TorchAudio",
        "torchvision": "TorchVision",
        "faiss": "FAISS (optional)",
        "librosa": "Librosa (audio)",
    }

    all_ok = True
    for module, desc in modules.items():
        try:
            __import__(module)
            print(f"[OK] {desc}")
        except ImportError:
            print(f"[MISSING] {desc}")
            if module in ("torch",):
                all_ok = False

    print("\nChecking local core modules (file check only)...")
    core_files = [
        "app/core/model_loader.py",
        "app/core/rvc_engine.py",
        "app/core/feature_cache.py",
        "app/core/model_cache.py",
    ]

    for f in core_files:
        if (ROOT_DIR / f).exists():
            print(f"[OK] {f} exists")
        else:
            print(f"[MISSING] {f}")
            all_ok = False

    assets = ROOT_DIR / "app" / "core" / "assets"
    if assets.exists():
        hubert = list(assets.glob("hubert_base*.pt"))
        if hubert:
            print(f"[OK] HuBERT model: {hubert[0].name}")

    return all_ok


def check_directory_structure():
    print("\n" + "=" * 60)
    print("6. DIRECTORY STRUCTURE")
    print("=" * 60)

    dirs = [
        "app",
        "app/core",
        "app/audio",
        "app/db",
        "app/utils",
        "SampleVoice",
        "logs",
        "logs/snapshots",
    ]

    for d in dirs:
        path = ROOT_DIR / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {d}")

    return True


def test_mock_audio_passthrough():
    print("\n" + "=" * 60)
    print("7. MOCK AUDIO TEST")
    print("=" * 60)

    try:
        import numpy as np

        sr = 48000
        samples = int(sr * 0.1)
        audio = np.random.randn(samples).astype(np.float32) * 0.1

        print(f"[OK] Generated mock audio: {samples} samples")
        print(f"  RMS: {np.sqrt((audio ** 2).mean()):.6f}")
        return True

    except Exception as e:
        print(f"[ERROR] Mock audio test failed: {e}")
        return False


# ============================================================
# ADVANCED TESTS (FROM PHASE 4)
# ============================================================


def test_real_inference():
    print("\n" + "=" * 60)
    print("8. REAL INFERENCE TEST (Mock)")
    print("=" * 60)

    try:
        from tests.fixtures.mock_models import create_mock_model, create_test_audio
        from app.core.rvc_inferencer import RVCInferencer

        model = create_mock_model()
        print("[OK] Mock model created")

        inferencer = RVCInferencer(
            model_data=model,
            device="cpu",
            use_fp16=False,
        )
        print("[OK] Inferencer created")

        audio = create_test_audio(duration_s=0.1)
        out = inferencer.infer(audio, pitch_shift=0)

        print(f"[OK] Inference successful: {len(out)} samples")
        return True

    except Exception as e:
        print(f"[ERROR] Inference test failed: {e}")
        return False


def test_feature_extraction():
    print("\n" + "=" * 60)
    print("9. FEATURE EXTRACTION TEST")
    print("=" * 60)

    try:
        import numpy as np
        from app.core.feature_extractor import F0Extractor
        from tests.fixtures.mock_models import create_test_audio

        audio = create_test_audio(duration_s=1.0, frequency=440.0)
        extractor = F0Extractor(method="harvest", sr=48000)
        f0 = extractor.extract(audio)

        print(f"[OK] F0 extracted: {len(f0)} frames")

        voiced = f0[f0 > 0]
        if len(voiced):
            print(f"  Mean F0: {np.mean(voiced):.1f} Hz")

        return True

    except Exception as e:
        print(f"[ERROR] Feature extraction test failed: {e}")
        return False


def benchmark_latency():
    print("\n" + "=" * 60)
    print("10. LATENCY BENCHMARK")
    print("=" * 60)

    try:
        import time
        import numpy as np
        from app.core.audio_preprocessor import AudioPreprocessor
        from app.core.audio_postprocessor import AudioPostprocessor
        from tests.fixtures.mock_models import create_test_audio

        pre = AudioPreprocessor(target_sr=48000)
        post = AudioPostprocessor()
        audio = create_test_audio(duration_s=0.1)

        lat = []
        for _ in range(10):
            t0 = time.perf_counter()
            out = post.process(pre.process(audio, 48000) * 0.95)
            lat.append((time.perf_counter() - t0) * 1000)

        avg = np.mean(lat)
        print(f"[OK] Average latency: {avg:.2f} ms")

        return avg <= 20.0

    except Exception as e:
        print(f"[ERROR] Latency benchmark failed: {e}")
        return False


# -------------------------------
# GENERATE REPORT
# -------------------------------


def generate_report():
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    results = {
        "python_version": check_python_version(),
        "dependencies": check_dependencies(),
        "cuda": check_cuda(),
        "audio_devices": check_audio_devices(),
        "ai_modules": check_ai_modules(),
        "directory_structure": check_directory_structure(),
        "mock_audio": test_mock_audio_passthrough(),
        "real_inference": test_real_inference(),
        "feature_extraction": test_feature_extraction(),
        "latency_benchmark": benchmark_latency(),
    }

    passed = sum(bool(v) for v in results.values())
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")

    report_path = ROOT_DIR / "logs" / "diagnostic_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # 🔧 FIXED: JSON-safe serialization
    report_path.write_text(json.dumps(make_json_safe(results), indent=2))

    print(f"Report saved: {report_path}")
    return passed == total


def main():
    return generate_report()


if __name__ == "__main__":
    print("VoiceForge-Nextgen Diagnostic Tool")
    print(f"Root directory: {ROOT_DIR}")
    sys.exit(0 if main() else 1)
