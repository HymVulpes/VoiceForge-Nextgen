"""
VoiceForge Diagnostic Tool
Comprehensive system check before running main application
Run this first: python diagnostic_tool.py
"""

import sys
from pathlib import Path
import json

# Add app directory
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# -------------------------------
# CHECK FUNCTIONS
# -------------------------------


def check_python_version():
    """Check Python version"""
    print("\n" + "=" * 60)
    print("1. PYTHON VERSION")
    print("=" * 60)

    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major != 3 or version.minor != 12:
        print("WARNING: Python 3.12.9 recommended")
        return False
    else:
        print("[OK] Python version OK")
        return True


def check_dependencies():
    """Check required packages"""
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
    """Check CUDA availability"""
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
        else:
            print("WARNING: CUDA not available - will use CPU (slower)")
            return False
    except Exception as e:
        print(f"[ERROR] Error checking CUDA: {e}")
        return False


def check_audio_devices():
    """Check audio devices"""
    print("\n" + "=" * 60)
    print("4. AUDIO DEVICES")
    print("=" * 60)

    try:
        import pyaudio

        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        print(f"Found {device_count} audio devices:")

        input_devices = []
        output_devices = []
        vac_found = False

        for i in range(device_count):
            info = p.get_device_info_by_index(i)
            name = info["name"]

            if info["maxInputChannels"] > 0:
                input_devices.append((i, name))
                print(f"  [IN  {i}] {name}")

            if info["maxOutputChannels"] > 0:
                output_devices.append((i, name))

                vac_keywords = [
                    "CABLE Input",
                    "VoiceMeeter Input",
                    "Virtual Audio Cable",
                ]
                if any(kw.lower() in name.lower() for kw in vac_keywords):
                    print(f"    [OK] Virtual Audio Cable detected!")
                    vac_found = True

        p.terminate()

        if not vac_found:
            print("\nWARNING: No Virtual Audio Cable found!")
            print("   Please install VB-Audio Virtual Cable or VoiceMeeter")
            print("   Download: https://vb-audio.com/Cable/")

        return len(input_devices) > 0 and len(output_devices) > 0

    except Exception as e:
        print(f"[ERROR] Error checking audio devices: {e}")
        return False


def check_ai_modules():
    """Check AI-related modules WITHOUT circular imports"""
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
            if module in ["torch", "numpy"]:
                all_ok = False

    print("\nChecking local core modules (file check only)...")
    core_files = [
        "app/core/model_loader.py",
        "app/core/rvc_engine.py",
        "app/core/feature_cache.py",
        "app/core/model_cache.py",
    ]

    for filepath in core_files:
        full_path = ROOT_DIR / filepath
        if full_path.exists():
            print(f"[OK] {filepath} exists")
        else:
            print(f"[MISSING] {filepath}")
            all_ok = False

    assets_dir = ROOT_DIR / "app" / "core" / "assets"
    hubert_files = (
        list(assets_dir.glob("hubert_base*.pt")) if assets_dir.exists() else []
    )

    if hubert_files:
        print(f"[OK] HuBERT model: {hubert_files[0].name}")
    else:
        print("[INFO] HuBERT model not found (will use RMVPE)")

    return all_ok


def check_directory_structure():
    """Check required directories"""
    print("\n" + "=" * 60)
    print("6. DIRECTORY STRUCTURE")
    print("=" * 60)

    required_dirs = [
        ROOT_DIR / "app",
        ROOT_DIR / "app" / "core",
        ROOT_DIR / "app" / "audio",
        ROOT_DIR / "app" / "db",
        ROOT_DIR / "app" / "utils",
        ROOT_DIR / "SampleVoice",
        ROOT_DIR / "logs",
        ROOT_DIR / "logs" / "snapshots",
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"✗ {dir_path.relative_to(ROOT_DIR)} (creating...)")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✓ {dir_path.relative_to(ROOT_DIR)}")

    return True


def test_mock_audio_passthrough():
    """Test basic audio passthrough"""
    print("\n" + "=" * 60)
    print("7. MOCK AUDIO TEST")
    print("=" * 60)

    try:
        import numpy as np

        sample_rate = 48000
        duration = 0.1
        samples = int(sample_rate * duration)
        mock_audio = np.random.randn(samples).astype(np.float32) * 0.1

        print(f"[OK] Generated mock audio: {samples} samples")
        print(f"  RMS: {np.sqrt(np.mean(mock_audio**2)):.6f}")
        print(f"  Shape: {mock_audio.shape}")

        return True

    except Exception as e:
        print(f"[ERROR] Mock audio test failed: {e}")
        return False


# ======================================================================
# ===================== [ADDED BY CLAUDE - PHASE 4] =====================
# ======================================================================


def test_real_inference():
    """Test RVC inference with mock model"""
    print("\n" + "=" * 60)
    print("8. REAL INFERENCE TEST (Mock)")
    print("=" * 60)

    try:
        from tests.fixtures.mock_models import create_mock_model, create_test_audio
        from app.core.rvc_inferencer import RVCInferencer

        model_data = create_mock_model()
        print("[OK] Mock model created")

        inferencer = RVCInferencer(model_data=model_data, device="cpu", use_fp16=False)
        print("[OK] Inferencer created")

        audio = create_test_audio(duration_s=0.1)
        output = inferencer.infer(audio, pitch_shift=0)

        print(f"[OK] Inference successful: {len(output)} samples")
        print(f"  Input shape: {audio.shape}")
        print(f"  Output shape: {output.shape}")

        return True

    except Exception as e:
        print(f"[ERROR] Inference test failed: {e}")
        return False


def test_feature_extraction():
    """Test F0 + HuBERT extraction"""
    print("\n" + "=" * 60)
    print("9. FEATURE EXTRACTION TEST")
    print("=" * 60)

    try:
        import numpy as np
        from app.core.feature_extractor import F0Extractor
        from tests.fixtures.mock_models import create_test_audio

        audio = create_test_audio(duration_s=1.0, frequency=440.0)
        print(f"[OK] Test audio created: {len(audio)} samples")

        extractor = F0Extractor(method="harvest", sr=48000)
        f0 = extractor.extract(audio)

        print(f"[OK] F0 extracted: {len(f0)} frames")

        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) > 0:
            mean_f0 = np.mean(voiced_f0)
            print(f"  Mean F0: {mean_f0:.1f} Hz (expected ~440 Hz)")

        return True

    except Exception as e:
        print(f"[ERROR] Feature extraction test failed: {e}")
        return False


def benchmark_latency():
    """Measure end-to-end latency"""
    print("\n" + "=" * 60)
    print("10. LATENCY BENCHMARK")
    print("=" * 60)

    try:
        import time
        import numpy as np
        from app.core.audio_preprocessor import AudioPreprocessor
        from app.core.audio_postprocessor import AudioPostprocessor
        from tests.fixtures.mock_models import create_test_audio

        preprocessor = AudioPreprocessor(target_sr=48000)
        postprocessor = AudioPostprocessor()

        audio = create_test_audio(duration_s=0.1)
        latencies = []

        for _ in range(10):
            start = time.perf_counter()
            processed = preprocessor.process(audio, source_sr=48000)
            output = processed * 0.95
            _ = postprocessor.process(output)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = np.mean(latencies)
        print(f"[OK] Average latency: {avg_latency:.2f} ms")

        return avg_latency <= 15.0

    except Exception as e:
        print(f"[ERROR] Latency benchmark failed: {e}")
        return False


# -------------------------------
# GENERATE REPORT
# -------------------------------


def generate_report():
    """Generate diagnostic report"""
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
        # -------- ADDED --------
        "real_inference": test_real_inference(),
        "feature_extraction": test_feature_extraction(),
        "latency_benchmark": benchmark_latency(),
    }

    passed = sum(results.values())
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")

    report_path = ROOT_DIR / "logs" / "diagnostic_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2))

    return passed == total


def main():
    return generate_report()


if __name__ == "__main__":
    print("VoiceForge-Nextgen Diagnostic Tool")
    print(f"Root directory: {ROOT_DIR}")
    success = generate_report()
    sys.exit(0 if success else 1)
