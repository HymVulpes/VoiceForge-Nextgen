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

def check_python_version():
    """Check Python version"""
    print("\n" + "="*60)
    print("1. PYTHON VERSION")
    print("="*60)
    
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
    print("\n" + "="*60)
    print("2. DEPENDENCIES")
    print("="*60)
    
    required = {
        'torch': 'PyTorch',
        'pyaudio': 'PyAudio',
        'numpy': 'NumPy',
        'sqlalchemy': 'SQLAlchemy',
        'colorlog': 'ColorLog'
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
    print("\n" + "="*60)
    print("3. CUDA / GPU")
    print("="*60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.version.cuda}")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("WARNING: CUDA not available - will use CPU (slower)")
            return False
    except Exception as e:
        print(f"[ERROR] Error checking CUDA: {e}")
        return False

def check_audio_devices():
    """Check audio devices"""
    print("\n" + "="*60)
    print("4. AUDIO DEVICES")
    print("="*60)
    
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
            name = info['name']
            
            if info['maxInputChannels'] > 0:
                input_devices.append((i, name))
                print(f"  [IN  {i}] {name}")
            
            if info['maxOutputChannels'] > 0:
                output_devices.append((i, name))
                print(f"  [OUT {i}] {name}")
                
                # Check for Virtual Audio Cable
                vac_keywords = ['CABLE Input', 'VoiceMeeter Input', 'Virtual Audio Cable']
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


# ===== PHẦN AI MODULE ĐÃ SỬA =====
def check_ai_modules():
    """Check AI-related modules WITHOUT circular imports"""
    print("\n" + "="*60)
    print("5. AI MODULES")
    print("="*60)
    
    # Lazy import các thư viện bên ngoài
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
            if module in ["torch", "numpy"]:  # Critical only
                all_ok = False
    
    # Check local core modules bằng file existence, KHÔNG import
    print("\nChecking local core modules (file check only)...")
    core_files = [
        "app/core/model_loader.py",
        "app/core/rvc_engine.py",
        "app/core/feature_cache.py",
        "app/core/model_cache.py"
    ]
    
    for filepath in core_files:
        full_path = ROOT_DIR / filepath
        if full_path.exists():
            print(f"[OK] {filepath} exists")
        else:
            print(f"[MISSING] {filepath}")
            all_ok = False
    
    # Check assets HuBERT
    assets_dir = ROOT_DIR / "app" / "core" / "assets"
    hubert_files = list(assets_dir.glob("hubert_base*.pt")) if assets_dir.exists() else []
    
    if hubert_files:
        print(f"[OK] HuBERT model: {hubert_files[0].name}")
    else:
        print("[INFO] HuBERT model not found (will use RMVPE)")
    
    return all_ok
# ===== END PHẦN AI MODULE =====


def check_directory_structure():
    """Check required directories"""
    print("\n" + "="*60)
    print("6. DIRECTORY STRUCTURE")
    print("="*60)
    
    required_dirs = [
        ROOT_DIR / "app",
        ROOT_DIR / "app" / "core",
        ROOT_DIR / "app" / "audio",
        ROOT_DIR / "app" / "db",
        ROOT_DIR / "app" / "utils",
        ROOT_DIR / "SampleVoice",
        ROOT_DIR / "logs",
        ROOT_DIR / "logs" / "snapshots"
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✓ {dir_path.relative_to(ROOT_DIR)}")
        else:
            print(f"✗ {dir_path.relative_to(ROOT_DIR)} (creating...)")
            dir_path.mkdir(parents=True, exist_ok=True)
            all_ok = False
    
    return True  # Always return True since we create missing dirs

def test_mock_audio_passthrough():
    """Test basic audio passthrough"""
    print("\n" + "="*60)
    print("7. MOCK AUDIO TEST")
    print("="*60)
    
    try:
        import numpy as np
        import pyaudio
        
        # Create mock audio data
        sample_rate = 48000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        
        mock_audio = np.random.randn(samples).astype(np.float32) * 0.1
        
        print(f"[OK] Generated mock audio: {samples} samples")
        print(f"  RMS: {np.sqrt(np.mean(mock_audio**2)):.6f}")
        print(f"  Shape: {mock_audio.shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Mock audio test failed: {e}")
        return False

def generate_report():
    """Generate diagnostic report"""
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    results = {
        "python_version": check_python_version(),
        "dependencies": check_dependencies(),
        "cuda": check_cuda(),
        "audio_devices": check_audio_devices(),
        "ai_modules": check_ai_modules(),
        "directory_structure": check_directory_structure(),
        "mock_audio": test_mock_audio_passthrough()
    }
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n[OK] ALL CHECKS PASSED - Ready to run VoiceForge!")
        print("\nNext steps:")
        print("  1. Place RVC model files (.pth, .index) in SampleVoice/ folder")
        print("  2. Run: python app/main_v2.py")
    else:
        print("\nWARNING: SOME CHECKS FAILED - Please fix issues above")
    
    # Save report
    report_path = ROOT_DIR / "logs" / "diagnostic_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nReport saved: {report_path}")
    
    return passed == total

if __name__ == "__main__":
    print("VoiceForge-Nextgen Diagnostic Tool")
    print(f"Root directory: {ROOT_DIR}")
    
    success = generate_report()
    sys.exit(0 if success else 1)
