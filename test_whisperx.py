#!/usr/bin/env python3
"""Quick test script to verify WhisperX installation"""

import sys

print("Testing WhisperX installation...")
print("=" * 50)

# Test 1: Import
try:
    import whisperx
    print("✓ WhisperX imported successfully")
except ImportError as e:
    print(f"✗ Failed to import WhisperX: {e}")
    sys.exit(1)

# Test 2: Check version
try:
    version = getattr(whisperx, '__version__', 'unknown')
    print(f"✓ WhisperX version: {version}")
except:
    print("✓ WhisperX installed (version unknown)")

# Test 3: Check key functions
required_functions = ['load_model', 'load_align_model', 'align', 'load_audio']
missing = []
for func in required_functions:
    if hasattr(whisperx, func):
        print(f"✓ Function '{func}' available")
    else:
        print(f"✗ Function '{func}' missing")
        missing.append(func)

if missing:
    print(f"\n✗ Missing functions: {missing}")
    sys.exit(1)

# Test 4: Check dependencies
print("\nChecking dependencies...")
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("✗ PyTorch not found")

try:
    import librosa
    print(f"✓ librosa available")
except ImportError:
    print("✗ librosa not found")

try:
    import numpy as np
    print(f"✓ numpy available")
except ImportError:
    print("✗ numpy not found")

# Test 5: Try to load a tiny model (quick test)
print("\nTesting model loading (this may take a moment)...")
try:
    print("  Loading tiny model...")
    model = whisperx.load_model("tiny", device="cpu", language="vi")
    print("✓ Model loaded successfully!")
    print("  Model type:", type(model))
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    print("  This might be normal if models need to be downloaded first")
    print("  Error type:", type(e).__name__)

print("\n" + "=" * 50)
print("Installation check complete!")
print("\nIf all checks passed, WhisperX should work correctly.")
print("Note: First run will download models, which may take time.")


