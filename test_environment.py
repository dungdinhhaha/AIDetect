#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script for ComparisonDetector
Tests if all imports and basic setup work
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("ComparisonDetector - Environment Test")
print("="*60)

# Test 1: Import TensorFlow
print("\n[1/7] Testing TensorFlow import...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} imported successfully")
except Exception as e:
    print(f"✗ TensorFlow import failed: {e}")
    sys.exit(1)

# Test 2: Import NumPy
print("\n[2/7] Testing NumPy import...")
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} imported successfully")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

# Test 3: Import OpenCV
print("\n[3/7] Testing OpenCV import...")
try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__} imported successfully")
except Exception as e:
    print(f"✗ OpenCV import failed: {e}")
    sys.exit(1)

# Test 4: Import Config
print("\n[4/7] Testing Config import...")
try:
    from configs.config import Config
    config = Config()
    print(f"✓ Config imported successfully")
    print(f"  - NUM_CLASS: {config.NUM_CLASS}")
    print(f"  - NET_NAME: {config.NET_NAME}")
    print(f"  - BATCH_SIZE: {config.BATCH_SIZE}")
except Exception as e:
    print(f"✗ Config import failed: {e}")
    sys.exit(1)

# Test 5: Import project modules
print("\n[5/7] Testing project modules...")
try:
    from libs import build_fpn, build_rpn, build_fast_rcnn
    from libs.networks.network_factory import get_network_byname
    print("✓ All project modules imported successfully")
except Exception as e:
    print(f"✗ Project module import failed: {e}")
    sys.exit(1)

# Test 6: Check reference images
print("\n[6/7] Testing reference image loader...")
try:
    from reference import load_reference_image
    print("✓ Reference image loader imported successfully")
except Exception as e:
    print(f"✗ Reference import failed: {e}")
    sys.exit(1)

# Test 7: Summary
print("\n[7/7] Environment summary...")
print(f"  - Python: {sys.version.split()[0]}")
print(f"  - TensorFlow: {tf.__version__}")
print(f"  - NumPy: {np.__version__}")
print(f"  - OpenCV: {cv2.__version__}")
print(f"  - Working directory: {os.getcwd()}")

print("\n" + "="*60)
print("✓ All tests passed! Environment is ready.")
print("="*60)

print("\nNext steps:")
print("1. Prepare your dataset in TFRecord format")
print("2. Update paths in configs/config.py:")
print("   - CHECKPOINT_DIR (ResNet pretrained weights)")
print("   - DATA_DIR (your TFRecord data)")
print("3. Run training: python tools/train.py")
print("\nNote: Training requires:")
print("  - TFRecord dataset in ./tfdata/")
print("  - ResNet checkpoint weights")
print("  - GPU recommended (CUDA 9.1 + cuDNN 7.0)")
