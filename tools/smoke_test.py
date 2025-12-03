# -*- coding: utf-8 -*-
"""
Smoke test: Kiểm tra import và xây dựng graph TensorFlow không cần GPU/dữ liệu thật.
Phù hợp cho máy yếu, chỉ test compatibility.
"""
import os
import sys
import tensorflow as tf
import numpy as np

# Disable eager execution for TF1-style graph mode
tf.compat.v1.disable_eager_execution()

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs.config import Config

print("=" * 60)
print("SMOKE TEST: Import và Graph Build")
print("=" * 60)

# Test 1: Import các module chính
print("\n[1/4] Testing imports...")
try:
    import tf_slim as slim
    from libs.networks.network_factory import get_network_byname
    from libs import build_rpn, build_fast_rcnn, build_fpn
    from reference import load_reference_image
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Kiểm tra phiên bản TensorFlow và eager mode
print("\n[2/4] Checking TensorFlow config...")
print(f"  TensorFlow version: {tf.__version__}")
print(f"  Eager execution: {tf.executing_eagerly()}")
if tf.executing_eagerly():
    print("  ✗ WARNING: Eager execution is still enabled!")
else:
    print("  ✓ Graph mode active")

# Test 3: Load reference images (test skimage resize)
print("\n[3/4] Testing reference image loading...")
try:
    ref_images = load_reference_image()
    print(f"  ✓ Reference images loaded: shape={ref_images.shape}, dtype={ref_images.dtype}")
except Exception as e:
    print(f"  ✗ Reference image loading failed: {e}")
    sys.exit(1)

# Test 4: Build small graph (không chạy session, chỉ xây graph)
print("\n[4/4] Testing graph construction (minimal)...")
try:
    net_config = Config()
    batch = 1
    h, w, c = 64, 64, 3  # Kích thước nhỏ để nhẹ
    
    # Tạo placeholder/constant giả
    image_placeholder = tf.compat.v1.placeholder(tf.float32, [batch, h, w, c], name="test_image")
    
    # Build network (chỉ tạo graph, không chạy)
    with tf.compat.v1.variable_scope("smoke_test"):
        _, share_net = get_network_byname(
            inputs=image_placeholder,
            config=net_config,
            is_training=False,
            reuse=False
        )
    
    print(f"  ✓ Graph built successfully. Share net keys: {list(share_net.keys())}")
    print(f"  ✓ Total trainable variables: {len(tf.compat.v1.trainable_variables())}")
    
except Exception as e:
    print(f"  ✗ Graph construction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ SMOKE TEST PASSED: Code tương thích TF2 + compat.v1")
print("=" * 60)
print("\nLưu ý:")
print("  - Đây chỉ là test import/graph, không train thật.")
print("  - Để train, cần chuẩn bị data TFRecord và chạy tools/train.py")
print("  - Colab: dùng notebook COLAB_GUIDE.ipynb trong repo")
