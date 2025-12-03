#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto setup and run ComparisonDetector
This script will install all dependencies and run training
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """Run a shell command and print output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n[WARNING] Command failed with exit code {result.returncode}")
        print(f"Continuing anyway...\n")
    return result.returncode

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     ComparisonDetector - Auto Setup & Run                 ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Change to project directory
    os.chdir(r'D:\ComparisonDetector')
    
    # Step 1: Install opencv-python-headless
    run_command(
        r'D:\anacoda\envs\tf1\python.exe -m pip install opencv-python-headless==4.2.0.34',
        "Installing opencv-python-headless"
    )
    
    # Step 2: Install TensorFlow 1.15.0
    run_command(
        r'D:\anacoda\envs\tf1\python.exe -m pip install tensorflow==1.15.0',
        "Installing TensorFlow 1.15.0"
    )
    
    # Step 3: Install Pillow and other dependencies
    run_command(
        r'D:\anacoda\envs\tf1\python.exe -m pip install pillow scikit-image scipy',
        "Installing additional dependencies"
    )
    
    # Step 4: Test imports
    print(f"\n{'='*60}")
    print("Testing imports...")
    print(f"{'='*60}\n")
    
    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
    except:
        print("✗ numpy import failed")
    
    try:
        import cv2
        print(f"✓ opencv-python {cv2.__version__}")
    except:
        print("✗ opencv-python import failed")
    
    try:
        import tensorflow as tf
        print(f"✓ tensorflow {tf.__version__}")
    except:
        print("✗ tensorflow import failed")
    
    # Step 5: Run training
    print(f"\n{'='*60}")
    print("Ready to run training!")
    print(f"{'='*60}\n")
    
    input("Press Enter to start training (or Ctrl+C to cancel)...")
    
    print("\nStarting training...\n")
    run_command(
        r'D:\anacoda\envs\tf1\python.exe tools\train.py',
        "Running train.py"
    )
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     Setup and training complete!                          ║
    ╚════════════════════════════════════════════════════════════╝
    """)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)
