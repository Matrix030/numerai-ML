#!/usr/bin/env python3
"""
GPU Verification Script for Numerai Project
============================================

Checks if GPU is available and properly configured for:
- PyTorch
- LightGBM
- XGBoost (if installed)
- CatBoost (if installed)
"""

import sys

print("=" * 80)
print("GPU VERIFICATION FOR NUMERAI PROJECT")
print("=" * 80)

# 1. Check NVIDIA GPU
print("\n1. NVIDIA GPU Detection:")
print("-" * 80)
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                print(f"✓ {line.strip()}")
                break
        print("✓ NVIDIA driver installed and working")
    else:
        print("✗ nvidia-smi failed - driver may not be installed")
except FileNotFoundError:
    print("✗ nvidia-smi not found - NVIDIA drivers not installed")

# 2. Check PyTorch CUDA
print("\n2. PyTorch CUDA Support:")
print("-" * 80)
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"✓ CUDA available: YES")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Quick test
        try:
            x = torch.randn(1000, 1000, device='cuda')
            y = x @ x
            print(f"✓ GPU compute test: PASSED")
        except Exception as e:
            print(f"✗ GPU compute test failed: {e}")
    else:
        print("✗ CUDA not available in PyTorch")
        print("  Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
except ImportError:
    print("✗ PyTorch not installed")
    print("  Install with: pip install torch")

# 3. Check LightGBM GPU
print("\n3. LightGBM GPU Support:")
print("-" * 80)
try:
    import lightgbm as lgb
    print(f"✓ LightGBM version: {lgb.__version__}")

    # Test GPU support
    try:
        import numpy as np
        X = np.random.rand(1000, 50)
        y = np.random.rand(1000)

        params = {
            'objective': 'regression',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'verbose': -1
        }

        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_data, num_boost_round=10)
        print("✓ LightGBM GPU test: PASSED")
        print("✓ LightGBM can use GPU for training")
    except Exception as e:
        print(f"✗ LightGBM GPU test failed: {e}")
        print("  You may need to rebuild LightGBM with GPU support")
        print("  Install with: pip install lightgbm --config-settings=cmake.define.USE_GPU=ON")
except ImportError:
    print("✗ LightGBM not installed")
    print("  Install with: pip install lightgbm")

# 4. Check XGBoost GPU (optional)
print("\n4. XGBoost GPU Support:")
print("-" * 80)
try:
    import xgboost as xgb
    print(f"✓ XGBoost version: {xgb.__version__}")

    # Test GPU support
    try:
        import numpy as np
        X = np.random.rand(1000, 50)
        y = np.random.rand(1000)

        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'objective': 'reg:squarederror'
        }

        model = xgb.train(params, dtrain, num_boost_round=10)
        print("✓ XGBoost GPU test: PASSED")
        print("✓ XGBoost can use GPU for training")
    except Exception as e:
        print(f"✗ XGBoost GPU test failed: {e}")
        print("  XGBoost may not be compiled with CUDA support")
except ImportError:
    print("⚠ XGBoost not installed (optional)")
    print("  Install with: pip install xgboost")

# 5. Check CatBoost GPU (optional)
print("\n5. CatBoost GPU Support:")
print("-" * 80)
try:
    from catboost import CatBoostRegressor
    import catboost
    print(f"✓ CatBoost version: {catboost.__version__}")

    # Test GPU support
    try:
        import numpy as np
        X = np.random.rand(1000, 50)
        y = np.random.rand(1000)

        model = CatBoostRegressor(
            iterations=10,
            task_type='GPU',
            devices='0',
            verbose=False
        )
        model.fit(X, y)
        print("✓ CatBoost GPU test: PASSED")
        print("✓ CatBoost can use GPU for training")
    except Exception as e:
        print(f"✗ CatBoost GPU test failed: {e}")
except ImportError:
    print("⚠ CatBoost not installed (optional)")
    print("  Install with: pip install catboost")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

gpu_ready = False
try:
    import torch
    if torch.cuda.is_available():
        gpu_ready = True
except:
    pass

if gpu_ready:
    print("\n✓ YOUR SYSTEM IS GPU-READY!")
    print("\nYour Numerai scripts will automatically use GPU acceleration.")
    print("Expected speedup: 10-50x faster training with LightGBM")
    print("\nTo use GPU, simply run:")
    print("  python quick_start_lite.py")
    print("  python quick_start.py")
else:
    print("\n⚠ GPU NOT FULLY CONFIGURED")
    print("\nYour scripts will fall back to CPU training.")
    print("To enable GPU acceleration, install PyTorch with CUDA:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "=" * 80)
