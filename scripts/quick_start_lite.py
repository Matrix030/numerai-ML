#!/usr/bin/env python3
"""
Numerai Quick Start Script - LITE VERSION
==========================================

Memory-optimized version for large datasets.
Uses only recent eras to reduce memory usage.

Usage:
    python scripts/quick_start_lite.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import time
import gc

from numerai_utils import (
    NumeraiMetrics,
    NumeraiDataLoader,
    SubmissionFormatter,
    print_model_comparison
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

from numerapi import NumerAPI

# Check for GPU availability
GPU_AVAILABLE = False
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"✓ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except ImportError:
    print("⚠ PyTorch not available - GPU detection skipped")


def download_data():
    """Download Numerai data if not already present."""
    print("=" * 80)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 80)

    napi = NumerAPI()
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    files = [
        ("v5.0/train.parquet", "train.parquet"),
        ("v5.0/validation.parquet", "validation.parquet"),
        ("v5.0/live.parquet", "live.parquet")
    ]

    for remote_path, local_name in files:
        local_path = data_dir / local_name
        if not local_path.exists():
            print(f"Downloading {local_name}...")
            napi.download_dataset(remote_path, str(local_path))
            print(f"✓ {local_name} downloaded")
        else:
            print(f"✓ {local_name} already exists")

    print("\n✓ All data ready!\n")


def load_and_prepare_data():
    """Load and prepare data - MEMORY OPTIMIZED VERSION."""
    print("=" * 80)
    print("STEP 2: LOADING DATA (MEMORY OPTIMIZED)")
    print("=" * 80)

    loader = NumeraiDataLoader("data")

    print("Loading training data with memory optimization...")

    # Load in chunks and use only recent eras to save memory
    train_df = pd.read_parquet("data/train.parquet")

    print(f"✓ Loaded {len(train_df):,} total samples")

    # Use only recent eras to reduce memory (optimized for 30GB RAM systems)
    unique_eras = sorted(train_df['era'].unique())
    recent_eras = unique_eras[-100:]  # Last 100 eras (~8-10GB memory usage)

    print(f"✓ Using recent {len(recent_eras)} eras (from {recent_eras[0]} to {recent_eras[-1]})")

    # Filter to recent eras
    train_df = train_df[train_df['era'].isin(recent_eras)].copy()

    print(f"✓ Reduced to {len(train_df):,} samples")

    # Get feature columns
    feature_cols = loader.get_feature_columns(train_df)
    print(f"✓ Found {len(feature_cols)} features")

    # Remove missing targets
    train_clean = train_df.dropna(subset=['target']).copy()
    print(f"✓ Clean samples: {len(train_clean):,}")

    # Free memory
    del train_df
    gc.collect()

    # Extract features, target, and eras
    print("\nExtracting features...")
    X = train_clean[feature_cols].values.astype(np.float32)  # Use float32 to save memory
    y = train_clean['target'].values.astype(np.float32)
    eras = train_clean['era'].values

    # Free more memory
    del train_clean
    gc.collect()

    # Era-based split (80/20)
    unique_eras = sorted(np.unique(eras))
    split_idx = int(len(unique_eras) * 0.8)

    train_eras = unique_eras[:split_idx]
    test_eras = unique_eras[split_idx:]

    train_mask = np.isin(eras, train_eras)
    test_mask = np.isin(eras, test_eras)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    eras_train, eras_test = eras[train_mask], eras[test_mask]

    print(f"\n✓ Train samples: {len(X_train):,}")
    print(f"✓ Test samples: {len(X_test):,}")
    print(f"✓ Train eras: {len(train_eras)} ({train_eras[0]} to {train_eras[-1]})")
    print(f"✓ Test eras: {len(test_eras)} ({test_eras[0]} to {test_eras[-1]})\n")

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features standardized\n")

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'eras_train': eras_train,
        'eras_test': eras_test,
        'scaler': scaler,
        'feature_cols': feature_cols
    }


def train_models(data):
    """Train models - reduced complexity for memory."""
    print("=" * 80)
    print("STEP 3: TRAINING MODELS (LITE)")
    print("=" * 80)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    eras_test = data['eras_test']

    models = {}
    results = {}

    # 1. Linear Regression
    print("\n1. Training Linear Regression...")
    start = time.time()
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred_lr = lr_model.predict(X_test)
    results['Linear Regression'] = NumeraiMetrics.evaluate_model(
        y_test, y_pred_lr, eras_test, "Linear Regression"
    )
    models['Linear Regression'] = lr_model
    print(f"✓ Trained in {train_time:.2f}s - Spearman: {results['Linear Regression']['spearman']:.6f}")

    # Free memory
    del lr_model
    gc.collect()

    # 2. Random Forest (smaller version for memory)
    print("\n2. Training Random Forest...")
    start = time.time()

    rf_model = RandomForestRegressor(
        n_estimators=50,  # Reduced from 100
        max_depth=8,
        min_samples_split=100,
        min_samples_leaf=50,
        max_features='sqrt',
        random_state=42,
        n_jobs=4,
        verbose=0
    )
    rf_model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred_rf = rf_model.predict(X_test)
    results['Random Forest'] = NumeraiMetrics.evaluate_model(
        y_test, y_pred_rf, eras_test, "Random Forest"
    )
    models['Random Forest'] = rf_model
    print(f"✓ Trained in {train_time:.2f}s - Spearman: {results['Random Forest']['spearman']:.6f}")

    # Free memory
    del rf_model
    gc.collect()

    # 3. LightGBM (GPU-Accelerated)
    if GPU_AVAILABLE:
        print("\n3. Training LightGBM with GPU acceleration...")
    else:
        print("\n3. Training LightGBM on CPU...")
    start = time.time()

    lgb_params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 255 if GPU_AVAILABLE else 31,  # Larger for GPU
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 6,
        'min_data_in_leaf': 100,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'random_state': 42
    }

    # Add GPU-specific parameters if available
    if GPU_AVAILABLE:
        lgb_params['device'] = 'gpu'
        lgb_params['gpu_platform_id'] = 0
        lgb_params['gpu_device_id'] = 0
    else:
        lgb_params['n_jobs'] = 4

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=500,  # Reduced from 1000
        valid_sets=[lgb_eval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )
    train_time = time.time() - start

    y_pred_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    results['LightGBM'] = NumeraiMetrics.evaluate_model(
        y_test, y_pred_lgb, eras_test, "LightGBM"
    )
    models['LightGBM'] = lgb_model
    print(f"✓ Trained in {train_time:.2f}s - Spearman: {results['LightGBM']['spearman']:.6f}")

    print("\n")
    return models, results


def generate_predictions(models, data):
    """Generate predictions on live data."""
    print("=" * 80)
    print("STEP 4: GENERATING LIVE PREDICTIONS")
    print("=" * 80)

    print("\n✓ Loading live data...")
    live_df = pd.read_parquet("data/live.parquet")
    print(f"✓ Loaded {len(live_df):,} live samples")

    # Prepare features
    feature_cols = data['feature_cols']
    X_live = live_df[feature_cols].values.astype(np.float32)
    X_live_scaled = data['scaler'].transform(X_live)

    # Use best model (LightGBM)
    lgb_model = models['LightGBM']
    predictions = lgb_model.predict(X_live_scaled, num_iteration=lgb_model.best_iteration)

    print(f"✓ Generated {len(predictions):,} predictions")
    print(f"\nPrediction statistics:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std:  {predictions.std():.6f}")
    print(f"  Min:  {predictions.min():.6f}")
    print(f"  Max:  {predictions.max():.6f}")

    # Create submission (id is the index)
    submission = SubmissionFormatter.create_submission(
        live_df.index,
        predictions,
        "data/submission.csv"
    )

    print("\n")
    return submission


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("NUMERAI QUICK START - LITE VERSION")
    print("(Memory-optimized for large datasets)")
    print("=" * 80)
    print()

    try:
        # Step 1: Download data
        download_data()

        # Step 2: Load and prepare data
        data = load_and_prepare_data()

        # Step 3: Train models
        models, results = train_models(data)

        # Print comparison
        print_model_comparison(results)

        # Step 4: Generate predictions
        submission = generate_predictions(models, data)

        print("=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print("\nYour predictions are ready in: data/submission.csv")
        print("\nNote: This lite version uses only recent 100 eras to save memory.")
        print("For full dataset training (all eras), 64GB+ RAM recommended.")
        print("\nGood luck! 🚀\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nIf you're getting memory errors:")
        print("  - This dataset is LARGE (2.7M samples)")
        print("  - Lite version uses only recent 100 eras")
        print("  - Reduce to 50 eras on line 79 if needed")
        print("  - Or use Google Colab with high RAM runtime")
        raise


if __name__ == "__main__":
    main()
