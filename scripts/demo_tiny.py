#!/usr/bin/env python3
"""
Numerai TINY Demo - Minimal Memory Usage
=========================================

Uses only 10 eras for demonstration purposes.
Perfect for testing the pipeline with limited RAM.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

print("="*80)
print("NUMERAI TINY DEMO - 10 ERAS ONLY")
print("="*80)

# Load only what we need
print("\nLoading data (last 10 eras only)...")
train_df = pd.read_parquet("data/train.parquet")

# Get last 10 eras
eras = sorted(train_df['era'].unique())[-10:]
train_df = train_df[train_df['era'].isin(eras)].copy()

print(f"✓ Loaded {len(train_df):,} samples from eras {eras[0]} to {eras[-1]}")

# Get features
feature_cols = [c for c in train_df.columns if c.startswith('feature')]
print(f"✓ Using {len(feature_cols)} features")

# Clean
train_df = train_df.dropna(subset=['target'])

# Split (last 2 eras as test)
train_eras = eras[:8]
test_eras = eras[8:]

train_mask = train_df['era'].isin(train_eras)
test_mask = train_df['era'].isin(test_eras)

X_train = train_df.loc[train_mask, feature_cols].values
y_train = train_df.loc[train_mask, 'target'].values
X_test = train_df.loc[test_mask, feature_cols].values
y_test = train_df.loc[test_mask, 'target'].values
eras_test = train_df.loc[test_mask, 'era'].values

print(f"✓ Train: {len(X_train):,} samples")
print(f"✓ Test: {len(X_test):,} samples")

# Scale
print("\nScaling...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train LightGBM
print("\nTraining LightGBM...")
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=6,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
spearman = spearmanr(y_test, y_pred)[0]

print(f"\n{'='*80}")
print(f"RESULTS")
print(f"{'='*80}")
print(f"Spearman Correlation: {spearman:.6f}")
print(f"{'='*80}")

print(f"\n✓ Demo complete!")
print(f"\nNote: This uses only 10 eras for testing.")
print(f"Run quick_start_lite.py for a more complete training.")
