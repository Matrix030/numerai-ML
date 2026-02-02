# Numerai Stock Market Prediction Project

A complete, production-ready machine learning project for predicting stock market returns using the Numerai Tournament Dataset (v5.0). This project is suitable for a university-level Machine Learning final project.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Components](#project-components)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Resources](#resources)

## Overview

### What is Numerai?

Numerai is a hedge fund that crowdsources machine learning models to predict stock market returns. The platform provides obfuscated and regularized data to protect proprietary information while enabling data scientists worldwide to build predictive models.

### Project Goals

- Build end-to-end ML pipeline for financial prediction
- Compare multiple regression models (Linear Regression, Random Forest, LightGBM)
- Implement proper time-series validation using era-based splits
- Generate production-ready predictions for live data
- Demonstrate best practices in ML engineering and data science

### Key Features

- **Comprehensive EDA**: Feature distributions, correlations, and temporal analysis
- **Feature Engineering**: Standardization and PCA dimensionality reduction
- **Multiple Models**: Linear, Random Forest, and LightGBM implementations
- **Proper Evaluation**: Spearman correlation, Sharpe ratio, per-era analysis
- **Ensembling**: Simple and weighted ensemble methods
- **Live Predictions**: Generate submission-ready predictions

## Project Structure

```
numerai/
├── notebooks/                  # Jupyter notebooks
│   ├── numerai_project.ipynb       # Main notebook with full pipeline
│   ├── numerai_project_lite.ipynb  # Lite version (recent eras only)
│   └── numerai_gpu_advanced.ipynb  # GPU-accelerated training
├── scripts/                    # Python scripts
│   ├── quick_start.py              # Automated pipeline (all eras)
│   ├── quick_start_lite.py         # Automated pipeline (recent eras)
│   ├── verify_setup.py             # Verify installation
│   ├── check_gpu.py                # Check GPU availability
│   └── demo_tiny.py                # Minimal demo
├── data/                       # Data directory (created on first run)
│   ├── train.parquet               # Training data (downloaded)
│   ├── validation.parquet          # Validation data (downloaded)
│   ├── live.parquet                # Live tournament data (downloaded)
│   └── submission.csv              # Generated predictions
├── numerai_utils.py            # Reusable utility module
├── requirements.txt            # Python dependencies
├── requirements-gpu.txt        # GPU dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```


## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended (dataset is large)
- Stable internet connection (for data download)

### Setup

1. Navigate to the project:
```bash
cd /home/rgmatr1x/dev/numerai
```

2. Install dependencies:

**Option A: Using UV (Recommended - Faster)**
```bash
uv venv
uv sync
source .venv/bin/activate  # On Linux/Mac

# Verify installation
python scripts/verify_setup.py
```

**Option B: Using pip (Traditional)**
```bash
pip install -r requirements.txt

# Or install from pyproject.toml
pip install -e .
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook notebooks/numerai_project.ipynb

# Or with UV (no activation needed)
uv run jupyter notebook notebooks/numerai_project.ipynb
```


## Quick Start

### Option 1: Run the Complete Notebook

Open `notebooks/numerai_project.ipynb` and run all cells sequentially. The notebook will:

1. Download Numerai dataset automatically (~500MB)
2. Perform exploratory data analysis
3. Train and compare multiple models
4. Generate predictions on live data
5. Create submission file

**Estimated runtime**: 15-30 minutes (depending on hardware)

### Option 2: Use the Python Script

For quick experimentation, use the utility module:

```python
from numerai_utils import *
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data
loader = NumeraiDataLoader("data")
train_df = loader.load_train_data()

# Get features and target
feature_cols = loader.get_feature_columns(train_df)
X = train_df[feature_cols].values
y = train_df['target'].values
eras = train_df['era'].values

# Split data
train_df, test_df = loader.era_split(train_df, split_ratio=0.8)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
results = quick_evaluate(model, X_test, y_test, eras_test, "Random Forest")
```

## Project Components

### 1. Introduction & Context
- Explanation of Numerai and project goals
- Understanding obfuscated financial data
- Importance of time-series validation

### 2. Data Loading
- Automatic download via NumerAPI
- Dataset statistics and overview
- Train/validation/live data structure

### 3. Exploratory Data Analysis (EDA)
- Era analysis (temporal structure)
- Target distribution visualization
- Feature correlation heatmaps
- Feature distribution analysis

### 4. Feature Engineering
- Feature selection
- Standardization with StandardScaler
- PCA dimensionality reduction
- Explained variance analysis

### 5. Model Training
Three models are trained and compared:

**Linear Regression (Baseline)**
- Simple, interpretable baseline
- Fast training
- Good for understanding linear relationships

**Random Forest Regressor**
- Ensemble tree-based method
- Captures non-linear patterns
- Provides feature importance

**LightGBM (Gradient Boosting)**
- State-of-the-art gradient boosting
- Typically best performer
- Optimized for large datasets

### 6. Evaluation Metrics

**Spearman Correlation** (Primary Metric)
- Measures rank-based prediction quality
- Range: [-1, 1], higher is better
- Primary scoring metric for Numerai

**Pearson Correlation**
- Measures linear correlation
- Complementary to Spearman

**Mean Squared Error (MSE)**
- Measures absolute prediction error
- Lower is better

**Sharpe Ratio**
- Measures consistency across eras
- Higher indicates robust predictions
- Critical for real-world deployment

### 7. Feature Importance Analysis
- Identify most predictive features
- Visualize feature contributions
- Opportunities for feature selection

### 8. Model Ensembling
- Simple average ensemble
- Weighted ensemble (performance-based)
- Improved consistency and robustness

### 9. Live Predictions
- Load live tournament data
- Generate predictions with best model
- Create submission file (CSV format)
- Instructions for actual submission

### 10. Conclusion
- Summary of results
- Lessons learned
- Future improvement suggestions

## Results

### Expected Performance

With default parameters, you should achieve:

| Model | Spearman | Sharpe | Training Time |
|-------|----------|--------|---------------|
| Linear Regression | ~0.003-0.005 | ~0.1-0.3 | <1 min |
| Random Forest | ~0.004-0.007 | ~0.2-0.4 | 5-10 min |
| LightGBM | ~0.005-0.010 | ~0.3-0.6 | 2-5 min |
| Ensemble | ~0.006-0.011 | ~0.3-0.7 | N/A |

**Note**: Performance varies based on specific data version and random seed.

### Interpretation

- **Spearman > 0.005**: Competitive baseline
- **Sharpe > 0.5**: Consistent across different market conditions
- **Positive correlation in >60% of eras**: Good generalization

## Future Improvements

### 1. Hyperparameter Tuning
```python
# Use Optuna or GridSearchCV
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 10)
    }
    # Train and evaluate...
    return spearman_score
```

### 2. Neural Networks
- TabNet for tabular data
- FT-Transformer architecture
- Multi-layer perceptrons (MLPs)

### 3. Advanced Feature Engineering
- Interaction features
- Rolling statistics across eras
- Target encoding

### 4. Cross-Validation
- GroupKFold with era-based groups
- Time-series cross-validation
- More robust performance estimates

### 5. Multiple Target Prediction
- Numerai provides multiple targets
- Multi-task learning
- Improved generalization

### 6. Model Stacking
- Train meta-model on base predictions
- More sophisticated ensembling
- Better than simple averaging

## Resources

### Official Numerai Resources
- **Website**: https://numer.ai
- **Forum**: https://forum.numer.ai
- **Documentation**: https://docs.numer.ai
- **Example Scripts**: https://github.com/numerai/example-scripts

### Learning Resources
- **Numerai Tournament Guide**: https://docs.numer.ai/tournament/learn
- **Time-Series ML**: Study proper temporal validation
- **Rank Correlation**: Understand Spearman vs Pearson
- **Ensemble Methods**: Explore stacking and blending

### Community
- Active forum with thousands of data scientists
- Weekly tournaments with real monetary stakes
- Regular webinars and discussions

## Submitting to Numerai

To actually submit predictions to the tournament:

1. Create account at https://numer.ai
2. Generate API keys in account settings
3. Create a model and get your model_id
4. Use the following code:

```python
from numerapi import NumerAPI

# Initialize with your credentials
napi = NumerAPI(
    public_id="YOUR_PUBLIC_ID",
    secret_key="YOUR_SECRET_KEY"
)

# Upload predictions
model_id = "YOUR_MODEL_ID"
napi.upload_predictions("data/submission.csv", model_id=model_id)
```

## Technical Notes

### Data Size
- Training data: ~500MB compressed
- Full dataset in memory: ~2-3GB
- Recommended RAM: 8GB+

### Computation Time
- Data download: 5-10 minutes
- Full notebook execution: 15-30 minutes
- Model training: Varies by model and parameters

### Era-Based Validation
**Critical Concept**: Always use era-based or time-based splits for financial data!

- Eras represent time periods (typically weekly)
- Random splits leak future information
- Use last N eras as test set
- Or use GroupKFold with eras as groups

### Why Spearman Correlation?
- Numerai cares about **ranking** stocks, not absolute returns
- Spearman measures rank correlation
- Robust to outliers
- More stable than Pearson for financial data

## License

MIT License - Feel free to use for educational purposes.

## Author

Machine Learning Student - University Final Project 2024

## Acknowledgments

- Numerai for providing the dataset and platform
- Scikit-learn and LightGBM teams for excellent ML libraries
- Open source community for tools and resources

---

**Good luck with your machine learning project!** =�=�

For questions or issues, consult the Numerai forum or documentation.
