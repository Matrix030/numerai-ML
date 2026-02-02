"""
Numerai Prediction Utility Module
==================================

This module provides reusable functions for working with Numerai data and models.
Can be imported into notebooks or used in production scripts.

Author: ML Student
Date: 2024
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class NumeraiMetrics:
    """
    Collection of evaluation metrics for Numerai predictions.
    """

    @staticmethod
    def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Spearman rank correlation.
        This is the PRIMARY metric for Numerai.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Correlation coefficient [-1, 1], higher is better
        """
        return spearmanr(y_true, y_pred)[0]

    @staticmethod
    def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Pearson linear correlation.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Correlation coefficient [-1, 1], higher is better
        """
        return pearsonr(y_true, y_pred)[0]

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            MSE, lower is better
        """
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def sharpe_ratio(era_correlations: np.ndarray) -> float:
        """
        Calculate Sharpe-like ratio: mean correlation / std correlation across eras.
        Measures consistency of performance.

        Args:
            era_correlations: Array of correlation values per era

        Returns:
            Sharpe ratio, higher is better
        """
        if len(era_correlations) == 0 or np.std(era_correlations) == 0:
            return 0.0
        return np.mean(era_correlations) / np.std(era_correlations)

    @staticmethod
    def evaluate_model(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        eras: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with all metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            eras: Era labels for grouping
            model_name: Name of model for display

        Returns:
            Dictionary containing all evaluation metrics
        """
        # Overall metrics
        spearman = NumeraiMetrics.spearman_correlation(y_true, y_pred)
        pearson = NumeraiMetrics.pearson_correlation(y_true, y_pred)
        mse_val = NumeraiMetrics.mse(y_true, y_pred)

        # Per-era correlations
        era_corrs = []
        unique_eras = np.unique(eras)

        for era in unique_eras:
            era_mask = eras == era
            if np.sum(era_mask) > 1:
                era_corr = NumeraiMetrics.spearman_correlation(
                    y_true[era_mask],
                    y_pred[era_mask]
                )
                if not np.isnan(era_corr):
                    era_corrs.append(era_corr)

        # Sharpe ratio
        sharpe = NumeraiMetrics.sharpe_ratio(era_corrs)

        return {
            'model': model_name,
            'spearman': spearman,
            'pearson': pearson,
            'mse': mse_val,
            'sharpe': sharpe,
            'mean_era_corr': np.mean(era_corrs),
            'std_era_corr': np.std(era_corrs),
            'era_corrs': era_corrs
        }


class NumeraiDataLoader:
    """
    Utility class for loading and preprocessing Numerai data.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing Numerai parquet files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def load_train_data(self) -> pd.DataFrame:
        """Load training data."""
        return pd.read_parquet(self.data_dir / "train.parquet")

    def load_validation_data(self) -> pd.DataFrame:
        """Load validation data."""
        return pd.read_parquet(self.data_dir / "validation.parquet")

    def load_live_data(self) -> pd.DataFrame:
        """Load live tournament data."""
        return pd.read_parquet(self.data_dir / "live.parquet")

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> List[str]:
        """
        Extract feature column names.

        Args:
            df: Numerai dataframe

        Returns:
            List of feature column names
        """
        return [c for c in df.columns if c.startswith("feature")]

    @staticmethod
    def get_target_columns(df: pd.DataFrame) -> List[str]:
        """
        Extract target column names.

        Args:
            df: Numerai dataframe

        Returns:
            List of target column names
        """
        return [c for c in df.columns if c.startswith("target")]

    @staticmethod
    def era_split(
        df: pd.DataFrame,
        split_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data based on eras (time-aware split).

        Args:
            df: Numerai dataframe
            split_ratio: Fraction of eras for training

        Returns:
            (train_df, test_df)
        """
        unique_eras = sorted(df['era'].unique())
        n_eras = len(unique_eras)
        split_idx = int(n_eras * split_ratio)

        train_eras = unique_eras[:split_idx]
        test_eras = unique_eras[split_idx:]

        train_df = df[df['era'].isin(train_eras)]
        test_df = df[df['era'].isin(test_eras)]

        return train_df, test_df


class NumeraiEnsemble:
    """
    Ensemble model class for combining multiple predictions.
    """

    def __init__(self, models: List[Any], weights: List[float] = None):
        """
        Initialize ensemble.

        Args:
            models: List of trained models
            weights: Optional weights for each model (must sum to 1)
        """
        self.models = models

        if weights is None:
            # Equal weighting
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            self.weights = weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            X: Feature matrix

        Returns:
            Weighted average predictions
        """
        predictions = np.zeros(len(X))

        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions += weight * pred

        return predictions

    @classmethod
    def from_performance(
        cls,
        models: List[Any],
        performance_scores: List[float]
    ):
        """
        Create ensemble with weights based on model performance.

        Args:
            models: List of trained models
            performance_scores: Performance metric for each model (higher is better)

        Returns:
            NumeraiEnsemble instance
        """
        # Normalize scores to create weights
        total = sum(performance_scores)
        weights = [s / total for s in performance_scores]

        return cls(models, weights)


class SubmissionFormatter:
    """
    Utility for formatting and validating Numerai submissions.
    """

    @staticmethod
    def create_submission(
        ids: pd.Series,
        predictions: np.ndarray,
        output_path: str = "submission.csv"
    ) -> pd.DataFrame:
        """
        Create submission file in Numerai format.

        Args:
            ids: ID column from live data
            predictions: Model predictions
            output_path: Path to save CSV

        Returns:
            Submission dataframe
        """
        submission = pd.DataFrame({
            'id': ids,
            'prediction': predictions
        })

        # Validate
        assert len(submission) > 0, "Empty submission"
        assert not submission['prediction'].isnull().any(), "Missing predictions"
        assert submission['id'].nunique() == len(submission), "Duplicate IDs"

        # Save
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        print(f"Total predictions: {len(submission):,}")

        return submission

    @staticmethod
    def validate_submission(submission_path: str) -> bool:
        """
        Validate submission file format.

        Args:
            submission_path: Path to submission CSV

        Returns:
            True if valid, raises exception otherwise
        """
        df = pd.read_csv(submission_path)

        # Check columns
        assert 'id' in df.columns, "Missing 'id' column"
        assert 'prediction' in df.columns, "Missing 'prediction' column"

        # Check for missing values
        assert not df['prediction'].isnull().any(), "Missing predictions"

        # Check for duplicates
        assert df['id'].nunique() == len(df), "Duplicate IDs"

        print(" Submission is valid!")
        return True


def print_model_comparison(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print formatted comparison table of model results.

    Args:
        results: Dictionary mapping model names to their evaluation results
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<25} {'Spearman':>10} {'Pearson':>10} {'MSE':>10} {'Sharpe':>10}")
    print("-" * 80)

    # Sort by Spearman (primary metric)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['spearman'],
        reverse=True
    )

    for model_name, metrics in sorted_models:
        print(
            f"{model_name:<25} "
            f"{metrics['spearman']:>10.6f} "
            f"{metrics['pearson']:>10.6f} "
            f"{metrics['mse']:>10.6f} "
            f"{metrics['sharpe']:>10.6f}"
        )

    print("=" * 80)
    print(f"\n Best Model: {sorted_models[0][0]}")
    print(f"   Spearman: {sorted_models[0][1]['spearman']:.6f}")


# Convenience function
def quick_evaluate(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    eras_test: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Quick evaluation of a single model.

    Args:
        model: Trained model with .predict() method
        X_test: Test features
        y_test: Test targets
        eras_test: Test era labels
        model_name: Model name for display

    Returns:
        Evaluation metrics dictionary
    """
    y_pred = model.predict(X_test)
    results = NumeraiMetrics.evaluate_model(y_test, y_pred, eras_test, model_name)

    print(f"\n{model_name} Evaluation:")
    print(f"  Spearman: {results['spearman']:.6f}")
    print(f"  Pearson:  {results['pearson']:.6f}")
    print(f"  MSE:      {results['mse']:.6f}")
    print(f"  Sharpe:   {results['sharpe']:.6f}")

    return results


if __name__ == "__main__":
    print("Numerai Utilities Module")
    print("=" * 60)
    print("\nAvailable classes:")
    print("  - NumeraiMetrics: Evaluation metrics")
    print("  - NumeraiDataLoader: Data loading utilities")
    print("  - NumeraiEnsemble: Ensemble model")
    print("  - SubmissionFormatter: Submission handling")
    print("\nImport this module in your notebooks:")
    print("  from numerai_utils import *")
