"""
Baseline validation: 80/20 random train/test split across all patients.
"""
import numpy as np
from sklearn.model_selection import train_test_split

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import TRAIN_TEST_RATIO, RANDOM_SEED
from validation.metrics import compute_all_metrics
from features.fusion import normalize_features
from models.model_factory import create_model, train_model, predict


def split_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """80/20 random split."""
    return train_test_split(
        X, y,
        test_size=1 - TRAIN_TEST_RATIO,
        random_state=RANDOM_SEED
    )


def run_baseline_experiment(X: np.ndarray, y: np.ndarray,
                             model_name: str) -> dict:
    """
    Train/test split experiment for one model.

    Returns:
        dict with RMSE, MAE, Pearson_R, n_train, n_test
    """
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Normalize
    X_train_norm, X_test_norm, scaler = normalize_features(X_train, X_test)

    # Train
    model = create_model(model_name)
    train_model(model, X_train_norm, y_train)

    # Predict
    y_pred = predict(model, X_test_norm)

    # Metrics
    metrics = compute_all_metrics(y_test, y_pred)
    metrics['model'] = model_name
    metrics['n_train'] = len(y_train)
    metrics['n_test'] = len(y_test)

    return metrics
