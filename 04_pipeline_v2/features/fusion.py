"""
Feature fusion: Combine entropy measures into 7 feature combinations,
average across channels, normalize, handle NaN.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import FEATURE_COMBOS


def average_across_channels(entropy_dict: dict) -> dict:
    """
    Average entropy values across channels for each window.

    Args:
        entropy_dict: {'SampEn': (n_windows, n_channels), ...}

    Returns:
        {'SampEn': (n_windows,), 'ApEn': (n_windows,), 'PE': (n_windows,)}
    """
    result = {}
    for key, vals in entropy_dict.items():
        if vals.ndim == 2:
            # Use nanmean to handle NaN channels
            result[key] = np.nanmean(vals, axis=1)
        else:
            result[key] = vals
    return result


def create_feature_matrix(entropy_avg: dict, combo_name: str) -> np.ndarray:
    """
    Create feature matrix for a given combination.

    Args:
        entropy_avg: {'SampEn': (n_windows,), 'ApEn': (n_windows,), 'PE': (n_windows,)}
        combo_name: key from FEATURE_COMBOS

    Returns:
        X: (n_windows, n_features)
    """
    measures = FEATURE_COMBOS[combo_name]
    features = [entropy_avg[m] for m in measures]
    return np.column_stack(features)


def handle_nan(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Remove rows with NaN values.

    Returns:
        (X_clean, y_clean, valid_mask)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    nan_mask = np.any(np.isnan(X), axis=1) | np.isnan(y)
    valid_mask = ~nan_mask

    return X[valid_mask], y[valid_mask], valid_mask


def normalize_features(X_train: np.ndarray, X_test: np.ndarray = None) -> tuple:
    """
    MinMax normalize features. Fit on train, transform both.

    Returns:
        (X_train_norm, X_test_norm, scaler)
        If X_test is None, returns (X_train_norm, None, scaler)
    """
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)

    X_test_norm = None
    if X_test is not None:
        X_test_norm = scaler.transform(X_test)

    return X_train_norm, X_test_norm, scaler


def prepare_features_for_case(entropy_dict: dict, bis: np.ndarray,
                               combo_name: str) -> tuple:
    """
    Full feature preparation for one case and one combo.

    Returns:
        (X, y) — cleaned, not yet normalized (normalization happens at train time)
    """
    avg = average_across_channels(entropy_dict)
    X = create_feature_matrix(avg, combo_name)
    X_clean, y_clean, _ = handle_nan(X, bis)
    return X_clean, y_clean


def prepare_all_combos_for_case(entropy_dict: dict, bis: np.ndarray) -> dict:
    """
    Prepare feature matrices for all 7 combinations.

    Returns:
        dict: combo_name -> (X, y)
    """
    result = {}
    for combo_name in FEATURE_COMBOS:
        X, y = prepare_features_for_case(entropy_dict, bis, combo_name)
        result[combo_name] = (X, y)
    return result
