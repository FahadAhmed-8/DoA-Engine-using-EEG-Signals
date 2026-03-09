"""
Leave-One-Patient-Out (LOPO) Cross-Validation.
24 folds: train on 23 patients, test on 1.
"""
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import NUM_CASES
from validation.metrics import compute_all_metrics
from features.fusion import normalize_features
from models.model_factory import create_model, train_model, predict


def run_lopo_cv(case_features: dict, model_name: str,
                verbose: bool = True) -> pd.DataFrame:
    """
    Leave-One-Patient-Out cross-validation.

    Args:
        case_features: dict of case_num -> (X, y)
            X: (n_windows, n_features), y: (n_windows,)
        model_name: 'ANN', 'RF', 'XGB', or 'SVR'
        verbose: print per-fold progress

    Returns:
        DataFrame with columns [fold, test_case, RMSE, MAE, Pearson_R, n_train, n_test]
    """
    case_nums = sorted(case_features.keys())
    fold_results = []

    for fold_idx, test_case in enumerate(case_nums):
        # Split: test = one patient, train = rest
        X_test, y_test = case_features[test_case]

        if len(y_test) == 0:
            if verbose:
                print(f"  Fold {fold_idx+1}/{len(case_nums)}: case{test_case} — SKIPPED (no data)")
            continue

        # Combine training data
        X_train_parts = []
        y_train_parts = []
        for case_num in case_nums:
            if case_num != test_case:
                X_c, y_c = case_features[case_num]
                if len(y_c) > 0:
                    X_train_parts.append(X_c)
                    y_train_parts.append(y_c)

        if len(X_train_parts) == 0:
            continue

        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)

        # Normalize (fit on train only)
        X_train_norm, X_test_norm, _ = normalize_features(X_train, X_test)

        # Train and predict
        model = create_model(model_name)
        train_model(model, X_train_norm, y_train)
        y_pred = predict(model, X_test_norm)

        # Metrics
        metrics = compute_all_metrics(y_test, y_pred)
        metrics['fold'] = fold_idx + 1
        metrics['test_case'] = test_case
        metrics['n_train'] = len(y_train)
        metrics['n_test'] = len(y_test)
        fold_results.append(metrics)

        if verbose:
            print(f"  Fold {fold_idx+1}/{len(case_nums)}: case{test_case} — "
                  f"RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, "
                  f"R={metrics['Pearson_R']:.3f}")

    return pd.DataFrame(fold_results)


def summarize_lopo(fold_df: pd.DataFrame) -> dict:
    """
    Aggregate LOPO results: mean ± std across folds.
    """
    return {
        'RMSE_mean': fold_df['RMSE'].mean(),
        'RMSE_std': fold_df['RMSE'].std(),
        'MAE_mean': fold_df['MAE'].mean(),
        'MAE_std': fold_df['MAE'].std(),
        'Pearson_R_mean': fold_df['Pearson_R'].mean(),
        'Pearson_R_std': fold_df['Pearson_R'].std(),
        'n_folds': len(fold_df),
    }
