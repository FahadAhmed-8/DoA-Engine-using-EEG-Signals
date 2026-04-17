"""
Evaluation metrics: RMSE, MAE, Pearson correlation.
"""
import numpy as np
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    if len(y_true) < 3:
        return 0.0
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all metrics at once."""
    return {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'Pearson_R': pearson_r(y_true, y_pred),
    }
