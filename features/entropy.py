"""
Entropy feature extraction: SampEn, ApEn, PE.
Uses the antropy library.
"""
import numpy as np
import warnings

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    SAMPLE_ENTROPY_ORDER, APP_ENTROPY_ORDER,
    PERM_ENTROPY_ORDER, PERM_ENTROPY_NORMALIZE
)


def sample_entropy(signal: np.ndarray, order: int = SAMPLE_ENTROPY_ORDER) -> float:
    """
    Compute Sample Entropy using antropy.
    r (tolerance) = 0.2 × std(signal) — set internally by antropy.
    """
    import antropy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = antropy.sample_entropy(signal, order=order)
        if np.isnan(val) or np.isinf(val):
            return np.nan
        return float(val)
    except Exception:
        return np.nan


def app_entropy(signal: np.ndarray, order: int = APP_ENTROPY_ORDER) -> float:
    """Compute Approximate Entropy using antropy."""
    import antropy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = antropy.app_entropy(signal, order=order)
        if np.isnan(val) or np.isinf(val):
            return np.nan
        return float(val)
    except Exception:
        return np.nan


def perm_entropy(signal: np.ndarray, order: int = PERM_ENTROPY_ORDER,
                 normalize: bool = PERM_ENTROPY_NORMALIZE) -> float:
    """Compute Permutation Entropy using antropy."""
    import antropy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = antropy.perm_entropy(signal, order=order, normalize=normalize)
        if np.isnan(val) or np.isinf(val):
            return np.nan
        return float(val)
    except Exception:
        return np.nan


def extract_entropies_window(window: np.ndarray) -> dict:
    """
    Extract all three entropy measures for a single window.

    Args:
        window: (n_channels, samples_per_window) or (samples_per_window,)

    Returns:
        dict with keys 'SampEn', 'ApEn', 'PE', each a list of values (one per channel)
    """
    if window.ndim == 1:
        window = window.reshape(1, -1)

    n_channels = window.shape[0]
    result = {'SampEn': [], 'ApEn': [], 'PE': []}

    for ch in range(n_channels):
        sig = window[ch]

        # Skip constant signals
        if np.std(sig) < 1e-10:
            result['SampEn'].append(np.nan)
            result['ApEn'].append(np.nan)
            result['PE'].append(np.nan)
            continue

        result['SampEn'].append(sample_entropy(sig))
        result['ApEn'].append(app_entropy(sig))
        result['PE'].append(perm_entropy(sig))

    return result


def extract_entropies_case(X: np.ndarray) -> dict:
    """
    Extract entropy features for all windows in a case.

    Args:
        X: (n_windows, n_channels, samples_per_window)

    Returns:
        dict with keys 'SampEn', 'ApEn', 'PE', each (n_windows, n_channels)
    """
    n_windows = X.shape[0]
    n_channels = X.shape[1]

    all_sampen = np.zeros((n_windows, n_channels))
    all_apen = np.zeros((n_windows, n_channels))
    all_pe = np.zeros((n_windows, n_channels))

    for i in range(n_windows):
        ent = extract_entropies_window(X[i])
        all_sampen[i] = ent['SampEn']
        all_apen[i] = ent['ApEn']
        all_pe[i] = ent['PE']

    return {
        'SampEn': all_sampen,
        'ApEn': all_apen,
        'PE': all_pe,
    }
