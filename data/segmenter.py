"""
Segmentation: Create 5-second non-overlapping windows from preprocessed EEG,
aligned with BIS labels.
"""
import os
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SAMPLING_RATE, SAMPLES_PER_WINDOW, BIS_INTERVAL


def align_eeg_bis(eeg: np.ndarray, bis: np.ndarray) -> tuple:
    """
    Align EEG and BIS data. BIS is recorded every 5 seconds.
    EEG is at 128 Hz. Each BIS value corresponds to one 5-sec EEG window.

    Args:
        eeg: (n_channels, n_samples)
        bis: (n_bis_values,)

    Returns:
        (eeg_trimmed, bis_trimmed) — aligned to the shorter duration
    """
    n_samples = eeg.shape[-1]
    n_windows_eeg = n_samples // SAMPLES_PER_WINDOW
    n_windows_bis = len(bis)

    # Use the minimum number of complete windows
    n_windows = min(n_windows_eeg, n_windows_bis)

    eeg_trimmed = eeg[:, :n_windows * SAMPLES_PER_WINDOW]
    bis_trimmed = bis[:n_windows]

    return eeg_trimmed, bis_trimmed


def create_windows(eeg: np.ndarray, bis: np.ndarray) -> tuple:
    """
    Create non-overlapping 5-second windows from aligned EEG and BIS.

    Args:
        eeg: (n_channels, n_eeg_samples) — aligned
        bis: (n_windows,) — one BIS per window

    Returns:
        X: (n_windows, n_channels, samples_per_window)  — EEG segments
        y: (n_windows,) — BIS labels
    """
    n_channels = eeg.shape[0]
    n_samples = eeg.shape[1]
    n_windows = n_samples // SAMPLES_PER_WINDOW

    # Truncate to exact number of windows
    n_windows = min(n_windows, len(bis))
    eeg = eeg[:, :n_windows * SAMPLES_PER_WINDOW]
    bis = bis[:n_windows]

    # Reshape EEG into windows
    X = eeg[:, :n_windows * SAMPLES_PER_WINDOW].reshape(n_channels, n_windows, SAMPLES_PER_WINDOW)
    X = X.transpose(1, 0, 2)  # (n_windows, n_channels, samples_per_window)

    # Filter out windows with invalid BIS (0 or NaN)
    valid_mask = (bis > 0) & (bis <= 100) & ~np.isnan(bis)
    X = X[valid_mask]
    y = bis[valid_mask]

    return X, y


def segment_case(eeg: np.ndarray, bis: np.ndarray) -> tuple:
    """
    Full segmentation pipeline for one case.

    Args:
        eeg: (n_channels, n_samples) preprocessed EEG
        bis: (n_bis_values,) cleaned BIS

    Returns:
        X: (n_windows, n_channels, 640) EEG segments
        y: (n_windows,) BIS labels
    """
    eeg_aligned, bis_aligned = align_eeg_bis(eeg, bis)
    X, y = create_windows(eeg_aligned, bis_aligned)
    return X, y
