"""
EEG Signal Preprocessing:
  1. Artifact removal
  2. 60 Hz notch filter
  3. EMD decomposition (with fallback)
  4. IMF reconstruction
"""
import os
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import warnings
import json

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    SAMPLING_RATE, NOTCH_FREQ, NOTCH_Q, BUTTERWORTH_ORDER,
    ARTIFACT_THRESHOLD_STD, ARTIFACT_FILL_WINDOW,
    BIS_MISSING_VALUE, BIS_FILL_WINDOW,
    BANDPASS_LOW, BANDPASS_HIGH, IMF_INDICES,
    PREPROCESS_DIR, LOGS_DIR
)


def remove_eeg_artifacts(eeg: np.ndarray, threshold_std: float = ARTIFACT_THRESHOLD_STD,
                         fill_window: int = ARTIFACT_FILL_WINDOW) -> np.ndarray:
    """
    Detect and replace artifact samples in EEG.
    Artifacts: samples > threshold_std × std from channel mean.
    Replacement: mean of surrounding clean samples.

    Args:
        eeg: (n_channels, n_samples)
    Returns:
        cleaned EEG (same shape)
    """
    eeg_clean = eeg.copy()
    n_channels, n_samples = eeg_clean.shape

    for ch in range(n_channels):
        signal = eeg_clean[ch]
        mean_val = np.mean(signal)
        std_val = np.std(signal)

        if std_val == 0:
            continue

        # Find artifact indices
        artifact_mask = np.abs(signal - mean_val) > threshold_std * std_val
        artifact_indices = np.where(artifact_mask)[0]

        for idx in artifact_indices:
            # Get surrounding clean samples
            start = max(0, idx - fill_window // 2)
            end = min(n_samples, idx + fill_window // 2)
            surrounding = signal[start:end]
            clean_surrounding = surrounding[~artifact_mask[start:end]]

            if len(clean_surrounding) > 0:
                signal[idx] = np.mean(clean_surrounding)
            else:
                signal[idx] = mean_val

        eeg_clean[ch] = signal

    return eeg_clean


def fill_bis_gaps(bis: np.ndarray, missing_value: float = BIS_MISSING_VALUE,
                  fill_window: int = BIS_FILL_WINDOW) -> np.ndarray:
    """
    Replace missing BIS values (marked as -1) with mean of surrounding valid values.
    """
    bis_clean = bis.copy()
    missing_mask = (bis_clean == missing_value) | np.isnan(bis_clean)
    missing_indices = np.where(missing_mask)[0]

    for idx in missing_indices:
        start = max(0, idx - fill_window // 2)
        end = min(len(bis_clean), idx + fill_window // 2 + 1)
        surrounding = bis_clean[start:end]
        valid = surrounding[(surrounding != missing_value) & ~np.isnan(surrounding)]

        if len(valid) > 0:
            bis_clean[idx] = np.mean(valid)
        else:
            # Expand search
            valid_all = bis_clean[(bis_clean != missing_value) & ~np.isnan(bis_clean)]
            bis_clean[idx] = np.mean(valid_all) if len(valid_all) > 0 else 50.0

    return bis_clean


def apply_notch_filter(signal: np.ndarray, fs: int = SAMPLING_RATE,
                       freq: float = NOTCH_FREQ, Q: float = NOTCH_Q) -> np.ndarray:
    """
    Apply 60 Hz notch filter using second-order IIR.
    Zero-phase filtering with filtfilt.

    Args:
        signal: (n_channels, n_samples) or (n_samples,)
    Returns:
        filtered signal (same shape)
    """
    b, a = iirnotch(freq, Q, fs)

    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        filtered = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            filtered[ch] = filtfilt(b, a, signal[ch])
        return filtered


def emd_decomposition(eeg: np.ndarray, max_imf: int = 6) -> tuple:
    """
    Attempt EMD decomposition with 3-tier fallback:
      Tier 1: Per-channel EMD (PyEMD)
      Tier 2: Bandpass filter fallback

    Args:
        eeg: (n_channels, n_samples)
    Returns:
        (reconstructed_signal, method_used)
    """
    n_channels, n_samples = eeg.shape

    # Tier 1: Per-channel EMD
    try:
        from PyEMD import EMD

        emd = EMD()
        reconstructed = np.zeros_like(eeg)

        for ch in range(n_channels):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imfs = emd(eeg[ch], max_imf=max_imf)

            # Reconstruct from selected IMFs
            recon = np.zeros(n_samples)
            for idx in IMF_INDICES:
                if idx < len(imfs):
                    recon += imfs[idx]
            reconstructed[ch] = recon

        return reconstructed, 'EMD'

    except Exception as e:
        print(f"    EMD failed: {e}. Trying bandpass fallback...")

    # Tier 2: Bandpass filter fallback
    try:
        nyquist = SAMPLING_RATE / 2
        low = BANDPASS_LOW / nyquist
        high = BANDPASS_HIGH / nyquist

        # Clamp to valid range
        low = max(low, 0.001)
        high = min(high, 0.999)

        b, a = butter(BUTTERWORTH_ORDER, [low, high], btype='band')
        reconstructed = np.zeros_like(eeg)

        for ch in range(n_channels):
            reconstructed[ch] = filtfilt(b, a, eeg[ch])

        return reconstructed, 'Bandpass_Fallback'

    except Exception as e:
        print(f"    Bandpass fallback also failed: {e}. Returning filtered signal.")
        return eeg, 'No_Decomposition'


def preprocess_case(case_num: int, eeg: np.ndarray, bis: np.ndarray,
                    verbose: bool = True) -> tuple:
    """
    Full preprocessing pipeline for one case:
      1. Artifact removal
      2. BIS gap filling
      3. Notch filter
      4. EMD decomposition + reconstruction

    Args:
        case_num: patient case number
        eeg: (n_channels, n_samples)
        bis: (n_bis_samples,)

    Returns:
        (preprocessed_eeg, cleaned_bis, metadata_dict)
    """
    metadata = {'case': case_num}

    # Step 1: Artifact removal
    if verbose:
        print(f"  [1/4] Removing EEG artifacts...")
    n_artifacts_before = np.sum(
        np.abs(eeg - np.mean(eeg, axis=1, keepdims=True)) >
        ARTIFACT_THRESHOLD_STD * np.std(eeg, axis=1, keepdims=True)
    )
    eeg_clean = remove_eeg_artifacts(eeg)
    metadata['artifacts_removed'] = int(n_artifacts_before)

    # Step 2: Fill BIS gaps
    if verbose:
        print(f"  [2/4] Filling BIS gaps...")
    n_missing = int(np.sum((bis == BIS_MISSING_VALUE) | np.isnan(bis)))
    bis_clean = fill_bis_gaps(bis)
    metadata['bis_gaps_filled'] = n_missing

    # Step 3: Notch filter (60 Hz)
    if verbose:
        print(f"  [3/4] Applying 60 Hz notch filter...")
    eeg_filtered = apply_notch_filter(eeg_clean)

    # Step 4: EMD decomposition
    if verbose:
        print(f"  [4/4] EMD decomposition...")
    eeg_reconstructed, emd_method = emd_decomposition(eeg_filtered)
    metadata['emd_method'] = emd_method

    if verbose:
        print(f"  Done. EMD method: {emd_method}")

    return eeg_reconstructed, bis_clean, metadata


def save_preprocessed(case_num: int, eeg: np.ndarray, bis: np.ndarray,
                      metadata: dict):
    """Save preprocessed data to disk."""
    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    np.save(os.path.join(PREPROCESS_DIR, f"case{case_num}_eeg.npy"), eeg)
    np.save(os.path.join(PREPROCESS_DIR, f"case{case_num}_bis.npy"), bis)

    # Append metadata to log
    log_path = os.path.join(LOGS_DIR, "preprocessing_metadata.json")
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            all_meta = json.load(f)
    else:
        all_meta = {}

    all_meta[f"case{case_num}"] = metadata
    with open(log_path, 'w') as f:
        json.dump(all_meta, f, indent=2)


def load_preprocessed(case_num: int) -> tuple:
    """Load preprocessed data from disk."""
    eeg = np.load(os.path.join(PREPROCESS_DIR, f"case{case_num}_eeg.npy"))
    bis = np.load(os.path.join(PREPROCESS_DIR, f"case{case_num}_bis.npy"))
    return eeg, bis
