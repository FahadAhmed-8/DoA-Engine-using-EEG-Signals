"""
Dataset I/O: Load .mat files (HDF5 v7.3 format) and inspect structure.
"""
import os
import numpy as np
import pandas as pd
import h5py

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATASET_DIR, NUM_CASES, SAMPLING_RATE


def load_case(case_num: int) -> dict:
    """
    Load a single case .mat file (MATLAB v7.3 / HDF5).

    Returns dict with keys:
        'eeg' : ndarray (1, n_samples) — single-channel EEG
        'bis' : ndarray (n_bis_samples,) — BIS values
        'raw_keys' : list of keys found in .mat file
    """
    filepath = os.path.join(DATASET_DIR, f"case{case_num}.mat")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Case file not found: {filepath}")

    result = {}
    with h5py.File(filepath, 'r') as f:
        keys = list(f.keys())
        result['raw_keys'] = keys

        # Load EEG
        if 'EEG' in f:
            eeg = np.array(f['EEG']).flatten().astype(float)
            result['eeg'] = eeg.reshape(1, -1)  # (1, n_samples)
        elif 'eeg' in f:
            eeg = np.array(f['eeg']).flatten().astype(float)
            result['eeg'] = eeg.reshape(1, -1)

        # Load BIS
        if 'bis' in f:
            result['bis'] = np.array(f['bis']).flatten().astype(float)
        elif 'BIS' in f:
            result['bis'] = np.array(f['BIS']).flatten().astype(float)

    return result


def inspect_case(case_num: int) -> dict:
    """Return summary stats for a single case."""
    try:
        data = load_case(case_num)
        eeg = data.get('eeg')
        bis = data.get('bis')

        info = {
            'case': case_num,
            'raw_keys': ', '.join(data['raw_keys']),
            'eeg_shape': str(eeg.shape) if eeg is not None else 'N/A',
            'n_channels': eeg.shape[0] if eeg is not None and eeg.ndim == 2 else 0,
            'n_eeg_samples': eeg.shape[-1] if eeg is not None else 0,
            'duration_sec': eeg.shape[-1] / SAMPLING_RATE if eeg is not None else 0,
            'duration_min': round(eeg.shape[-1] / SAMPLING_RATE / 60, 1) if eeg is not None else 0,
            'bis_count': len(bis) if bis is not None else 0,
            'bis_min': float(np.min(bis)) if bis is not None and len(bis) > 0 else None,
            'bis_max': float(np.max(bis)) if bis is not None and len(bis) > 0 else None,
            'bis_mean': round(float(np.mean(bis[bis >= 0])), 1) if bis is not None and np.any(bis >= 0) else None,
            'bis_missing': int(np.sum(bis == -1)) if bis is not None else 0,
            'status': 'OK'
        }
    except Exception as e:
        info = {'case': case_num, 'status': f'ERROR: {e}'}

    return info


def inspect_all_cases() -> pd.DataFrame:
    """Scan all 24 cases, return summary DataFrame."""
    rows = []
    for i in range(1, NUM_CASES + 1):
        rows.append(inspect_case(i))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = inspect_all_cases()
    print(df.to_string(index=False))
