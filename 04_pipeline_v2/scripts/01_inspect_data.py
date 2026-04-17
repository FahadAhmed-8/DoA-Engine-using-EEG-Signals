#!/usr/bin/env python3
"""
Step 1: Inspect all 24 .mat files.
Prints summary stats and saves to CSV.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import inspect_all_cases, load_case
from config.config import RESULTS_DIR, NUM_CASES
import numpy as np


def main():
    print("=" * 60)
    print("STEP 1: Dataset Inspection")
    print("=" * 60)

    # Inspect all cases
    summary = inspect_all_cases()
    print("\n--- Case Summary ---")
    print(summary.to_string(index=False))

    # Save to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "case_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")

    # Print detailed info for first case
    print("\n--- Detailed view: Case 1 ---")
    data = load_case(1)
    print(f"  Raw keys: {data['raw_keys']}")
    if 'eeg' in data:
        eeg = data['eeg']
        print(f"  EEG shape: {eeg.shape}")
        print(f"  EEG dtype: {eeg.dtype}")
        print(f"  EEG range: [{np.min(eeg):.2f}, {np.max(eeg):.2f}]")
        print(f"  EEG mean: {np.mean(eeg):.2f}, std: {np.std(eeg):.2f}")
    if 'bis' in data:
        bis = data['bis']
        print(f"  BIS shape: {bis.shape}")
        valid_bis = bis[bis >= 0]
        print(f"  BIS range (valid): [{np.min(valid_bis):.1f}, {np.max(valid_bis):.1f}]")
        print(f"  BIS missing (-1): {np.sum(bis == -1)}")

    # Count total usable windows
    total_windows = 0
    for i in range(1, NUM_CASES + 1):
        try:
            d = load_case(i)
            if 'eeg' in d and 'bis' in d:
                n_eeg = d['eeg'].shape[-1] // 640
                n_bis = len(d['bis'])
                total_windows += min(n_eeg, n_bis)
        except:
            pass

    print(f"\n--- Total estimated windows across all cases: {total_windows} ---")
    print("\nInspection complete!")


if __name__ == '__main__':
    main()
