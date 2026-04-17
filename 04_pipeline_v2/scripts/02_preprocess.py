#!/usr/bin/env python3
"""
Step 2: Preprocess all 24 cases.
Artifact removal → Notch filter → EMD → IMF reconstruction.
"""
import sys, os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_case
from data.preprocessor import preprocess_case, save_preprocessed
from config.config import NUM_CASES


def main():
    print("=" * 60)
    print("STEP 2: Preprocessing")
    print("=" * 60)

    total_start = time.time()
    success = 0
    failed = []

    for case_num in range(1, NUM_CASES + 1):
        print(f"\nCase {case_num}/{NUM_CASES}:")
        start = time.time()

        try:
            data = load_case(case_num)

            if 'eeg' not in data or 'bis' not in data:
                print(f"  SKIPPED: Missing EEG or BIS data")
                failed.append(case_num)
                continue

            eeg_preprocessed, bis_clean, metadata = preprocess_case(
                case_num, data['eeg'], data['bis']
            )

            save_preprocessed(case_num, eeg_preprocessed, bis_clean, metadata)

            elapsed = time.time() - start
            print(f"  Saved. Shape: {eeg_preprocessed.shape}, "
                  f"BIS: {len(bis_clean)}, Time: {elapsed:.1f}s")
            success += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(case_num)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Preprocessing complete: {success}/{NUM_CASES} cases")
    if failed:
        print(f"Failed cases: {failed}")
    print(f"Total time: {total_elapsed:.1f}s")


if __name__ == '__main__':
    main()
