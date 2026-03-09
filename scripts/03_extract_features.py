#!/usr/bin/env python3
"""
Step 3: Extract entropy features for all cases.
SampEn, ApEn, PE → 7 feature combinations → save as CSV.
"""
import sys, os
import time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessor import load_preprocessed
from data.segmenter import segment_case
from features.entropy import extract_entropies_case
from features.fusion import average_across_channels, prepare_all_combos_for_case
from config.config import NUM_CASES, FEATURES_DIR, FEATURE_COMBOS
import pickle


def main():
    print("=" * 60)
    print("STEP 3: Feature Extraction")
    print("=" * 60)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    total_start = time.time()
    success = 0

    for case_num in range(1, NUM_CASES + 1):
        print(f"\nCase {case_num}/{NUM_CASES}:")
        start = time.time()

        try:
            # Load preprocessed
            eeg, bis = load_preprocessed(case_num)
            print(f"  Loaded: EEG {eeg.shape}, BIS {bis.shape}")

            # Segment
            X, y = segment_case(eeg, bis)
            print(f"  Segments: {X.shape[0]} windows, {X.shape[1]} channels, {X.shape[2]} samples/window")

            if len(y) == 0:
                print(f"  SKIPPED: No valid windows")
                continue

            # Extract entropies
            print(f"  Computing entropies (this may take a moment)...")
            entropy_dict = extract_entropies_case(X)

            # Save raw entropy per channel (for channel analysis later)
            raw_path = os.path.join(FEATURES_DIR, f"case{case_num}_entropy_raw.pkl")
            pickle.dump({'entropy': entropy_dict, 'bis': y, 'n_channels': X.shape[1]},
                       open(raw_path, 'wb'))

            # Prepare all 7 combos
            all_combos = prepare_all_combos_for_case(entropy_dict, y)

            for combo_name, (X_feat, y_feat) in all_combos.items():
                csv_path = os.path.join(FEATURES_DIR, f"case{case_num}_{combo_name}.csv")
                df = pd.DataFrame(X_feat, columns=[f"feat_{i}" for i in range(X_feat.shape[1])])
                df['BIS'] = y_feat
                df.to_csv(csv_path, index=False)

            elapsed = time.time() - start
            n_nan = sum(np.any(np.isnan(X_feat)) for _, (X_feat, _) in all_combos.items())
            print(f"  Done. {len(y)} windows, {elapsed:.1f}s, "
                  f"NaN combos: {n_nan}/7")
            success += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Feature extraction complete: {success}/{NUM_CASES} cases")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Features saved to: {FEATURES_DIR}")


if __name__ == '__main__':
    main()
