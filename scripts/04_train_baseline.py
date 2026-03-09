#!/usr/bin/env python3
"""
Step 4: Baseline training with 80/20 train/test split.
All 4 models × 7 feature combos = 28 experiments.
"""
import sys, os
import time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import NUM_CASES, FEATURES_DIR, RESULTS_DIR, FEATURE_COMBOS, TARGET_RMSE
from validation.train_test_split import run_baseline_experiment
from models.model_factory import get_model_names


def load_combo_data(combo_name: str) -> tuple:
    """Load and combine feature data for all cases for a given combo."""
    X_parts = []
    y_parts = []

    for case_num in range(1, NUM_CASES + 1):
        csv_path = os.path.join(FEATURES_DIR, f"case{case_num}_{combo_name}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        feat_cols = [c for c in df.columns if c.startswith('feat_')]
        X = df[feat_cols].values
        y = df['BIS'].values

        # Remove NaN
        valid = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
        X_parts.append(X[valid])
        y_parts.append(y[valid])

    if len(X_parts) == 0:
        return np.array([]), np.array([])

    return np.vstack(X_parts), np.concatenate(y_parts)


def main():
    print("=" * 60)
    print("STEP 4: Baseline Training (80/20 Split)")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []
    total_start = time.time()

    combos = list(FEATURE_COMBOS.keys())
    models = get_model_names()

    for combo_name in combos:
        print(f"\n--- Feature Combo: {combo_name} ---")
        X, y = load_combo_data(combo_name)

        if len(y) == 0:
            print(f"  No data available, skipping.")
            continue

        print(f"  Total samples: {len(y)}, Features: {X.shape[1]}")

        for model_name in models:
            start = time.time()
            try:
                metrics = run_baseline_experiment(X, y, model_name)
                metrics['feature_combo'] = combo_name
                elapsed = time.time() - start

                # Check if beats baseline
                beat = "YES" if metrics['RMSE'] < TARGET_RMSE else "no"
                print(f"  {model_name:4s}: RMSE={metrics['RMSE']:.2f}, "
                      f"MAE={metrics['MAE']:.2f}, R={metrics['Pearson_R']:.3f}, "
                      f"Beat baseline: {beat} ({elapsed:.1f}s)")

                all_results.append(metrics)

            except Exception as e:
                print(f"  {model_name}: ERROR — {e}")

    # Save results
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    results_df.to_csv(csv_path, index=False)

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"Baseline complete: {len(all_results)} experiments")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"\nResults saved to: {csv_path}")

    # Print best result
    if len(results_df) > 0:
        best = results_df.loc[results_df['RMSE'].idxmin()]
        print(f"\nBest result: {best['feature_combo']} + {best['model']}")
        print(f"  RMSE={best['RMSE']:.2f}, MAE={best['MAE']:.2f}, R={best['Pearson_R']:.3f}")
        print(f"  Target RMSE: {TARGET_RMSE}")


if __name__ == '__main__':
    main()
