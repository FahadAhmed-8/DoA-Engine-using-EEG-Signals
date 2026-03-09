#!/usr/bin/env python3
"""
Step 5: Leave-One-Patient-Out (LOPO) Cross-Validation.
Runs on all 7 combos × 4 models with concise output.
"""
import sys, os
import time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import NUM_CASES, FEATURES_DIR, RESULTS_DIR, FEATURE_COMBOS
from validation.lopo import run_lopo_cv, summarize_lopo
from models.model_factory import get_model_names


def load_case_features(combo_name: str) -> dict:
    """Load feature data per case for LOPO."""
    case_features = {}

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
        if np.sum(valid) > 0:
            case_features[case_num] = (X[valid], y[valid])

    return case_features


def main():
    print("=" * 60)
    print("STEP 5: LOPO Cross-Validation")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_summaries = []
    all_folds = []
    total_start = time.time()

    combos = list(FEATURE_COMBOS.keys())
    models = get_model_names()

    for combo_name in combos:
        print(f"\n--- {combo_name} ---")

        case_features = load_case_features(combo_name)
        if len(case_features) < 3:
            print(f"  Too few cases, skipping.")
            continue

        for model_name in models:
            start = time.time()

            try:
                fold_df = run_lopo_cv(case_features, model_name, verbose=False)
                fold_df['feature_combo'] = combo_name
                fold_df['model'] = model_name
                all_folds.append(fold_df)

                summary = summarize_lopo(fold_df)
                summary['feature_combo'] = combo_name
                summary['model'] = model_name
                all_summaries.append(summary)

                elapsed = time.time() - start
                print(f"  {model_name:4s}: RMSE={summary['RMSE_mean']:.2f}±{summary['RMSE_std']:.2f}, "
                      f"MAE={summary['MAE_mean']:.2f}±{summary['MAE_std']:.2f}, "
                      f"R={summary['Pearson_R_mean']:.3f} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  {model_name}: ERROR — {e}")

    # Save results
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(RESULTS_DIR, "lopo_results.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nLOPO summary saved to: {summary_path}")

    if all_folds:
        folds_df = pd.concat(all_folds, ignore_index=True)
        folds_path = os.path.join(RESULTS_DIR, "lopo_fold_details.csv")
        folds_df.to_csv(folds_path, index=False)

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Print best
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        best = summary_df.loc[summary_df['RMSE_mean'].idxmin()]
        print(f"\nBest LOPO: {best['feature_combo']} + {best['model']}")
        print(f"  RMSE={best['RMSE_mean']:.2f}±{best['RMSE_std']:.2f}")


if __name__ == '__main__':
    main()
