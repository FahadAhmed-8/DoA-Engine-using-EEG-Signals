#!/usr/bin/env python3
"""
Step 8: Complete missing LOPO configurations (resumable).

Why this exists
---------------
`05_train_lopo.py` is designed to run all 7 feature combos * 4 models = 28
configurations in a single run, but it saves results only after the whole loop
finishes. If the run is interrupted (power blip, Ctrl-C, hitting a slow model),
nothing is saved and you redo everything.

The v2 outputs on disk contain only 6 of the 28 LOPO configs (RF and XGB on
SampEn/PE/All), meaning 22 configs are missing. This script fills that gap.

What it does
------------
1. Loads the existing `outputs/results/lopo_results.csv` and
   `outputs/results/lopo_fold_details.csv` (if they exist).
2. Computes the set of missing (feature_combo, model) pairs from the full
   grid defined in `config.FEATURE_COMBOS` x models returned by
   `model_factory.get_model_names()`.
3. Runs ONLY the missing pairs, ordered fast-first (RF -> XGB -> ANN -> SVR)
   so you get quick wins before the expensive SVR fits.
4. After EACH pair finishes, merges its results into the CSVs on disk so an
   interrupt (Ctrl-C, power loss) loses at most one config, not all of them.
5. Uses the SAME random seed, same normalization, same feature files, and
   same LOPO splitter as `05_train_lopo.py`, so results are directly
   comparable with the existing 6 rows.

Usage
-----
From the `04_pipeline_v2/` directory:

    python scripts/08_complete_missing_lopo.py

Resumable: re-running picks up where it left off.
Dry-run: `python scripts/08_complete_missing_lopo.py --dry-run` prints what
would run without running it.
Force: `python scripts/08_complete_missing_lopo.py --force` re-runs every
configuration, overwriting existing rows.

Dependencies: same as 05_train_lopo.py (numpy, pandas, scikit-learn, xgboost).
"""
import argparse
import os
import signal
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import NUM_CASES, FEATURES_DIR, RESULTS_DIR, FEATURE_COMBOS
from validation.lopo import run_lopo_cv, summarize_lopo
from models.model_factory import get_model_names


RESULTS_CSV = os.path.join(RESULTS_DIR, "lopo_results.csv")
FOLDS_CSV = os.path.join(RESULTS_DIR, "lopo_fold_details.csv")

# Fast-to-slow ordering. RF/XGB are parallelised internally; ANN is serial;
# SVR with RBF on ~26k samples is the slowest.
MODEL_SPEED_ORDER = ["RF", "XGB", "ANN", "SVR"]


def load_case_features(combo_name: str) -> dict:
    """Load per-case feature data for a combo. Mirrors 05_train_lopo.py."""
    case_features = {}
    for case_num in range(1, NUM_CASES + 1):
        csv_path = os.path.join(FEATURES_DIR, f"case{case_num}_{combo_name}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        X = df[feat_cols].values
        y = df["BIS"].values
        valid = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
        if np.sum(valid) > 0:
            case_features[case_num] = (X[valid], y[valid])
    return case_features


def load_existing() -> tuple:
    """Load existing results CSVs if present. Returns (summary_df, folds_df)."""
    summary_df = (
        pd.read_csv(RESULTS_CSV) if os.path.exists(RESULTS_CSV) else pd.DataFrame()
    )
    folds_df = (
        pd.read_csv(FOLDS_CSV) if os.path.exists(FOLDS_CSV) else pd.DataFrame()
    )
    return summary_df, folds_df


def done_pairs(summary_df: pd.DataFrame) -> set:
    """Set of (combo, model) pairs already in the summary CSV."""
    if summary_df.empty or "feature_combo" not in summary_df.columns:
        return set()
    return set(zip(summary_df["feature_combo"], summary_df["model"]))


def missing_pairs(summary_df: pd.DataFrame, force: bool = False) -> list:
    """Return list of (combo, model) pairs to run, ordered fast-first."""
    all_pairs = []
    for combo in FEATURE_COMBOS.keys():
        for model in sorted(get_model_names(), key=lambda m: MODEL_SPEED_ORDER.index(m)
                            if m in MODEL_SPEED_ORDER else 99):
            all_pairs.append((combo, model))

    if force:
        return all_pairs

    already = done_pairs(summary_df)
    return [p for p in all_pairs if p not in already]


def save_incremental(
    summary_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    new_summary_row: dict,
    new_folds: pd.DataFrame,
    force: bool,
) -> tuple:
    """Merge a new config's results into the CSVs on disk.

    If `force` is True and the (combo, model) pair is already present,
    the old rows are replaced.
    """
    combo = new_summary_row["feature_combo"]
    model = new_summary_row["model"]

    # Drop old rows for this (combo, model) if we're forcing a rerun.
    if force and not summary_df.empty:
        summary_df = summary_df[
            ~((summary_df["feature_combo"] == combo) & (summary_df["model"] == model))
        ]
    if force and not folds_df.empty:
        folds_df = folds_df[
            ~((folds_df["feature_combo"] == combo) & (folds_df["model"] == model))
        ]

    summary_df = pd.concat([summary_df, pd.DataFrame([new_summary_row])], ignore_index=True)
    folds_df = pd.concat([folds_df, new_folds], ignore_index=True)

    # Sort rows deterministically so diffs are clean between runs.
    combo_order = list(FEATURE_COMBOS.keys())
    model_order = list(MODEL_SPEED_ORDER)
    summary_df["_combo_idx"] = summary_df["feature_combo"].map(
        {c: i for i, c in enumerate(combo_order)}
    )
    summary_df["_model_idx"] = summary_df["model"].map(
        {m: i for i, m in enumerate(model_order)}
    )
    summary_df = summary_df.sort_values(["_combo_idx", "_model_idx"]).drop(
        columns=["_combo_idx", "_model_idx"]
    )

    folds_df["_combo_idx"] = folds_df["feature_combo"].map(
        {c: i for i, c in enumerate(combo_order)}
    )
    folds_df["_model_idx"] = folds_df["model"].map(
        {m: i for i, m in enumerate(model_order)}
    )
    folds_df = folds_df.sort_values(["_combo_idx", "_model_idx", "fold"]).drop(
        columns=["_combo_idx", "_model_idx"]
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_df.to_csv(RESULTS_CSV, index=False)
    folds_df.to_csv(FOLDS_CSV, index=False)
    return summary_df, folds_df


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run and exit without training.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every (combo, model) pair, overwriting existing rows.",
    )
    args = parser.parse_args()

    # Graceful Ctrl-C: let KeyboardInterrupt propagate but remind the user
    # their partial progress is saved.
    def handle_sigint(_sig, _frame):
        print("\n[interrupt] Received Ctrl-C. Progress is saved to CSVs. Exiting.")
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    print("=" * 64)
    print("STEP 8: Complete missing LOPO configurations")
    print("=" * 64)

    summary_df, folds_df = load_existing()
    already = done_pairs(summary_df)
    to_run = missing_pairs(summary_df, force=args.force)

    print(f"Existing results CSV: {RESULTS_CSV}")
    print(f"  Already done: {len(already)} configs")
    if already:
        for c, m in sorted(already):
            print(f"    done   : {c:15s} x {m}")

    total_grid = len(FEATURE_COMBOS) * len(get_model_names())
    print(f"\nFull grid     : {total_grid} configs ({len(FEATURE_COMBOS)} combos x "
          f"{len(get_model_names())} models)")
    print(f"To run        : {len(to_run)} configs"
          + (" [FORCE MODE: re-running everything]" if args.force else ""))
    for c, m in to_run:
        print(f"    pending: {c:15s} x {m}")

    if args.dry_run:
        print("\n[dry-run] Exiting without training.")
        return 0

    if not to_run:
        print("\nNothing to do — all 28 configs already present. Use --force to rerun.")
        return 0

    total_start = time.time()
    per_config_times = []
    completed = 0

    for combo_name, model_name in to_run:
        completed += 1
        label = f"{combo_name} x {model_name}"
        print(f"\n[{completed}/{len(to_run)}] {label}")

        case_features = load_case_features(combo_name)
        if len(case_features) < 3:
            print(f"  SKIP: only {len(case_features)} cases with features. "
                  f"Did you run 03_extract_features.py?")
            continue

        t0 = time.time()
        try:
            fold_df = run_lopo_cv(case_features, model_name, verbose=False)
        except Exception as exc:
            print(f"  ERROR training {label}: {exc}")
            # Keep going on other configs; don't let one failure stop the batch.
            continue
        elapsed = time.time() - t0
        per_config_times.append(elapsed)

        fold_df["feature_combo"] = combo_name
        fold_df["model"] = model_name
        summary = summarize_lopo(fold_df)
        summary["feature_combo"] = combo_name
        summary["model"] = model_name

        print(
            f"  RMSE={summary['RMSE_mean']:.2f}+/-{summary['RMSE_std']:.2f}  "
            f"MAE={summary['MAE_mean']:.2f}+/-{summary['MAE_std']:.2f}  "
            f"R={summary['Pearson_R_mean']:.3f}  ({format_duration(elapsed)})"
        )

        summary_df, folds_df = save_incremental(
            summary_df, folds_df, summary, fold_df, force=args.force
        )
        print(f"  -> saved to {RESULTS_CSV}")

        remaining = len(to_run) - completed
        if remaining > 0 and per_config_times:
            avg = sum(per_config_times) / len(per_config_times)
            eta = avg * remaining
            print(f"  ETA: {format_duration(eta)} remaining "
                  f"(avg {format_duration(avg)}/config)")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 64}")
    print(f"Done. Total: {format_duration(total_elapsed)} "
          f"across {completed} configs.")

    # Print the best config overall.
    if not summary_df.empty:
        best = summary_df.loc[summary_df["RMSE_mean"].idxmin()]
        print(f"\nBest LOPO: {best['feature_combo']} + {best['model']}")
        print(f"  RMSE = {best['RMSE_mean']:.2f} +/- {best['RMSE_std']:.2f}")
        print(f"  MAE  = {best['MAE_mean']:.2f} +/- {best['MAE_std']:.2f}")
        print(f"  R    = {best['Pearson_R_mean']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
