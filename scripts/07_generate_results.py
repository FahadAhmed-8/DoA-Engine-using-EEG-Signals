#!/usr/bin/env python3
"""
Step 7: Aggregate all results, generate summary tables and plots.
"""
import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import RESULTS_DIR, PLOTS_DIR, TARGET_RMSE, TARGET_MAE
from analysis.plotter import (
    plot_model_comparison, plot_feature_combo_summary, plot_lopo_boxplot
)


def main():
    print("=" * 60)
    print("STEP 7: Results Aggregation & Visualization")
    print("=" * 60)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Load baseline results ──
    baseline_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        print(f"\nBaseline Results ({len(baseline_df)} experiments):")
        print(baseline_df[['feature_combo', 'model', 'RMSE', 'MAE', 'Pearson_R']].to_string(index=False))

        # Plot baseline model comparison
        try:
            plot_model_comparison(baseline_df, 'RMSE',
                                  os.path.join(PLOTS_DIR, "baseline_model_comparison_RMSE.png"))
            plot_model_comparison(baseline_df, 'MAE',
                                  os.path.join(PLOTS_DIR, "baseline_model_comparison_MAE.png"))
            plot_feature_combo_summary(baseline_df,
                                       os.path.join(PLOTS_DIR, "baseline_feature_combo_best.png"))
            print("\nBaseline plots saved.")
        except Exception as e:
            print(f"Baseline plot error: {e}")
    else:
        baseline_df = None
        print("No baseline results found.")

    # ── Load LOPO results ──
    lopo_path = os.path.join(RESULTS_DIR, "lopo_results.csv")
    lopo_folds_path = os.path.join(RESULTS_DIR, "lopo_fold_details.csv")

    if os.path.exists(lopo_path):
        lopo_df = pd.read_csv(lopo_path)
        print(f"\nLOPO Results ({len(lopo_df)} experiments):")
        cols = ['feature_combo', 'model', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std',
                'Pearson_R_mean', 'Pearson_R_std']
        available_cols = [c for c in cols if c in lopo_df.columns]
        print(lopo_df[available_cols].to_string(index=False))

        # Plot LOPO boxplot
        if os.path.exists(lopo_folds_path):
            try:
                folds_df = pd.read_csv(lopo_folds_path)
                plot_lopo_boxplot(folds_df,
                                  os.path.join(PLOTS_DIR, "lopo_rmse_boxplot.png"))
                print("LOPO boxplot saved.")
            except Exception as e:
                print(f"LOPO boxplot error: {e}")
    else:
        lopo_df = None
        print("No LOPO results found.")

    # ── Summary Report ──
    report_lines = [
        "=" * 60,
        "DoA EEG Pipeline — Summary Report",
        "=" * 60,
        f"\nTarget Baseline: RMSE = {TARGET_RMSE}, MAE = {TARGET_MAE}",
        ""
    ]

    if baseline_df is not None and len(baseline_df) > 0:
        best_baseline = baseline_df.loc[baseline_df['RMSE'].idxmin()]
        report_lines.append("--- BASELINE (80/20 Split) ---")
        report_lines.append(f"Best: {best_baseline['feature_combo']} + {best_baseline['model']}")
        report_lines.append(f"  RMSE = {best_baseline['RMSE']:.2f}")
        report_lines.append(f"  MAE  = {best_baseline['MAE']:.2f}")
        report_lines.append(f"  R    = {best_baseline['Pearson_R']:.3f}")
        beat = "YES" if best_baseline['RMSE'] < TARGET_RMSE else "NO"
        report_lines.append(f"  Beats target: {beat}")
        report_lines.append("")

    if lopo_df is not None and len(lopo_df) > 0:
        best_lopo = lopo_df.loc[lopo_df['RMSE_mean'].idxmin()]
        report_lines.append("--- LOPO CROSS-VALIDATION ---")
        report_lines.append(f"Best: {best_lopo['feature_combo']} + {best_lopo['model']}")
        report_lines.append(f"  RMSE = {best_lopo['RMSE_mean']:.2f} ± {best_lopo['RMSE_std']:.2f}")
        report_lines.append(f"  MAE  = {best_lopo['MAE_mean']:.2f} ± {best_lopo['MAE_std']:.2f}")
        report_lines.append(f"  R    = {best_lopo['Pearson_R_mean']:.3f} ± {best_lopo['Pearson_R_std']:.3f}")
        report_lines.append("")

    # Channel analysis
    ch_path = os.path.join(RESULTS_DIR, "channel_rankings.csv")
    if os.path.exists(ch_path):
        ch_df = pd.read_csv(ch_path)
        report_lines.append("--- CHANNEL RANKINGS ---")
        for _, row in ch_df.iterrows():
            report_lines.append(f"  Rank {int(row['rank'])}: Channel {int(row['channel'])} "
                                f"(avg |r| = {row['avg_abs_corr']:.3f})")
        report_lines.append("")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(RESULTS_DIR, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\n{report_text}")
    print(f"\nReport saved to: {report_path}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print("\nResults generation complete!")


if __name__ == '__main__':
    main()
