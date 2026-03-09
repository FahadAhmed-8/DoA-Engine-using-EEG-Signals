#!/usr/bin/env python3
"""
Step 6: Multi-channel entropy analysis.
Compute per-channel entropy-BIS correlations, rank channels.
"""
import sys, os
import pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import NUM_CASES, FEATURES_DIR, RESULTS_DIR, PLOTS_DIR
from analysis.channel_analysis import (
    compute_channel_correlations, aggregate_channel_correlations, rank_channels
)
from analysis.plotter import plot_channel_heatmap


def main():
    print("=" * 60)
    print("STEP 6: Multi-Channel Analysis")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_corrs = []

    for case_num in range(1, NUM_CASES + 1):
        raw_path = os.path.join(FEATURES_DIR, f"case{case_num}_entropy_raw.pkl")
        if not os.path.exists(raw_path):
            print(f"  Case {case_num}: no raw entropy data, skipping")
            continue

        data = pickle.load(open(raw_path, 'rb'))
        entropy_dict = data['entropy']
        bis = data['bis']

        print(f"  Case {case_num}: {data['n_channels']} channels, {len(bis)} windows")

        corr_df = compute_channel_correlations(entropy_dict, bis)
        corr_df['case'] = case_num
        all_corrs.append(corr_df)

    if not all_corrs:
        print("No channel data available!")
        return

    # Aggregate across patients
    agg_df = aggregate_channel_correlations(all_corrs)
    agg_path = os.path.join(RESULTS_DIR, "channel_correlations.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"\nChannel correlations saved to: {agg_path}")

    # Rank channels
    rank_df = rank_channels(agg_df)
    rank_path = os.path.join(RESULTS_DIR, "channel_rankings.csv")
    rank_df.to_csv(rank_path, index=False)
    print(f"Channel rankings saved to: {rank_path}")
    print("\nChannel Rankings:")
    print(rank_df.to_string(index=False))

    # Plot heatmap
    heatmap_path = os.path.join(PLOTS_DIR, "channel_heatmap.png")
    try:
        plot_channel_heatmap(agg_df, heatmap_path)
        print(f"\nHeatmap saved to: {heatmap_path}")
    except Exception as e:
        print(f"Heatmap plotting failed: {e}")

    print("\nChannel analysis complete!")


if __name__ == '__main__':
    main()
