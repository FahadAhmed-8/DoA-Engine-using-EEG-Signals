"""
Multi-channel analysis: Compute per-channel entropy-BIS correlations.
"""
import numpy as np
import pandas as pd
from scipy import stats

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import NUM_CASES


def compute_channel_correlations(entropy_dict: dict, bis: np.ndarray) -> pd.DataFrame:
    """
    Compute Pearson correlation between each channel's entropy and BIS.

    Args:
        entropy_dict: {'SampEn': (n_windows, n_channels), ...}
        bis: (n_windows,)

    Returns:
        DataFrame [channel, entropy_type, correlation, p_value]
    """
    rows = []

    for entropy_name, values in entropy_dict.items():
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        n_channels = values.shape[1]

        for ch in range(n_channels):
            ch_vals = values[:, ch]

            # Remove NaN
            valid = ~(np.isnan(ch_vals) | np.isnan(bis))
            if np.sum(valid) < 5:
                continue

            r, p = stats.pearsonr(ch_vals[valid], bis[valid])

            rows.append({
                'channel': ch,
                'entropy_type': entropy_name,
                'correlation': r,
                'abs_correlation': abs(r),
                'p_value': p,
            })

    return pd.DataFrame(rows)


def aggregate_channel_correlations(all_case_corrs: list) -> pd.DataFrame:
    """
    Average correlations across all patients.

    Args:
        all_case_corrs: list of DataFrames from compute_channel_correlations

    Returns:
        DataFrame with mean correlations per channel × entropy type
    """
    combined = pd.concat(all_case_corrs, ignore_index=True)

    agg = combined.groupby(['channel', 'entropy_type']).agg(
        mean_corr=('correlation', 'mean'),
        std_corr=('correlation', 'std'),
        mean_abs_corr=('abs_correlation', 'mean'),
        n_cases=('correlation', 'count'),
    ).reset_index()

    agg = agg.sort_values('mean_abs_corr', ascending=False)
    return agg


def rank_channels(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank channels by overall entropy-BIS correlation strength.
    """
    channel_rank = agg_df.groupby('channel').agg(
        avg_abs_corr=('mean_abs_corr', 'mean')
    ).reset_index()

    channel_rank = channel_rank.sort_values('avg_abs_corr', ascending=False)
    channel_rank['rank'] = range(1, len(channel_rank) + 1)
    return channel_rank
