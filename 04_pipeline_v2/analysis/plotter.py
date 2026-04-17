"""
Visualization: Generate publication-quality plots.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PLOTS_DIR, TARGET_RMSE


def setup_style():
    """Set consistent plot style."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })


def plot_prediction_scatter(y_true, y_pred, title, output_path):
    """Scatter plot: actual vs predicted BIS."""
    setup_style()
    fig, ax = plt.subplots()

    ax.scatter(y_true, y_pred, alpha=0.3, s=10, c='steelblue')
    ax.plot([0, 100], [0, 100], 'r--', linewidth=1.5, label='Ideal')
    ax.set_xlabel('Actual BIS')
    ax.set_ylabel('Predicted BIS')
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_model_comparison(results_df, metric, output_path):
    """Bar chart comparing models on a given metric."""
    setup_style()
    fig, ax = plt.subplots()

    combos = results_df['feature_combo'].unique()
    models = results_df['model'].unique()
    x = np.arange(len(combos))
    width = 0.18

    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        values = [model_data[model_data['feature_combo'] == c][metric].values[0]
                  if len(model_data[model_data['feature_combo'] == c]) > 0 else 0
                  for c in combos]
        ax.bar(x + i * width, values, width, label=model)

    if metric == 'RMSE':
        ax.axhline(y=TARGET_RMSE, color='red', linestyle='--', alpha=0.7,
                    label=f'Baseline Target ({TARGET_RMSE})')

    ax.set_xlabel('Feature Combination')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison: Models × Feature Combos')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(combos, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_lopo_boxplot(lopo_fold_df, output_path):
    """Boxplot of RMSE across LOPO folds for each model."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    models = lopo_fold_df['model'].unique()
    data_by_model = [lopo_fold_df[lopo_fold_df['model'] == m]['RMSE'].values for m in models]

    bp = ax.boxplot(data_by_model, labels=models, patch_artist=True)
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    for patch, color in zip(bp['boxes'], colors[:len(models)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=TARGET_RMSE, color='red', linestyle='--', alpha=0.7,
                label=f'Baseline Target ({TARGET_RMSE})')
    ax.set_ylabel('RMSE')
    ax.set_title('LOPO Cross-Validation: RMSE Distribution by Model')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_channel_heatmap(agg_df, output_path):
    """Heatmap: channels × entropy types, values = mean correlation."""
    setup_style()

    pivot = agg_df.pivot_table(index='channel', columns='entropy_type',
                                values='mean_corr')

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'Ch {i}' for i in pivot.index])
    ax.set_title('Channel × Entropy Correlation with BIS')

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color='white' if abs(val) > 0.5 else 'black', fontsize=9)

    plt.colorbar(im, ax=ax, label='Pearson R')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_combo_summary(results_df, output_path):
    """Bar chart: best RMSE per feature combo (across all models)."""
    setup_style()
    fig, ax = plt.subplots()

    best_per_combo = results_df.groupby('feature_combo')['RMSE'].min().sort_values()

    bars = ax.barh(range(len(best_per_combo)), best_per_combo.values, color='steelblue')
    ax.set_yticks(range(len(best_per_combo)))
    ax.set_yticklabels(best_per_combo.index)
    ax.set_xlabel('Best RMSE')
    ax.set_title('Best RMSE per Feature Combination')
    ax.axvline(x=TARGET_RMSE, color='red', linestyle='--', alpha=0.7,
                label=f'Baseline ({TARGET_RMSE})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
