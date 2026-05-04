"""
Plot eigenvalue angle statistics across sequence lengths.

X-axis: seq_len
Y-axis: metric statistics (mean ± quantile interval)
Subfigures: one per metric, grouped by semantic category.
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Metric groups for subplot organization
SEPARATION_METRICS = [
    'kl_divergence',
    'tv_distance',
    'kolmogorov_distance',
    'wasserstein_distance',
]
RAYLEIGH_METRICS = [
    'rayleigh_R',
    'rayleigh_stat',
    'rayleigh_p_value',
]
CIRCULAR_METRICS = [
    'circular_variance',
    'circular_entropy',
]
ALL_METRICS = SEPARATION_METRICS + RAYLEIGH_METRICS + CIRCULAR_METRICS

# Human-readable labels
METRIC_LABELS = {
    'kl_divergence': 'KL Divergence',
    'tv_distance': 'TV Distance',
    'kolmogorov_distance': 'Kolmogorov Distance',
    'wasserstein_distance': 'Wasserstein Distance',
    'rayleigh_R': 'Rayleigh R',
    'rayleigh_stat': 'Rayleigh Statistic',
    'rayleigh_p_value': 'Rayleigh p-value',
    'circular_variance': 'Circular Variance',
    'circular_entropy': 'Circular Entropy',
}


def load_and_aggregate(csv_path):
    """Load CSV and compute per-seq_len statistics for each metric."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Unique seq_len values: {df['seq_len'].nunique()}")
    print(f"Layers per seq_len: {df.groupby('seq_len')['layer_idx'].nunique().iloc[0]}")

    # Group by seq_len and compute statistics
    aggregated = {}
    seq_lens_sorted = None

    for metric in ALL_METRICS:
        grouped = df.groupby('seq_len')[metric]
        stats_df = grouped.agg(['mean', 'median', 'count', 'min', 'max',
                                lambda x: np.percentile(x, 10),
                                lambda x: np.percentile(x, 25),
                                lambda x: np.percentile(x, 75),
                                lambda x: np.percentile(x, 90)])
        stats_df = stats_df.rename(columns={
            '<lambda_0>': 'q10',
            '<lambda_1>': 'q25',
            '<lambda_2>': 'q75',
            '<lambda_3>': 'q90',
        })
        stats_df = stats_df.sort_index()
        aggregated[metric] = stats_df
        if seq_lens_sorted is None:
            seq_lens_sorted = stats_df.index.values

    return seq_lens_sorted, aggregated


def plot_metrics(seq_lens, aggregated, results_dir):
    """Create multi-panel figure: one subplot per metric (linear scale)."""
    n_metrics = len(ALL_METRICS)
    nrows, ncols = 3, 4  # 12 slots for 9 metrics

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()

    for idx, metric in enumerate(ALL_METRICS):
        ax = axes[idx]
        stats_df = aggregated[metric]

        x = seq_lens
        y_mean = stats_df['mean'].values
        y_q25 = stats_df['q25'].values
        y_q75 = stats_df['q75'].values
        y_q10 = stats_df['q10'].values
        y_q90 = stats_df['q90'].values

        # Plot mean line
        ax.plot(x, y_mean, color='steelblue', linewidth=2, label='Mean')

        # Shaded inter-quartile region (25–75%)
        ax.fill_between(x, y_q25, y_q75, color='steelblue', alpha=0.3, label='IQR (25–75%)')

        # Optional: lighter shading for 10–90% envelope
        ax.fill_between(x, y_q10, y_q90, color='steelblue', alpha=0.1, label='10–90% range')

        ax.set_xlabel('Sequence Length')
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.grid(True, alpha=0.3)

        # Only show legend in first subplot to avoid clutter
        if idx == 0:
            ax.legend(loc='best', fontsize=8)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Eigenvalue Angle Uniformity Metrics vs Sequence Length (GPT-2)',
                 fontsize=14, y=1.02)

    # Save PNG
    save_path_png = Path(results_dir) / 'eigen_angle_statistics.png'
    plt.savefig(save_path_png, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path_png}")

    # Save SVG
    save_path_svg = Path(results_dir) / 'eigen_angle_statistics.svg'
    plt.savefig(save_path_svg, bbox_inches='tight')
    print(f"Saved: {save_path_svg}")

    plt.close(fig)


def plot_distance_metrics_linear_log(seq_lens, aggregated, results_dir):
    """Create a 2×3 figure: KL, Kolmogorov, Wasserstein — linear and log scales.

    Row 1 (linear): raw metric values
    Row 2 (log): log10(metric) — these are distance metrics (near 0 when uniform)
    Output: distance_metrics_three.png / .svg
    """
    DIST_METRICS = ['kl_divergence', 'kolmogorov_distance', 'wasserstein_distance']
    n_metrics = len(DIST_METRICS)
    nrows, ncols = 2, 3
    eps = 1e-6

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6), constrained_layout=True)
    axes = axes.flatten()

    for idx, metric in enumerate(DIST_METRICS):
        stats_df = aggregated[metric]
        x = seq_lens
        y_mean = stats_df['mean'].values
        y_q25 = stats_df['q25'].values
        y_q75 = stats_df['q75'].values
        y_q10 = stats_df['q10'].values
        y_q90 = stats_df['q90'].values

        # Row 1: linear scale
        ax_lin = axes[idx]
        ax_lin.plot(x, y_mean, color='steelblue', linewidth=2, label='Mean')
        ax_lin.fill_between(x, y_q25, y_q75, color='steelblue', alpha=0.3, label='IQR (25–75%)')
        ax_lin.fill_between(x, y_q10, y_q90, color='steelblue', alpha=0.1, label='10–90%')
        ax_lin.set_xlabel('Sequence Length')
        # ax_lin.set_ylabel(METRIC_LABELS[metric])
        ax_lin.set_title(METRIC_LABELS[metric], fontsize=10)
        ax_lin.grid(True, alpha=0.3)
        if idx == 0:
            ax_lin.legend(loc='best', fontsize=8)

        # Row 2: log scale
        ax_log = axes[idx + ncols]
        # y_mean_t = np.log10(y_mean + eps)
        # y_q25_t = np.log10(y_q25 + eps)
        # y_q75_t = np.log10(y_q75 + eps)
        # y_q10_t = np.log10(y_q10 + eps)
        # y_q90_t = np.log10(y_q90 + eps)

        # ax_log.plot(x, y_mean_t, color='crimson', linewidth=2)
        # ax_log.fill_between(x, y_q25_t, y_q75_t, color='crimson', alpha=0.3)
        # ax_log.fill_between(x, y_q10_t, y_q90_t, color='crimson', alpha=0.1)
        # ax_log.set_xlabel('Sequence Length')
        # ax_log.set_ylabel(f"log10({METRIC_LABELS[metric]})")
        # ax_log.set_title(f"{METRIC_LABELS[metric]} (log)", fontsize=10)
        # ax_log.grid(True, alpha=0.3)
        ax_log.plot(x, y_mean, color='crimson', linewidth=2, label='Mean')
        ax_log.fill_between(x, y_q25, y_q75, color='crimson', alpha=0.3, label='IQR')
        ax_log.fill_between(x, y_q10, y_q90, color='crimson', alpha=0.1, label='10–90%')
        ax_log.set_xlabel('Sequence Length')
        # ax_log.set_ylabel(METRIC_LABELS[metric])
        ax_log.set_yscale('log')
        ax_log.set_title(f"{METRIC_LABELS[metric]} (log-scale)", fontsize=10)
        ax_log.grid(True, alpha=0.3)

    # Save PNG
    save_path_png = Path(results_dir) / 'distance_metrics_three.png'
    plt.savefig(save_path_png, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path_png}")

    # Save SVG
    save_path_svg = Path(results_dir) / 'distance_metrics_three.svg'
    plt.savefig(save_path_svg, bbox_inches='tight')
    print(f"Saved: {save_path_svg}")

    plt.close(fig)


def plot_metrics_log_scale(seq_lens, aggregated, results_dir):
    """Create multi-panel figure: one subplot per metric on log scale.

    Distance metrics (kl, tv, kolmogorov, wasserstein, rayleigh_R, rayleigh_stat):
        y-axis = log10(metric + eps) — these are near 0 when uniform
    Similarity metrics (rayleigh_p_value, circular_variance, circular_entropy):
        y-axis = log10(1 − y + eps) — these cluster near 1 when uniform
    """
    n_metrics = len(ALL_METRICS)
    nrows, ncols = 3, 4
    eps = 1e-6  # small constant to avoid log(0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()

    # Metrics that need (1-y) transformation: these cluster near 1 when uniform
    SIMILARITY_METRICS = ['rayleigh_p_value', 'circular_variance', 'circular_entropy']

    for idx, metric in enumerate(ALL_METRICS):
        ax = axes[idx]
        stats_df = aggregated[metric]

        x = seq_lens
        y_mean = stats_df['mean'].values
        y_q25 = stats_df['q25'].values
        y_q75 = stats_df['q75'].values
        y_q10 = stats_df['q10'].values
        y_q90 = stats_df['q90'].values

        # Transform for log scale
        if metric in SIMILARITY_METRICS:
            # Similarity metrics: log(1-y) to show deviation from 1
            y_mean_t = np.log10(1 - y_mean + eps)
            y_q25_t = np.log10(1 - y_q25 + eps)
            y_q75_t = np.log10(1 - y_q75 + eps)
            y_q10_t = np.log10(1 - y_q10 + eps)
            y_q90_t = np.log10(1 - y_q90 + eps)
            ylabel = f"log10(1 - {METRIC_LABELS[metric]})"
            title_suffix = " (1−y) log scale"
        else:
            # Distance metrics: log(y)
            y_mean_t = np.log10(y_mean + eps)
            y_q25_t = np.log10(y_q25 + eps)
            y_q75_t = np.log10(y_q75 + eps)
            y_q10_t = np.log10(y_q10 + eps)
            y_q90_t = np.log10(y_q90 + eps)
            ylabel = f"log10({METRIC_LABELS[metric]})"
            title_suffix = " log scale"

        # Plot mean line
        ax.plot(x, y_mean_t, color='crimson', linewidth=2, label='Mean')

        # Shaded inter-quartile region
        ax.fill_between(x, y_q25_t, y_q75_t, color='crimson', alpha=0.3, label='IQR (25–75%)')

        # Lighter envelope for 10–90%
        ax.fill_between(x, y_q10_t, y_q90_t, color='crimson', alpha=0.1, label='10–90% range')

        ax.set_xlabel('Sequence Length')
        ax.set_ylabel(ylabel)
        ax.set_title(METRIC_LABELS[metric] + title_suffix, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Only show legend in first subplot
        if idx == 0:
            ax.legend(loc='best', fontsize=8)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Eigenvalue Angle Uniformity Metrics vs Sequence Length — Log Scale (GPT-2)',
                 fontsize=14, y=1.02)

    # Save PNG
    save_path_png = Path(results_dir) / 'eigen_angle_statistics_log.png'
    plt.savefig(save_path_png, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path_png}")

    # Save SVG
    save_path_svg = Path(results_dir) / 'eigen_angle_statistics_log.svg'
    plt.savefig(save_path_svg, bbox_inches='tight')
    print(f"Saved: {save_path_svg}")

    plt.close(fig)


def main():
    csv_path = Path('results') / 'eigen_angle_statistics.csv'
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    print("="*60)
    print("Plot Eigen Angle Statistics — seq_len vs metrics")
    print("="*60)

    seq_lens, aggregated = load_and_aggregate(csv_path)

    print(f"\nSequence length range: {seq_lens.min()} – {seq_lens.max()}")
    print(f"\nMetrics to plot: {len(ALL_METRICS)}")
    for m in ALL_METRICS:
        mean_val = aggregated[m]['mean'].mean()
        print(f"  {METRIC_LABELS[m]:25s}: mean={mean_val:.4f}")

    plot_metrics(seq_lens, aggregated, results_dir)
    plot_metrics_log_scale(seq_lens, aggregated, results_dir)
    plot_distance_metrics_linear_log(seq_lens, aggregated, results_dir)

    print("\n" + "="*60)
    print("Done! Generated:")
    print("  - results/eigen_angle_statistics.png/.svg      (all 9 metrics, linear)")
    print("  - results/eigen_angle_statistics_log.png/.svg  (all 9 metrics, log scale)")
    print("  - results/distance_metrics_three.png/.svg      (KL, Kolmogorov, Wasserstein)")
    print("="*60)


if __name__ == "__main__":
    main()
