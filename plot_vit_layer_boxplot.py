"""
Box plot visualization of distance metrics across 12 ViT layers.

Creates a 1x3 subplot figure showing the distribution of:
  - KL Divergence
  - Kolmogorov Distance
  - Wasserstein Distance
for each of the 12 transformer layers.

Input CSV: results/eigenvalue_uniformity_results.csv
Output: results/vit_layer_distance_boxplot.png / .svg
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Human-readable labels
METRIC_LABELS = {
    'kl_divergence': 'KL Divergence',
    'kolmogorov_distance': 'Kolmogorov Distance',
    'wasserstein_distance': 'Wasserstein Distance',
}

DIST_METRICS = ['kl_divergence', 'kolmogorov_distance', 'wasserstein_distance']


def load_data(csv_path):
    """Load CSV and verify expected columns."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Check required columns
    for col in ['layer'] + DIST_METRICS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Report layer counts
    layer_counts = df['layer'].value_counts().sort_index()
    print(f"Samples per layer:\n{layer_counts}")

    return df


def plot_boxplots(df, results_dir):
    """Create 1x3 box plot figure: one subplot per distance metric."""
    n_metrics = len(DIST_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(14, 5), constrained_layout=True)

    # Ensure axes is iterable even for single subplot
    if n_metrics == 1:
        axes = [axes]

    # Get unique layer indices sorted
    layers_sorted = sorted(df['layer'].unique())
    n_layers = len(layers_sorted)
    print(f"Plotting across {n_layers} layers: {layers_sorted}")

    for idx, metric in enumerate(DIST_METRICS):
        ax = axes[idx]

        # Box plot: x=layer, y=metric value
        sns.boxplot(
            data=df,
            x='layer',
            y=metric,
            ax=ax,
            palette='Set2',
            whis=1.5,  # standard whisker length
            showfliers=True,  # show outliers
        )

        ax.set_xlabel('Layer')
        ax.set_ylabel("")
        ax.set_yscale('log')
        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Rotate x-tick labels if many layers
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # fig.suptitle('Distance Metrics Distribution Across ViT Layers', fontsize=14, y=1.02)

    # Save PNG
    save_path_png = Path(results_dir) / 'vit_layer_distance_boxplot.png'
    plt.savefig(save_path_png, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path_png}")

    # Save SVG
    save_path_svg = Path(results_dir) / 'vit_layer_distance_boxplot.svg'
    plt.savefig(save_path_svg, bbox_inches='tight')
    print(f"Saved: {save_path_svg}")

    plt.close(fig)


def main():
    csv_path = Path('results') / 'eigenvalue_uniformity_results.csv'
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Plot ViT Layer Distance Metrics — Box Plot")
    print("=" * 60)

    df = load_data(csv_path)

    print(f"\nLayers: {sorted(df['layer'].unique())}")
    print(f"Total samples: {len(df)}")
    for metric in DIST_METRICS:
        mean_val = df[metric].mean()
        print(f"  {METRIC_LABELS[metric]:20s}: overall mean = {mean_val:.6f}")

    plot_boxplots(df, results_dir)

    print("\n" + "=" * 60)
    print("Done! Generated:")
    print("  - results/vit_layer_distance_boxplot.png")
    print("  - results/vit_layer_distance_boxplot.svg")
    print("=" * 60)


if __name__ == '__main__':
    main()
