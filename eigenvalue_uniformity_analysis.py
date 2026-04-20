import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms
from scipy import stats
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Distribution Distance/Divergence Measures for Circular Uniformity
# ============================================================

def kl_divergence_uniform(angles, n_bins=50):
    """
    Compute KL divergence between observed angle distribution and uniform distribution.
    Uses scipy.stats.entropy for computation.
    Angles should be in range [-π, π)
    """
    # Convert to [0, 2π) for binning
    angles_0_2pi = angles + np.pi
    bin_edges = np.linspace(0, 2*np.pi, n_bins+1)

    # Observed distribution
    observed, _ = np.histogram(angles_0_2pi, bins=bin_edges, density=False)

    # Normalize to probabilities
    observed_prob = observed / observed.sum()

    # Uniform distribution
    uniform_prob = np.full(n_bins, 1.0 / n_bins)

    # KL divergence using scipy.stats.entropy: entropy(p, q) = sum(p * log(p/q))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    kl_div = stats.entropy(observed_prob + epsilon, uniform_prob + epsilon)

    return kl_div

def total_variation_distance(angles, n_bins=50):
    """
    Compute Total Variation distance between observed and uniform distribution.
    TV = 0.5 * sum(|p_i - q_i|)
    """
    angles_0_2pi = angles + np.pi
    bin_edges = np.linspace(0, 2*np.pi, n_bins+1)

    observed, _ = np.histogram(angles_0_2pi, bins=bin_edges, density=False)

    # Normalize to probabilities
    observed_prob = observed / observed.sum()
    uniform_prob = 1.0 / n_bins

    # TV distance
    tv_distance = 0.5 * np.sum(np.abs(observed_prob - uniform_prob))

    return tv_distance

def kolmogorov_distance(angles):
    """
    Compute Kolmogorov-Smirnov distance for circular uniformity.
    Uses scipy.stats.kstest for computation.
    """
    # Convert to [0, 2π)
    angles_0_2pi = angles + np.pi

    # Normalize to [0, 1) for uniform distribution test
    angles_normalized = angles_0_2pi / (2*np.pi)

    # KS test against uniform distribution on [0, 1]
    ks_stat, p_value = stats.kstest(angles_normalized, 'uniform')

    return ks_stat

def wasserstein_distance_uniform(angles):
    """
    Compute Wasserstein (Earth Mover's) distance between observed angles and uniform.
    """
    angles_0_2pi = angles + np.pi

    # Sample from uniform distribution [0, 2π)
    uniform_samples = np.random.uniform(0, 2*np.pi, size=len(angles_0_2pi))

    # Wasserstein distance
    w_dist = wasserstein_distance(angles_0_2pi, uniform_samples)

    return w_dist

def rayleigh_test(angles):
    """
    Rayleigh test for circular uniformity using scipy.stats.rayleigh_test.
    Returns test statistic and p-value.
    Small p-value indicates non-uniformity.
    """
    # Use scipy's rayleigh test (available in scipy >= 1.8.0)
    try:
        result = stats.rayleigh_test(angles)
        R = result.statistic
        p_value = result.pvalue
        z = result.statistic * len(angles)  # Rayleigh Z statistic
    except AttributeError:
        # Fallback for older scipy versions
        n = len(angles)
        R = np.abs(np.mean(np.exp(1j * angles)))
        z = n * R**2
        p_value = np.exp(-z)

    return R, z, p_value

def circular_variance(angles):
    """
    Compute circular variance using scipy.stats.circvar.
    0 = all angles concentrated, 1 = uniform distribution.
    """
    # scipy.stats.circvar returns variance in [0, 1] for circular data
    circ_var = stats.circvar(angles)
    return circ_var

def circular_entropy(angles, n_bins=50):
    """
    Compute circular entropy using scipy.stats.entropy.
    Lower = more concentrated, higher = more uniform.
    Normalized by max entropy so that 1 = uniform distribution.
    """
    angles_0_2pi = angles + np.pi
    bin_edges = np.linspace(0, 2*np.pi, n_bins+1)

    observed, _ = np.histogram(angles_0_2pi, bins=bin_edges, density=False)

    # Normalize to probabilities
    observed_prob = observed / observed.sum()

    # Compute entropy using scipy.stats.entropy
    entropy = stats.entropy(observed_prob)

    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log(n_bins)
    normalized_entropy = entropy / max_entropy

    return normalized_entropy

# ============================================================
# Process Multiple Samples
# ============================================================

def process_samples(num_samples=None, max_layers=12):
    """
    Process multiple Imagenette samples and compute statistics for each layer.

    Parameters:
    -----------
    num_samples : int or None
        Number of samples to process. If None, processes all samples in the dataset.
    max_layers : int
        Maximum number of layers to analyze per sample.
    """
    # Load data and model
    print("Loading Imagenette dataset and ViT model...")

    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imagenette_data = datasets.Imagenette(root='./data', split='val', transform=transform)

    # If num_samples is None, process all samples
    if num_samples is None:
        num_samples = len(imagenette_data)
        print(f"Processing all {num_samples} samples in the dataset...")
    else:
        print(f"Processing {num_samples} samples...")

    model = torchvision.models.vit_b_16(weights='IMAGENET1K_SWAG_E2E_V1')
    model.eval()

    results = []

    for sample_idx in tqdm(range(num_samples), desc="Processing samples", unit="sample"):
        # Get sample
        sample_image, sample_label = imagenette_data[sample_idx]
        input_tensor = sample_image.unsqueeze(0)

        # Capture layer outputs
        layer_outputs = []

        def hook_fn(module, input, output):
            layer_outputs.append(output.detach().cpu())

        hooks = []
        for encoder_layer in model.encoder.layers:
            hook = encoder_layer.ln_1.register_forward_hook(hook_fn)
            hooks.append(hook)

        with torch.no_grad():
            _ = model(input_tensor)

        for hook in hooks:
            hook.remove()

        # Process each layer
        for layer_idx, X_tensor in enumerate(layer_outputs[:max_layers]):
            X = X_tensor.squeeze(0).numpy()

            # SVD
            U, S, Vh = np.linalg.svd(X, full_matrices=True)

            # Eigenvalues of U
            eigvals = np.linalg.eigvals(U)
            angles = np.angle(eigvals)

            # Compute all distance measures
            kl_div = kl_divergence_uniform(angles, n_bins=50)
            tv_dist = total_variation_distance(angles, n_bins=50)
            ks_dist = kolmogorov_distance(angles)
            w_dist = wasserstein_distance_uniform(angles)
            R, rayleigh_z, rayleigh_p = rayleigh_test(angles)
            circ_var = circular_variance(angles)
            circ_entropy = circular_entropy(angles, n_bins=50)

            results.append({
                'sample_idx': sample_idx,
                'layer': layer_idx,
                'kl_divergence': kl_div,
                'tv_distance': tv_dist,
                'kolmogorov_distance': ks_dist,
                'wasserstein_distance': w_dist,
                'rayleigh_R': R,
                'rayleigh_stat': rayleigh_z,
                'rayleigh_p_value': rayleigh_p,
                'circular_variance': circ_var,
                'circular_entropy': circ_entropy,
                'num_eigenvalues': len(angles)
            })

    return pd.DataFrame(results)

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze eigenvalue uniformity in Vision Transformer attention layers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to process. If not provided, processes all samples in the dataset.'
    )
    parser.add_argument(
        '--max_layers',
        type=int,
        default=12,
        help='Maximum number of layers to analyze per sample.'
    )
    # parser.add_argument(
    #     '--log_dir',
    #     type=str,
    #     default='logs',
    #     help='Directory to save log files.'
    # )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory to save result files.'
    )

    args = parser.parse_args()

    # Create directories using pathlib
    # log_dir = Path(args.log_dir)
    results_dir = Path(args.results_dir)
    # log_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print("="*80)
    print("EIGENVALUE UNIFORMITY ANALYSIS FOR VISION TRANSFORMER")
    print("="*80)
    print("\nConfiguration:")
    print(f"  - Number of samples: {args.num_samples if args.num_samples else 'ALL'}")
    print(f"  - Max layers: {args.max_layers}")
    # print(f"  - Log directory: {log_dir}")
    print(f"  - Results directory: {results_dir}")
    print("\nThis script analyzes the distribution of eigenvalues from the")
    print("left singular matrix (U) of attention input X across Imagenette samples.\n")

    # Run analysis
    df_results = process_samples(num_samples=args.num_samples, max_layers=args.max_layers)

    # ============================================================
    # Create Statistics Table
    # ============================================================

    stats_table = df_results.groupby('layer').agg({
        'kl_divergence': ['mean', 'std', 'median'],
        'tv_distance': ['mean', 'std', 'median'],
        'kolmogorov_distance': ['mean', 'std', 'median'],
        'wasserstein_distance': ['mean', 'std', 'median'],
        'rayleigh_R': ['mean', 'std', 'median'],
        'rayleigh_p_value': ['mean', 'std'],
        'circular_variance': ['mean', 'std', 'median'],
        'circular_entropy': ['mean', 'std', 'median']
    }).round(4)

    print("\n" + "="*80)
    print("AGGREGATED STATISTICS BY LAYER")
    print("="*80)
    print("\n(mean, std, median across 50 samples):\n")
    print(stats_table.to_string())

    # ============================================================
    # Summary Table (simplified)
    # ============================================================

    summary_table = df_results.groupby('layer').agg({
        'kl_divergence': 'mean',
        'tv_distance': 'mean',
        'kolmogorov_distance': 'mean',
        'wasserstein_distance': 'mean',
        'rayleigh_R': 'mean',
        'circular_variance': 'mean',
        'circular_entropy': 'mean'
    }).round(4)

    summary_table.columns = [
        'KL Div',
        'TV Dist',
        'KS Dist',
        'Wasserstein',
        'Rayleigh R',
        'Circ Var',
        'Circ Entropy'
    ]

    print("\n" + "="*80)
    print("SUMMARY TABLE (Mean values across 50 samples)")
    print("="*80)
    print("\nLegend:")
    print("  - KL Div: KL Divergence (0 = uniform, higher = less uniform)")
    print("  - TV Dist: Total Variation Distance (0 = uniform, 1 = maximally different)")
    print("  - KS Dist: Kolmogorov-Smirnov Distance (0 = uniform, 1 = maximally different)")
    print("  - Wasserstein: Earth Mover's Distance")
    print("  - Rayleigh R: Mean resultant length (0 = uniform, 1 = concentrated)")
    print("  - Circ Var: Circular Variance (0 = concentrated, 1 = uniform)")
    print("  - Circ Entropy: Normalized Entropy (0 = concentrated, 1 = uniform)")
    print()
    print(summary_table.to_string())

    # ============================================================
    # Save Results
    # ============================================================

    results_path = results_dir / 'eigenvalue_uniformity_results.csv'
    summary_path = results_dir / 'eigenvalue_uniformity_summary.csv'

    df_results.to_csv(results_path, index=False)
    print(f"\n✓ Detailed results saved to '{results_path}'")

    summary_table.to_csv(summary_path)
    print(f"✓ Summary table saved to '{summary_path}'")

    # ============================================================
    # Interpretation
    # ============================================================

    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("""
For eigenvalues uniformly distributed on the unit circle:
  • KL Divergence ≈ 0 (ideal)
  • TV Distance ≈ 0 (ideal)
  • KS Distance ≈ 0 (ideal)
  • Wasserstein Distance ≈ 0 (ideal)
  • Rayleigh R ≈ 0 (uniform) vs ≈ 1 (concentrated)
  • Circular Variance ≈ 1 (uniform) vs ≈ 0 (concentrated)
  • Circular Entropy ≈ 1 (uniform) vs ≈ 0 (concentrated)

Statistical significance:
  • Rayleigh p-value < 0.05 suggests non-uniformity
  • Small distances + high entropy → eigenvalues are uniformly distributed
""")

    # Display first few rows of detailed results
    print("\nFirst 10 rows of detailed results:")
    print(df_results.head(10).to_string())

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
