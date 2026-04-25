import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_dataset
from scipy import stats
from scipy.stats import wasserstein_distance
from tqdm import tqdm

# Set random seed for reproducible wasserstein distance sampling
np.random.seed(233)

# ============================================================
# Circular Uniformity Metrics (reused from existing analysis)
# ============================================================

def kl_divergence_uniform(angles, n_bins=50):
    angles_0_2pi = angles + np.pi
    bin_edges = np.linspace(0, 2*np.pi, n_bins+1)
    observed, _ = np.histogram(angles_0_2pi, bins=bin_edges, density=False)
    observed_prob = observed / observed.sum()
    uniform_prob = np.full(n_bins, 1.0 / n_bins)
    epsilon = 1e-10
    kl_div = stats.entropy(observed_prob + epsilon, uniform_prob + epsilon)
    return kl_div

def total_variation_distance(angles, n_bins=50):
    angles_0_2pi = angles + np.pi
    bin_edges = np.linspace(0, 2*np.pi, n_bins+1)
    observed, _ = np.histogram(angles_0_2pi, bins=bin_edges, density=False)
    observed_prob = observed / observed.sum()
    uniform_prob = 1.0 / n_bins
    tv_distance = 0.5 * np.sum(np.abs(observed_prob - uniform_prob))
    return tv_distance

def kolmogorov_distance(angles):
    angles_0_2pi = angles + np.pi
    angles_normalized = angles_0_2pi / (2*np.pi)
    ks_stat, _ = stats.kstest(angles_normalized, 'uniform')
    return ks_stat

def wasserstein_distance_uniform(angles):
    angles_0_2pi = angles + np.pi
    uniform_samples = np.random.uniform(0, 2*np.pi, size=len(angles_0_2pi))
    w_dist = wasserstein_distance(angles_0_2pi, uniform_samples)
    return w_dist

def rayleigh_test(angles):
    n = len(angles)
    R = np.abs(np.mean(np.exp(1j * angles)))
    z = n * R**2
    p_value = np.exp(-z)
    return R, z, p_value

def circular_variance(angles):
    circ_var = stats.circvar(angles)
    return circ_var

def circular_entropy(angles, n_bins=50):
    angles_0_2pi = angles + np.pi
    bin_edges = np.linspace(0, 2*np.pi, n_bins+1)
    observed, _ = np.histogram(angles_0_2pi, bins=bin_edges, density=False)
    observed_prob = observed / observed.sum()
    entropy = stats.entropy(observed_prob)
    max_entropy = np.log(n_bins)
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def compute_uniformity_metrics(angles):
    """Compute all circular uniformity metrics for a set of angles."""
    kl_div = kl_divergence_uniform(angles, n_bins=50)
    tv_dist = total_variation_distance(angles, n_bins=50)
    ks_dist = kolmogorov_distance(angles)
    w_dist = wasserstein_distance_uniform(angles)
    R, rayleigh_z, rayleigh_p = rayleigh_test(angles)
    circ_var = circular_variance(angles)
    circ_entropy = circular_entropy(angles, n_bins=50)
    return {
        'kl_divergence': kl_div,
        'tv_distance': tv_dist,
        'kolmogorov_distance': ks_dist,
        'wasserstein_distance': w_dist,
        'rayleigh_R': R,
        'rayleigh_stat': rayleigh_z,
        'rayleigh_p_value': rayleigh_p,
        'circular_variance': circ_var,
        'circular_entropy': circ_entropy,
    }

# ============================================================
# Hook & Processing
# ============================================================

def make_capture_hook(layer_idx, captured_data):
    """Factory to create a forward hook that captures the attention input matrix X."""
    def hook(module, input, output):
        # input[0] is the hidden states: [batch, seq_len, hidden_dim]
        X = input[0].detach().cpu()
        # Remove batch dimension (assuming batch_size=1)
        captured_data[layer_idx] = X.squeeze(0)  # shape: [seq_len, hidden_dim]
    return hook

def process_article(model, tokenizer, text, layer_indices, max_length, device):
    """Tokenize text, run forward pass, capture X per layer, compute eigen angle metrics."""
    # Tokenize with truncation
    encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = encoding['input_ids'].to(device)
    seq_len = input_ids.shape[1]

    # Register hooks
    captured = {}
    hooks = []
    for idx in layer_indices:
        hook = model.h[idx].attn.register_forward_hook(make_capture_hook(idx, captured))
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        model(input_ids=input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute metrics per layer
    rows = []
    for layer_idx in layer_indices:
        X = captured[layer_idx].numpy()  # [seq_len, hidden_dim]
        # Compute full SVD: X = U * Sigma * Vh, U is [seq_len, seq_len] square orthogonal
        U, S, Vh = np.linalg.svd(X, full_matrices=True)
        # Eigenvalues of orthogonal U lie on unit circle
        eigvals = np.linalg.eigvals(U)
        angles = np.angle(eigvals)  # radians in [-pi, pi]
        metrics = compute_uniformity_metrics(angles)
        row = {
            'layer_idx': int(layer_idx),
            'seq_len': int(seq_len),
        }
        row.update(metrics)
        rows.append(row)

    return rows, seq_len

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GPT2 Eigen Angle Statistics per Article")
    parser.add_argument('--output_csv', type=str, default='results/eigen_angle_statistics.csv',
                        help='Path to output CSV file')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum token length per article (truncate longer articles). Default: auto-detect from tokenizer.model_max_length')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for model inference (cpu or cuda)')
    args = parser.parse_args()

    print("=" * 60)
    print("GPT2 Eigen Angle Statistics (per-article)")
    print("=" * 60)

    # Load dataset
    print("\n[1/4] Loading WikiText-103 dataset...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    val_data = dataset['validation']
    print(f"  Total articles in validation split: {len(val_data)}")

    # Filter out empty/short articles
    raw_texts = [ex['text'] for ex in val_data]
    # Keep articles with non-trivial content (> 100 chars)
    texts = [t for t in raw_texts if len(t.strip()) > 100]
    print(f"  After filtering empty/short: {len(texts)} articles")

    # Compute token lengths (without truncation) for statistics
    print("\n[2/4] Computing article length statistics...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect max_length from tokenizer if not specified
    if args.max_length is None:
        args.max_length = tokenizer.model_max_length
        print(f"  Auto-detected max_length from tokenizer: {args.max_length}")

    lengths = []
    for text in texts:
        tokens = tokenizer(text, truncation=False, return_tensors='pt')
        lengths.append(tokens['input_ids'].shape[1])

    lengths_arr = np.array(lengths)
    print(f"  Min length: {lengths_arr.min()}")
    print(f"  Max length: {lengths_arr.max()}")
    print(f"  Mean length: {lengths_arr.mean():.1f}")
    print(f"  Median length: {np.median(lengths_arr):.1f}")
    print(f"  Articles exceeding max_length={args.max_length}: {(lengths_arr > args.max_length).sum()}")

    # Load model
    print("\n[3/4] Loading GPT2 model...")
    model = GPT2Model.from_pretrained('gpt2', attn_implementation='eager', output_attentions=False)
    model.to(args.device)
    model.eval()
    print(f"  Model loaded on {args.device}")
    print(f"  Number of layers: {len(model.h)}")

    layer_indices = list(range(len(model.h)))  # 0..11

    # Process each article
    print(f"\n[4/4] Processing {len(texts)} articles (max_length={args.max_length}, truncating as needed)...")
    all_rows = []
    for idx, text in enumerate(tqdm(texts, desc="Articles", unit="article")):
        try:
            rows, actual_len = process_article(model, tokenizer, text, layer_indices, args.max_length, args.device)
            for row in rows:
                row['article_idx'] = idx
                row['article_token_count_raw'] = lengths[idx]  # before truncation
                all_rows.append(row)
        except Exception as e:
            tqdm.write(f"Error processing article {idx}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Ensure output directory exists
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")

    # Summary statistics by layer
    print("\n" + "=" * 60)
    print("Summary: mean ± std across articles (per layer)")
    print("=" * 60)
    metric_cols = ['kl_divergence', 'tv_distance', 'kolmogorov_distance',
                   'wasserstein_distance', 'rayleigh_R', 'circular_variance',
                   'circular_entropy']
    summary = df.groupby('layer_idx')[metric_cols].agg(['mean', 'std'])
    print(summary.to_string())

    # Overall average across layers
    print("\nOverall average across all layers:")
    overall = df[metric_cols].mean()
    print(overall.to_string())

if __name__ == '__main__':
    main()
