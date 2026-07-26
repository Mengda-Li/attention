"""Rank-proxy comparison for real and spectral-canonical attention.

This experiment complements the direct singular-value-distribution comparison.
For every matched sample, layer, and attention head it constructs

    S_h = Q_h K_h^T / sqrt(d_h),
    Sigma_h = diag(svdvals(S_h)),
    A_real,h = softmax_row(S_h + M),
    A_canonical,h = softmax_row(Sigma_h + M),

where M is the GPT-2 causal mask and is zero for ViT.  It then computes stable
rank, entropy effective ranks erank_1 and erank_2, their paired signed
    differences, total-variation, Kolmogorov, Wasserstein-1, Wasserstein-2,
    and smoothed log-histogram KL divergences between the two per-head
    singular-value empirical distributions.

The default run uses a deterministic common cohort of WikiText-103 validation
documents.  Every document is evaluated at the same sequence lengths by taking
nested prefixes, so changes along the x-axis are not confounded by using
different texts at different lengths.  Imagenette uses a class-balanced
20-image validation subset.  Every layer and every head is retained;
aggregation happens only after the per-head SVD and paired comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision
import transformers
from datasets import Dataset
from torch import nn
from torchvision import datasets as vision_datasets
from torchvision.models import ViT_B_16_Weights
from transformers import AutoTokenizer, GPT2Model, PreTrainedTokenizerBase

from plot_real_vs_canonical_attention_lsd import (
    find_cached_wikitext_validation_arrow,
    iter_wikitext_documents,
    resolve_device,
    set_seed,
)


SEED = 233
DEFAULT_OUTPUT_DIR = Path("results/real_vs_canonical_rank_proxies")
DEFAULT_IMAGENETTE_ROOT = Path.home() / "dataset" / "imagenette2"
ATTENTION_RECONSTRUCTION_TOLERANCE = 5e-5
REAL_COLOR = "#0072B2"
CANONICAL_COLOR = "#D55E00"
DIFFERENCE_COLOR = "#7A5195"
WASSERSTEIN_COLOR = "#2A9D8F"
RANK_METRICS = ("stable_rank", "erank_1", "erank_2")
LSD_DISTANCE_METRICS = (
    "total_variation_distance",
    "kolmogorov_distance",
    "wasserstein_distance",
)
WASSERSTEIN_METRICS = (
    "wasserstein_distance",
    "wasserstein_2_distance",
)
ALL_LSD_DISTANCE_METRICS = (
    *LSD_DISTANCE_METRICS,
    "wasserstein_2_distance",
)
RANK_LABELS = {
    "stable_rank": "Stable rank",
    "erank_1": r"$\mathrm{erank}_1$",
    "erank_2": r"$\mathrm{erank}_2$",
}
LSD_DISTANCE_LABELS = {
    "total_variation_distance": "Histogram total variation",
    "kolmogorov_distance": "Kolmogorov distance",
    "wasserstein_distance": r"Wasserstein distance $W_1$",
}
LSD_DISTANCE_COLORS = {
    "total_variation_distance": "#E76F51",
    "kolmogorov_distance": "#6A4C93",
    "wasserstein_distance": WASSERSTEIN_COLOR,
}
TV_LOG10_MIN = -8.0
TV_LOG10_MAX = 2.0
TV_NUM_BINS = 100
KL_LOG10_MIN = TV_LOG10_MIN
KL_LOG10_MAX = TV_LOG10_MAX
KL_PRIMARY_NUM_BINS = TV_NUM_BINS
KL_PRIMARY_UNIFORM_CONTAMINATION = 1e-3
KL_SENSITIVITY_NUM_BINS = (50, 100)
KL_SENSITIVITY_UNIFORM_CONTAMINATIONS = (1e-4, 1e-3, 1e-2)
KL_SENSITIVITY_SPECS = (
    ("symmetric_kl_divergence_bins_50_eta_1e_4", 50, 1e-4),
    ("symmetric_kl_divergence_bins_50_eta_1e_3", 50, 1e-3),
    ("symmetric_kl_divergence_bins_50_eta_1e_2", 50, 1e-2),
    ("symmetric_kl_divergence_bins_100_eta_1e_4", 100, 1e-4),
    ("symmetric_kl_divergence_bins_100_eta_1e_3", 100, 1e-3),
    ("symmetric_kl_divergence_bins_100_eta_1e_2", 100, 1e-2),
)
KL_PRIMARY_METRICS = (
    "kl_real_to_canonical",
    "kl_canonical_to_real",
    "symmetric_kl_divergence",
)
KL_CLIPPING_METRICS = (
    "real_low_clipping_fraction",
    "real_high_clipping_fraction",
    "canonical_low_clipping_fraction",
    "canonical_high_clipping_fraction",
)
KL_SENSITIVITY_METRICS = tuple(spec[0] for spec in KL_SENSITIVITY_SPECS)
ALL_KL_METRICS = (
    *KL_PRIMARY_METRICS,
    *KL_CLIPPING_METRICS,
    *KL_SENSITIVITY_METRICS,
)


@dataclass(frozen=True)
class TextExample:
    sample_idx: int
    row_start: int
    row_end: int
    seq_len: int
    input_ids: list[int]


def configure_plot_style() -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def row_softmax_batched(score: torch.Tensor, causal: bool) -> torch.Tensor:
    if causal:
        seq_len = int(score.shape[-1])
        forbidden = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=score.device),
            diagonal=1,
        )
        score = score.masked_fill(forbidden, torch.finfo(score.dtype).min)
    return torch.softmax(score, dim=-1)


@torch.inference_mode()
def score_singular_values_from_factors(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Exact nonzero singular values of QK^T times ``scale``, padded to seq_len.

    Compact QR reduces the score SVD to at most head_dim x head_dim while
    preserving the nonzero singular values.
    """

    query_cpu = query.detach().to(device="cpu", dtype=torch.float32)
    key_cpu = key.detach().to(device="cpu", dtype=torch.float32)
    _, query_r = torch.linalg.qr(query_cpu, mode="reduced")
    _, key_r = torch.linalg.qr(key_cpu, mode="reduced")
    core = torch.matmul(query_r, key_r.transpose(-1, -2)) * scale
    nonzero = torch.linalg.svdvals(core)

    seq_len = int(query_cpu.shape[-2])
    if nonzero.shape[-1] == seq_len:
        return nonzero
    padding = torch.zeros(
        (*nonzero.shape[:-1], seq_len - nonzero.shape[-1]),
        dtype=nonzero.dtype,
    )
    return torch.cat([nonzero, padding], dim=-1)


@torch.inference_mode()
def attention_singular_values(attention: torch.Tensor) -> torch.Tensor:
    # Float32 batched SVD can return non-finite values for otherwise finite,
    # highly ill-conditioned attention matrices on some LAPACK backends.
    # The matrices originate in float32, but performing the decomposition in
    # float64 avoids that backend failure and stabilizes the small tail values.
    singular_values = torch.linalg.svdvals(
        attention.detach().to(device="cpu", dtype=torch.float64)
    )
    if not torch.isfinite(singular_values).all():
        raise RuntimeError("Attention SVD produced non-finite singular values.")
    return singular_values


def log_singular_value_histogram(
    values: torch.Tensor,
    num_bins: int,
    log10_min: float = TV_LOG10_MIN,
    log10_max: float = TV_LOG10_MAX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed log10-bin probabilities and clipping fractions per spectrum."""

    values = values.to(dtype=torch.float64)
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("Spectra must have a nonempty singular-value dimension.")
    if num_bins <= 0:
        raise ValueError("The number of histogram bins must be positive.")
    if not log10_min < log10_max:
        raise ValueError("The log10 histogram range must have positive width.")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Singular values must be finite and nonnegative.")

    length = int(values.shape[-1])
    lower = 10.0**log10_min
    upper = 10.0**log10_max
    low_clipping_fraction = (values < lower).to(torch.float64).mean(dim=-1)
    high_clipping_fraction = (values > upper).to(torch.float64).mean(dim=-1)

    log_values = torch.log10(torch.clamp(values, min=lower))
    log_values = torch.clamp(log_values, min=log10_min, max=log10_max)
    scaled = (log_values - log10_min) / (log10_max - log10_min)
    indices = torch.clamp((scaled * num_bins).floor().to(torch.long), 0, num_bins - 1)
    counts = torch.zeros(
        (*values.shape[:-1], num_bins), dtype=torch.float64, device=values.device
    )
    counts.scatter_add_(
        dim=-1,
        index=indices,
        src=torch.ones_like(values, dtype=torch.float64),
    )
    return counts / length, low_clipping_fraction, high_clipping_fraction


def uniformly_smoothed_histogram(
    histogram: torch.Tensor,
    uniform_contamination: float,
) -> torch.Tensor:
    """Mix a histogram with a fixed fraction of the uniform distribution."""

    if histogram.ndim < 1 or histogram.shape[-1] == 0:
        raise ValueError("Histograms must have a nonempty bin dimension.")
    if not 0.0 < uniform_contamination < 1.0:
        raise ValueError("Uniform contamination must lie strictly between zero and one.")
    num_bins = int(histogram.shape[-1])
    return (
        (1.0 - uniform_contamination) * histogram
        + uniform_contamination / num_bins
    )


def directional_kl_divergence(
    source_histogram: torch.Tensor,
    target_histogram: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(source || target) in nats for strictly positive histograms."""

    if source_histogram.shape != target_histogram.shape:
        raise ValueError("KL histograms must have the same shape.")
    if (source_histogram <= 0).any() or (target_histogram <= 0).any():
        raise ValueError("KL histograms must be strictly positive.")
    divergence = torch.sum(
        source_histogram
        * (torch.log(source_histogram) - torch.log(target_histogram)),
        dim=-1,
    )
    return torch.clamp(divergence, min=0.0)


def smoothed_symmetric_kl_divergence(
    real_histogram: torch.Tensor,
    canonical_histogram: torch.Tensor,
    uniform_contamination: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return both directional KLs and their half-sum after uniform smoothing."""

    real_smoothed = uniformly_smoothed_histogram(
        real_histogram, uniform_contamination
    )
    canonical_smoothed = uniformly_smoothed_histogram(
        canonical_histogram, uniform_contamination
    )
    real_to_canonical = directional_kl_divergence(real_smoothed, canonical_smoothed)
    canonical_to_real = directional_kl_divergence(canonical_smoothed, real_smoothed)
    symmetric = 0.5 * (real_to_canonical + canonical_to_real)
    return real_to_canonical, canonical_to_real, symmetric


def lsd_distances(
    real_singular_values: torch.Tensor,
    canonical_singular_values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare paired LSDs, preserving the head dimension.

    Wasserstein-1, Wasserstein-2, and Kolmogorov distances are computed exactly
    for the equal-mass empirical measures of the raw singular values. Exact TV
    and KL between two finite empirical measures are generally uninformative or
    infinite because their floating-point atoms do not coincide. TV therefore
    uses common fixed log10 histograms. KL uses the same primary histograms with
    fixed uniform-contamination smoothing and is reported in both directions
    and as their half-sum.
    """

    real = real_singular_values.to(dtype=torch.float64)
    canonical = canonical_singular_values.to(dtype=torch.float64)
    if real.shape != canonical.shape:
        raise ValueError("Real and canonical spectra must have the same shape.")
    length = int(real.shape[-1])

    real_sorted = torch.sort(real, dim=-1).values
    canonical_sorted = torch.sort(canonical, dim=-1).values
    paired_gaps = torch.abs(real_sorted - canonical_sorted)
    wasserstein = torch.mean(paired_gaps, dim=-1)
    wasserstein_2 = torch.sqrt(torch.mean(paired_gaps.square(), dim=-1))

    pooled = torch.sort(torch.cat([real_sorted, canonical_sorted], dim=-1), dim=-1).values
    real_cdf = torch.searchsorted(real_sorted, pooled, right=True).to(torch.float64) / length
    canonical_cdf = (
        torch.searchsorted(canonical_sorted, pooled, right=True).to(torch.float64) / length
    )
    kolmogorov = torch.amax(torch.abs(real_cdf - canonical_cdf), dim=-1)

    real_histogram, real_low_clip, real_high_clip = log_singular_value_histogram(
        real,
        num_bins=TV_NUM_BINS,
    )
    canonical_histogram, canonical_low_clip, canonical_high_clip = (
        log_singular_value_histogram(
            canonical,
            num_bins=TV_NUM_BINS,
        )
    )
    total_variation = 0.5 * torch.sum(torch.abs(real_histogram - canonical_histogram), dim=-1)
    real_to_canonical_kl, canonical_to_real_kl, symmetric_kl = (
        smoothed_symmetric_kl_divergence(
            real_histogram,
            canonical_histogram,
            uniform_contamination=KL_PRIMARY_UNIFORM_CONTAMINATION,
        )
    )

    distances = {
        "total_variation_distance": total_variation,
        "kolmogorov_distance": kolmogorov,
        "wasserstein_distance": wasserstein,
        "wasserstein_2_distance": wasserstein_2,
        "kl_real_to_canonical": real_to_canonical_kl,
        "kl_canonical_to_real": canonical_to_real_kl,
        "symmetric_kl_divergence": symmetric_kl,
        "real_low_clipping_fraction": real_low_clip,
        "real_high_clipping_fraction": real_high_clip,
        "canonical_low_clipping_fraction": canonical_low_clip,
        "canonical_high_clipping_fraction": canonical_high_clip,
    }

    histograms_by_num_bins = {
        KL_PRIMARY_NUM_BINS: (real_histogram, canonical_histogram),
    }
    for column, num_bins, uniform_contamination in KL_SENSITIVITY_SPECS:
        if num_bins not in histograms_by_num_bins:
            sensitivity_real, _, _ = log_singular_value_histogram(
                real,
                num_bins=num_bins,
            )
            sensitivity_canonical, _, _ = log_singular_value_histogram(
                canonical,
                num_bins=num_bins,
            )
            histograms_by_num_bins[num_bins] = (
                sensitivity_real,
                sensitivity_canonical,
            )
        sensitivity_real, sensitivity_canonical = histograms_by_num_bins[num_bins]
        _, _, sensitivity_symmetric = smoothed_symmetric_kl_divergence(
            sensitivity_real,
            sensitivity_canonical,
            uniform_contamination=uniform_contamination,
        )
        distances[column] = sensitivity_symmetric
    return distances


def run_kl_unit_checks() -> None:
    """Exercise finite-histogram KL invariants without loading a model."""

    expected_sensitivity_grid = {
        (num_bins, uniform_contamination)
        for num_bins in KL_SENSITIVITY_NUM_BINS
        for uniform_contamination in KL_SENSITIVITY_UNIFORM_CONTAMINATIONS
    }
    actual_sensitivity_grid = {
        (num_bins, uniform_contamination)
        for _, num_bins, uniform_contamination in KL_SENSITIVITY_SPECS
    }
    if actual_sensitivity_grid != expected_sensitivity_grid:
        raise AssertionError("KL sensitivity specifications do not cover the requested grid.")
    if len(set(ALL_KL_METRICS)) != len(ALL_KL_METRICS):
        raise AssertionError("KL export column names must be unique.")

    real = torch.tensor(
        [
            [0.0, 1e-9, 1e-5, 1e-2, 1.0, 120.0],
            [2e-8, 3e-6, 4e-4, 0.2, 0.9, 2.0],
        ],
        dtype=torch.float64,
    )
    canonical = torch.tensor(
        [
            [0.0, 2e-8, 2e-4, 3e-2, 0.8, 3.0],
            [1e-9, 8e-6, 7e-4, 0.1, 1.1, 1.8],
        ],
        dtype=torch.float64,
    )

    identity = lsd_distances(real, real)
    zeros = torch.zeros(real.shape[0], dtype=torch.float64)
    for column in (*KL_PRIMARY_METRICS, *KL_SENSITIVITY_METRICS):
        torch.testing.assert_close(identity[column], zeros, rtol=0.0, atol=0.0)

    forward = lsd_distances(real, canonical)
    reverse = lsd_distances(canonical, real)
    torch.testing.assert_close(
        forward["kl_real_to_canonical"],
        reverse["kl_canonical_to_real"],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        forward["kl_canonical_to_real"],
        reverse["kl_real_to_canonical"],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        forward["symmetric_kl_divergence"],
        reverse["symmetric_kl_divergence"],
        rtol=1e-12,
        atol=1e-12,
    )

    exact_symmetric = 0.5 * (
        forward["kl_real_to_canonical"]
        + forward["kl_canonical_to_real"]
    )
    torch.testing.assert_close(
        forward["symmetric_kl_divergence"],
        exact_symmetric,
        rtol=0.0,
        atol=0.0,
    )

    real_permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    canonical_permutation = torch.tensor([1, 5, 3, 0, 4, 2])
    permuted = lsd_distances(
        real[:, real_permutation],
        canonical[:, canonical_permutation],
    )
    for column in (*KL_PRIMARY_METRICS, *KL_SENSITIVITY_METRICS):
        torch.testing.assert_close(
            forward[column],
            permuted[column],
            rtol=1e-12,
            atol=1e-12,
        )
        if not torch.isfinite(forward[column]).all() or (forward[column] < 0).any():
            raise AssertionError(f"KL unit check failed for {column}.")

    histogram, low_clip, high_clip = log_singular_value_histogram(
        real,
        num_bins=KL_PRIMARY_NUM_BINS,
    )
    torch.testing.assert_close(
        histogram.sum(dim=-1),
        torch.ones(real.shape[0], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    smoothed = uniformly_smoothed_histogram(
        histogram,
        KL_PRIMARY_UNIFORM_CONTAMINATION,
    )
    torch.testing.assert_close(
        smoothed.sum(dim=-1),
        torch.ones(real.shape[0], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    minimum_mass = KL_PRIMARY_UNIFORM_CONTAMINATION / KL_PRIMARY_NUM_BINS
    if (smoothed < minimum_mass - 1e-15).any():
        raise AssertionError("Uniform smoothing produced a bin below its probability floor.")
    expected_low_clip = torch.tensor([2 / 6, 0.0], dtype=torch.float64)
    expected_high_clip = torch.tensor([1 / 6, 0.0], dtype=torch.float64)
    torch.testing.assert_close(low_clip, expected_low_clip, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(high_clip, expected_high_clip, rtol=1e-12, atol=1e-12)

    primary_sensitivity_column = "symmetric_kl_divergence_bins_100_eta_1e_3"
    torch.testing.assert_close(
        forward[primary_sensitivity_column],
        forward["symmetric_kl_divergence"],
        rtol=0.0,
        atol=0.0,
    )


def rank_proxies_from_singular_values(singular_values: torch.Tensor) -> dict[str, torch.Tensor]:
    singular_values = singular_values.to(dtype=torch.float64)
    squared = singular_values.square()

    max_squared = squared[..., 0]
    stable = torch.where(
        max_squared > 0,
        squared.sum(dim=-1) / max_squared,
        torch.zeros_like(max_squared),
    )

    sum_l1 = singular_values.sum(dim=-1, keepdim=True)
    q1 = torch.where(sum_l1 > 0, singular_values / sum_l1, torch.zeros_like(singular_values))
    entropy_1 = -torch.where(q1 > 0, q1 * torch.log(q1), torch.zeros_like(q1)).sum(dim=-1)
    erank_1 = torch.where(
        sum_l1.squeeze(-1) > 0,
        torch.exp(entropy_1),
        torch.zeros_like(entropy_1),
    )

    sum_l2 = squared.sum(dim=-1, keepdim=True)
    q2 = torch.where(sum_l2 > 0, squared / sum_l2, torch.zeros_like(squared))
    entropy_2 = -torch.where(q2 > 0, q2 * torch.log(q2), torch.zeros_like(q2)).sum(dim=-1)
    erank_2 = torch.where(
        sum_l2.squeeze(-1) > 0,
        torch.exp(entropy_2),
        torch.zeros_like(entropy_2),
    )

    return {
        "stable_rank": stable,
        "erank_1": erank_1,
        "erank_2": erank_2,
    }


def paired_metric_rows(
    real_singular_values: torch.Tensor,
    canonical_singular_values: torch.Tensor,
    base: dict[str, Any],
    reconstruction_errors: torch.Tensor,
) -> list[dict[str, Any]]:
    real_proxies = rank_proxies_from_singular_values(real_singular_values)
    canonical_proxies = rank_proxies_from_singular_values(canonical_singular_values)
    distances = lsd_distances(real_singular_values, canonical_singular_values)

    rows: list[dict[str, Any]] = []
    num_heads = int(real_singular_values.shape[0])
    for head_idx in range(num_heads):
        row = {**base, "head_idx": head_idx}
        for metric in RANK_METRICS:
            real_value = float(real_proxies[metric][head_idx].item())
            canonical_value = float(canonical_proxies[metric][head_idx].item())
            row[f"real_{metric}"] = real_value
            row[f"canonical_{metric}"] = canonical_value
            row[f"delta_{metric}"] = real_value - canonical_value
        for distance in ALL_LSD_DISTANCE_METRICS:
            row[distance] = float(distances[distance][head_idx].item())
        for metric in ALL_KL_METRICS:
            row[metric] = float(distances[metric][head_idx].item())
        row["attention_reconstruction_max_abs_error"] = float(
            reconstruction_errors[head_idx].item()
        )
        rows.append(row)
    return rows


def load_wikitext_examples(
    tokenizer: PreTrainedTokenizerBase,
    sequence_lengths: Iterable[int],
    num_text_samples: int,
    seed: int,
    arrow_path: Path | None,
) -> tuple[list[TextExample], dict[str, Any]]:
    """Select a common document cohort and return nested prefixes at each length."""

    resolved_lengths = sorted({int(length) for length in sequence_lengths})
    if not resolved_lengths or resolved_lengths[0] <= 0:
        raise ValueError("At least one positive GPT-2 sequence length is required.")

    resolved_arrow = (
        arrow_path.expanduser() if arrow_path is not None else find_cached_wikitext_validation_arrow()
    )
    validation = Dataset.from_file(str(resolved_arrow))

    original_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = 10**9
    documents: list[dict[str, Any]] = []
    try:
        for row_start, row_end, text in iter_wikitext_documents(validation):
            input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            documents.append(
                {
                    "row_start": int(row_start),
                    "row_end": int(row_end),
                    "token_count": len(input_ids),
                    "input_ids": input_ids,
                    "text_preview": " ".join(text.split())[:200],
                }
            )
    finally:
        tokenizer.model_max_length = original_max_length

    required_token_count = resolved_lengths[-1]
    eligible = [
        document for document in documents if int(document["token_count"]) >= required_token_count
    ]
    if not eligible:
        longest = max((int(document["token_count"]) for document in documents), default=0)
        raise RuntimeError(
            f"No WikiText validation document has {required_token_count} tokens; "
            f"the longest has {longest}."
        )

    rng = np.random.default_rng(seed)
    if num_text_samples <= 0 or num_text_samples >= len(eligible):
        selected = eligible
    else:
        indices = np.sort(rng.choice(len(eligible), size=num_text_samples, replace=False))
        selected = [eligible[int(index)] for index in indices]
    selected.sort(key=lambda item: (int(item["row_start"]), int(item["row_end"])))

    examples: list[TextExample] = []
    for sample_idx, item in enumerate(selected):
        for seq_len in resolved_lengths:
            examples.append(
                TextExample(
                    sample_idx=sample_idx,
                    row_start=int(item["row_start"]),
                    row_end=int(item["row_end"]),
                    seq_len=seq_len,
                    input_ids=list(item["input_ids"][:seq_len]),
                )
            )

    metadata = {
        "arrow_path": str(resolved_arrow),
        "validation_rows": len(validation),
        "validation_documents": len(documents),
        "minimum_document_tokens": required_token_count,
        "eligible_documents": len(eligible),
        "selected_documents": len(selected),
        "sequence_lengths": resolved_lengths,
        "document_length_design": "same nested prefix lengths for every selected document",
        "selected_document_metadata": [
            {
                "sample_idx": sample_idx,
                "row_start": int(item["row_start"]),
                "row_end": int(item["row_end"]),
                "full_document_token_count": int(item["token_count"]),
                "text_preview": str(item["text_preview"]),
            }
            for sample_idx, item in enumerate(selected)
        ],
        "model_inputs": len(examples),
    }
    return examples, metadata


def make_gpt2_capture_hook(
    layer_idx: int,
    captured_inputs: dict[int, torch.Tensor],
    captured_attention: dict[int, torch.Tensor],
):
    def hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        _ = module
        captured_inputs[layer_idx] = inputs[0].detach()
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            raise RuntimeError(
                "GPT-2 eager attention did not return per-head attention weights."
            )
        captured_attention[layer_idx] = output[1].detach()

    return hook


@torch.inference_mode()
def collect_gpt2_metrics(
    device: torch.device,
    output_csv: Path,
    sequence_lengths: Iterable[int],
    num_text_samples: int,
    seed: int,
    arrow_path: Path | None,
    max_layers: int | None,
    max_heads: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    model = GPT2Model.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    model.to(device)

    examples, sample_metadata = load_wikitext_examples(
        tokenizer=tokenizer,
        sequence_lengths=sequence_lengths,
        num_text_samples=num_text_samples,
        seed=seed,
        arrow_path=arrow_path,
    )

    max_sequence_length = max(example.seq_len for example in examples)
    model_context_length = int(model.config.n_positions)
    if max_sequence_length > model_context_length:
        raise ValueError(
            f"Requested sequence length {max_sequence_length} exceeds GPT-2 context "
            f"length {model_context_length}."
        )

    n_layers_available = len(model.h)
    n_layers = n_layers_available if max_layers is None else min(max_layers, n_layers_available)
    n_heads_available = int(model.config.num_attention_heads)
    n_heads = n_heads_available if max_heads is None else min(max_heads, n_heads_available)

    captured_inputs: dict[int, torch.Tensor] = {}
    captured_attention: dict[int, torch.Tensor] = {}
    hooks = [
        model.h[layer_idx].attn.register_forward_hook(
            make_gpt2_capture_hook(layer_idx, captured_inputs, captured_attention)
        )
        for layer_idx in range(n_layers)
    ]

    rows: list[dict[str, Any]] = []
    max_reconstruction_error = 0.0
    try:
        for sample_number, example in enumerate(examples, start=1):
            captured_inputs.clear()
            captured_attention.clear()
            input_ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
            model(input_ids=input_ids, use_cache=False)

            for layer_idx in range(n_layers):
                hidden_states = captured_inputs[layer_idx]
                attention_module = model.h[layer_idx].attn
                qkv = attention_module.c_attn(hidden_states)
                query, key, _ = qkv.split(attention_module.split_size, dim=2)
                query = query.view(
                    1,
                    example.seq_len,
                    n_heads_available,
                    attention_module.head_dim,
                ).transpose(1, 2)[0, :n_heads]
                key = key.view(
                    1,
                    example.seq_len,
                    n_heads_available,
                    attention_module.head_dim,
                ).transpose(1, 2)[0, :n_heads]

                scale = float(attention_module.scaling)
                score = torch.matmul(query, key.transpose(-1, -2)) * scale
                real_attention = captured_attention[layer_idx][0, :n_heads]
                manual_attention = row_softmax_batched(score, causal=True)
                reconstruction_errors = torch.amax(
                    torch.abs(real_attention - manual_attention).flatten(start_dim=1),
                    dim=1,
                )
                reconstruction_error = float(reconstruction_errors.max().item())
                max_reconstruction_error = max(max_reconstruction_error, reconstruction_error)
                if reconstruction_error > ATTENTION_RECONSTRUCTION_TOLERANCE:
                    raise RuntimeError(
                        f"GPT-2 layer {layer_idx} reconstruction error "
                        f"{reconstruction_error:.3e} exceeds tolerance."
                    )

                score_singular_values = score_singular_values_from_factors(
                    query=query,
                    key=key,
                    scale=scale,
                )
                canonical_attention = row_softmax_batched(
                    torch.diag_embed(score_singular_values),
                    causal=True,
                )
                real_singular_values = attention_singular_values(real_attention)
                canonical_singular_values = attention_singular_values(canonical_attention)

                rows.extend(
                    paired_metric_rows(
                        real_singular_values=real_singular_values,
                        canonical_singular_values=canonical_singular_values,
                        base={
                            "model": "GPT-2",
                            "dataset": "WikiText-103 validation",
                            "sample_idx": example.sample_idx,
                            "row_start": example.row_start,
                            "row_end": example.row_end,
                            "seq_len": example.seq_len,
                            "layer_idx": layer_idx,
                        },
                        reconstruction_errors=reconstruction_errors,
                    )
                )

            if sample_number == 1 or sample_number % 10 == 0 or sample_number == len(examples):
                print(
                    f"GPT-2 document-length inputs {sample_number}/{len(examples)} "
                    f"(document={example.sample_idx}, seq_len={example.seq_len})"
                )
    finally:
        for hook in hooks:
            hook.remove()

    metrics = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_csv, index=False)

    metadata = {
        **sample_metadata,
        "model": "gpt2",
        "causal": True,
        "num_layers": n_layers,
        "num_heads": n_heads,
        "num_layers_available": n_layers_available,
        "num_heads_available": n_heads_available,
        "head_dim": int(model.h[0].attn.head_dim),
        "head_rows": len(metrics),
        "max_attention_reconstruction_error": max_reconstruction_error,
        "device": str(device),
        "context_length": model_context_length,
    }

    del model, rows, captured_inputs, captured_attention
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return metrics, metadata


def make_vit_capture_hook(layer_idx: int, captured: dict[int, torch.Tensor]):
    def hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        _ = module
        _ = inputs
        captured[layer_idx] = output.detach()

    return hook


def select_balanced_imagenette_indices(
    imagenette: vision_datasets.ImageFolder,
    num_samples: int,
    seed: int,
) -> list[int]:
    if num_samples <= 0 or num_samples >= len(imagenette):
        return list(range(len(imagenette)))

    by_class: dict[int, list[int]] = defaultdict(list)
    for sample_idx, (_, class_idx) in enumerate(imagenette.samples):
        by_class[int(class_idx)].append(sample_idx)

    rng = np.random.default_rng(seed)
    for class_indices in by_class.values():
        rng.shuffle(class_indices)

    selected: list[int] = []
    round_idx = 0
    classes = sorted(by_class)
    while len(selected) < num_samples:
        added = False
        for class_idx in classes:
            class_indices = by_class[class_idx]
            if round_idx < len(class_indices):
                selected.append(class_indices[round_idx])
                added = True
                if len(selected) == num_samples:
                    break
        if not added:
            break
        round_idx += 1
    return selected


@torch.inference_mode()
def collect_vit_metrics(
    device: torch.device,
    output_csv: Path,
    imagenette_root: Path,
    num_image_samples: int,
    seed: int,
    max_layers: int | None,
    max_heads: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    validation_root = imagenette_root.expanduser() / "val"
    if not validation_root.is_dir():
        raise FileNotFoundError(f"Imagenette validation directory not found: {validation_root}")

    imagenette = vision_datasets.ImageFolder(
        root=validation_root,
        transform=weights.transforms(),
    )
    selected_indices = select_balanced_imagenette_indices(
        imagenette=imagenette,
        num_samples=num_image_samples,
        seed=seed,
    )

    model = torchvision.models.vit_b_16(weights=weights)
    model.eval()
    model.to(device)

    n_layers_available = len(model.encoder.layers)
    n_layers = n_layers_available if max_layers is None else min(max_layers, n_layers_available)
    attention_probe = model.encoder.layers[0].self_attention
    n_heads_available = int(attention_probe.num_heads)
    n_heads = n_heads_available if max_heads is None else min(max_heads, n_heads_available)
    embed_dim = int(attention_probe.embed_dim)
    head_dim = int(embed_dim // n_heads_available)

    captured_ln1: dict[int, torch.Tensor] = {}
    hooks = [
        model.encoder.layers[layer_idx].ln_1.register_forward_hook(
            make_vit_capture_hook(layer_idx, captured_ln1)
        )
        for layer_idx in range(n_layers)
    ]

    rows: list[dict[str, Any]] = []
    max_reconstruction_error = 0.0
    selected_sample_metadata: list[dict[str, Any]] = []
    try:
        for sample_number, dataset_idx in enumerate(selected_indices, start=1):
            image, class_idx = imagenette[dataset_idx]
            sample_path = Path(imagenette.samples[dataset_idx][0])
            class_name = imagenette.classes[class_idx]
            selected_sample_metadata.append(
                {
                    "sample_idx": sample_number - 1,
                    "dataset_idx": dataset_idx,
                    "sample_path": str(sample_path),
                    "class_synset": class_name,
                }
            )

            captured_ln1.clear()
            image_batch = image.unsqueeze(0).to(device)
            model(image_batch)

            seq_len = int(captured_ln1[0].shape[1])
            for layer_idx in range(n_layers):
                hidden_states = captured_ln1[layer_idx]
                attention_module = model.encoder.layers[layer_idx].self_attention
                in_proj_weight = attention_module.in_proj_weight
                in_proj_bias = attention_module.in_proj_bias

                query = F.linear(
                    hidden_states,
                    in_proj_weight[:embed_dim],
                    None if in_proj_bias is None else in_proj_bias[:embed_dim],
                )
                key = F.linear(
                    hidden_states,
                    in_proj_weight[embed_dim : 2 * embed_dim],
                    None if in_proj_bias is None else in_proj_bias[embed_dim : 2 * embed_dim],
                )
                query = query.view(1, seq_len, n_heads_available, head_dim).transpose(1, 2)[
                    0, :n_heads
                ]
                key = key.view(1, seq_len, n_heads_available, head_dim).transpose(1, 2)[
                    0, :n_heads
                ]
                scale = 1.0 / math.sqrt(float(head_dim))
                score = torch.matmul(query, key.transpose(-1, -2)) * scale
                manual_attention = row_softmax_batched(score, causal=False)

                _, actual_attention = attention_module(
                    hidden_states,
                    hidden_states,
                    hidden_states,
                    need_weights=True,
                    average_attn_weights=False,
                )
                real_attention = actual_attention[0, :n_heads]
                reconstruction_errors = torch.amax(
                    torch.abs(real_attention - manual_attention).flatten(start_dim=1),
                    dim=1,
                )
                reconstruction_error = float(reconstruction_errors.max().item())
                max_reconstruction_error = max(max_reconstruction_error, reconstruction_error)
                if reconstruction_error > ATTENTION_RECONSTRUCTION_TOLERANCE:
                    raise RuntimeError(
                        f"ViT layer {layer_idx} reconstruction error "
                        f"{reconstruction_error:.3e} exceeds tolerance."
                    )

                score_singular_values = score_singular_values_from_factors(
                    query=query,
                    key=key,
                    scale=scale,
                )
                canonical_attention = row_softmax_batched(
                    torch.diag_embed(score_singular_values),
                    causal=False,
                )
                real_singular_values = attention_singular_values(real_attention)
                canonical_singular_values = attention_singular_values(canonical_attention)

                rows.extend(
                    paired_metric_rows(
                        real_singular_values=real_singular_values,
                        canonical_singular_values=canonical_singular_values,
                        base={
                            "model": "ViT-B/16",
                            "dataset": "Imagenette validation",
                            "sample_idx": sample_number - 1,
                            "dataset_idx": dataset_idx,
                            "sample_id": sample_path.name,
                            "class_synset": class_name,
                            "seq_len": seq_len,
                            "layer_idx": layer_idx,
                        },
                        reconstruction_errors=reconstruction_errors,
                    )
                )

            print(
                f"ViT samples {sample_number}/{len(selected_indices)} "
                f"({class_name}/{sample_path.name})"
            )
    finally:
        for hook in hooks:
            hook.remove()

    metrics = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_csv, index=False)

    class_counts = metrics[["sample_idx", "class_synset"]].drop_duplicates()[
        "class_synset"
    ].value_counts()
    metadata = {
        "model": "vit_b_16",
        "weights": weights.name,
        "causal": False,
        "imagenette_root": str(imagenette_root.expanduser()),
        "validation_images": len(imagenette),
        "selected_images": len(selected_indices),
        "selected_class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "selected_samples": selected_sample_metadata,
        "num_layers": n_layers,
        "num_heads": n_heads,
        "num_layers_available": n_layers_available,
        "num_heads_available": n_heads_available,
        "head_dim": head_dim,
        "seq_len": int(metrics["seq_len"].iloc[0]),
        "head_rows": len(metrics),
        "max_attention_reconstruction_error": max_reconstruction_error,
        "device": str(device),
    }

    del model, rows, captured_ln1
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return metrics, metadata


def sample_level_gpt2_summary(
    metrics: pd.DataFrame,
    layer_scope: str,
) -> pd.DataFrame:
    if layer_scope == "all":
        selected = metrics
    elif layer_scope == "first":
        selected = metrics[metrics["layer_idx"] == metrics["layer_idx"].min()]
    elif layer_scope == "last":
        selected = metrics[metrics["layer_idx"] == metrics["layer_idx"].max()]
    else:
        raise ValueError(f"Unknown layer scope: {layer_scope}")

    value_columns = list(ALL_LSD_DISTANCE_METRICS)
    for metric in RANK_METRICS:
        value_columns.extend(
            [f"real_{metric}", f"canonical_{metric}", f"delta_{metric}"]
        )
    summary = (
        selected.groupby(["sample_idx", "row_start", "row_end", "seq_len"], as_index=False)[
            value_columns
        ]
        .median()
        .sort_values(["seq_len", "sample_idx"])
    )
    return summary


def select_gpt2_scope(metrics: pd.DataFrame, layer_scope: str) -> pd.DataFrame:
    """Select raw per-head rows for a plotting scope without averaging heads."""

    if layer_scope == "all":
        return metrics
    if layer_scope == "first":
        return metrics[metrics["layer_idx"] == metrics["layer_idx"].min()]
    if layer_scope == "last":
        return metrics[metrics["layer_idx"] == metrics["layer_idx"].max()]
    raise ValueError(f"Unknown layer scope: {layer_scope}")


def sequence_quantile_summary(
    sample_summary: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seq_len, group in sample_summary.groupby("seq_len", sort=True):
        row: dict[str, Any] = {
            "seq_len": int(seq_len),
            "sample_count": int(group["sample_idx"].nunique()),
            "observation_count": int(len(group)),
            "layer_count": int(group["layer_idx"].nunique())
            if "layer_idx" in group
            else None,
            "head_count": int(group["head_idx"].nunique())
            if "head_idx" in group
            else None,
        }
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_median"] = float(np.quantile(values, 0.50))
            row[f"{column}_q25"] = float(np.quantile(values, 0.25))
            row[f"{column}_q75"] = float(np.quantile(values, 0.75))
            row[f"{column}_q10"] = float(np.quantile(values, 0.10))
            row[f"{column}_q90"] = float(np.quantile(values, 0.90))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seq_len")


def draw_quantile_curve(
    ax: plt.Axes,
    summary: pd.DataFrame,
    column: str,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    x = summary["seq_len"].to_numpy(dtype=float)
    median = summary[f"{column}_median"].to_numpy(dtype=float)
    q25 = summary[f"{column}_q25"].to_numpy(dtype=float)
    q75 = summary[f"{column}_q75"].to_numpy(dtype=float)
    q10 = summary[f"{column}_q10"].to_numpy(dtype=float)
    q90 = summary[f"{column}_q90"].to_numpy(dtype=float)
    ax.fill_between(x, q10, q90, color=color, alpha=0.06, linewidth=0)
    ax.fill_between(x, q25, q75, color=color, alpha=0.20, linewidth=0)
    ax.plot(x, median, color=color, linestyle=linestyle, linewidth=2.0, label=label)


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs: list[Path] = []
    for suffix in (".png", ".svg"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def scope_display_name(scope: str, metrics: pd.DataFrame) -> str:
    if scope == "all":
        return "all layers"
    if scope == "first":
        return f"layer {int(metrics['layer_idx'].min()) + 1}"
    if scope == "last":
        return f"layer {int(metrics['layer_idx'].max()) + 1}"
    raise ValueError(scope)


def plot_gpt2_rank_proxy_scope(
    metrics: pd.DataFrame,
    scope: str,
    output_dir: Path,
) -> tuple[list[Path], pd.DataFrame]:
    selected = select_gpt2_scope(metrics, scope)
    columns: list[str] = []
    for metric in RANK_METRICS:
        columns.extend([f"real_{metric}", f"canonical_{metric}", f"delta_{metric}"])
    binned = sequence_quantile_summary(selected, columns)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharex="col")
    for column_idx, metric in enumerate(RANK_METRICS):
        top_ax = axes[0, column_idx]
        bottom_ax = axes[1, column_idx]
        draw_quantile_curve(
            top_ax,
            binned,
            f"real_{metric}",
            REAL_COLOR,
            r"Real $A_h$",
        )
        draw_quantile_curve(
            top_ax,
            binned,
            f"canonical_{metric}",
            CANONICAL_COLOR,
            r"Canonical $A_{\Sigma,h}$",
            linestyle="--",
        )
        top_ax.set_title(RANK_LABELS[metric])
        top_ax.set_ylabel("Raw rank proxy")
        top_ax.grid(alpha=0.25)

        draw_quantile_curve(
            bottom_ax,
            binned,
            f"delta_{metric}",
            DIFFERENCE_COLOR,
            r"$\Delta=$ real $-$ canonical",
        )
        bottom_ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle=":")
        bottom_ax.set_xlabel("Sequence length")
        bottom_ax.set_ylabel(r"Paired difference $\Delta$")
        bottom_ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2)
    scope_name = scope_display_name(scope, metrics)
    fig.suptitle(
        f"GPT-2 real vs. spectral-canonical attention rank proxies: {scope_name}",
        y=0.99,
        fontsize=14,
    )
    num_texts = int(selected["sample_idx"].nunique())
    scope_observation_text = (
        "all layer-head pairs"
        if scope == "all"
        else "all heads in the selected layer"
    )
    fig.text(
        0.5,
        0.008,
        f"Curves are medians with IQR (dark band) and 10–90% range (light band) "
        f"over paired per-head observations ({scope_observation_text}) from the same "
        f"{num_texts} WikiText documents. Bands are descriptive, not confidence intervals.\n"
        "The bottom row uses paired per-head differences, not differences of the two top medians.  "
        r"Positive $\Delta$ means real attention has the larger rank proxy.",
        ha="center",
        fontsize=8,
        linespacing=1.35,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.88))
    paths = save_figure(fig, output_dir / f"gpt2_rank_proxies_{scope}_vs_seq_len")
    binned = binned.assign(layer_scope=scope)
    return paths, binned


def plot_gpt2_lsd_distances(
    metrics: pd.DataFrame,
    output_dir: Path,
) -> tuple[list[Path], pd.DataFrame]:
    scopes = ("all", "first", "last")
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.0), sharex="col")
    summaries: list[pd.DataFrame] = []
    for row_idx, scope in enumerate(scopes):
        selected = select_gpt2_scope(metrics, scope)
        binned = sequence_quantile_summary(selected, LSD_DISTANCE_METRICS)
        summaries.append(binned.assign(layer_scope=scope))
        for column_idx, distance in enumerate(LSD_DISTANCE_METRICS):
            ax = axes[row_idx, column_idx]
            draw_quantile_curve(
                ax,
                binned,
                distance,
                LSD_DISTANCE_COLORS[distance],
                LSD_DISTANCE_LABELS[distance],
            )
            if row_idx == 0:
                ax.set_title(LSD_DISTANCE_LABELS[distance])
            if row_idx == len(scopes) - 1:
                ax.set_xlabel("Sequence length")
            if column_idx == 0:
                ax.set_ylabel(f"{scope_display_name(scope, metrics).capitalize()}\nDistance")
            if distance in ("total_variation_distance", "kolmogorov_distance"):
                ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.25)
    fig.suptitle(
        "GPT-2 real vs. spectral-canonical per-head LSD distances",
        y=0.99,
        fontsize=14,
    )
    num_texts = int(metrics["sample_idx"].nunique())
    fig.text(
        0.5,
        0.006,
        rf"TV uses {TV_NUM_BINS} fixed log10 bins on [{TV_LOG10_MIN:g}, {TV_LOG10_MAX:g}]; "
        r"Kolmogorov is $\sup_x|F_h(x)-F_{\Sigma,h}(x)|$; "
        r"$W_1=\ell^{-1}\sum_j|\sigma_j(A_h)-\sigma_j(A_{\Sigma,h})|$ on raw values."
        "\n"
        f"Lines/bands are descriptive median, IQR, and 10–90% over paired head-level "
        f"observations from the same {num_texts} documents.",
        ha="center",
        fontsize=8,
        linespacing=1.35,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.95))
    paths = save_figure(fig, output_dir / "gpt2_lsd_distances_vs_seq_len")
    return paths, pd.concat(summaries, ignore_index=True)


def vit_image_layer_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = list(ALL_LSD_DISTANCE_METRICS) + [
        f"delta_{metric}" for metric in RANK_METRICS
    ]
    return (
        metrics.groupby(
            ["sample_idx", "dataset_idx", "sample_id", "class_synset", "seq_len", "layer_idx"],
            as_index=False,
        )[columns]
        .median()
        .sort_values(["layer_idx", "sample_idx"])
    )


def plot_vit_rank_proxy_differences(
    metrics: pd.DataFrame,
    output_dir: Path,
) -> tuple[list[Path], pd.DataFrame]:
    image_summary = vit_image_layer_summary(metrics)
    plot_data = metrics.assign(
        layer=(metrics["layer_idx"].astype(int) + 1).astype(str)
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, metric in zip(axes, RANK_METRICS, strict=True):
        column = f"delta_{metric}"
        sns.boxplot(
            data=plot_data,
            x="layer",
            y=column,
            color=DIFFERENCE_COLOR,
            width=0.62,
            whis=1.5,
            showfliers=False,
            ax=ax,
        )
        sns.stripplot(
            data=plot_data,
            x="layer",
            y=column,
            color="0.20",
            size=1.6,
            alpha=0.24,
            jitter=0.20,
            ax=ax,
        )
        ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle=":")
        ax.set_title(RANK_LABELS[metric])
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"Paired difference $\Delta$")
        ax.grid(axis="y", alpha=0.25)

    num_images = int(image_summary["sample_idx"].nunique())
    num_heads = int(metrics["head_idx"].nunique())
    fig.suptitle(
        "ViT-B/16 rank-proxy difference: real minus spectral-canonical attention",
        y=0.99,
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        f"Boxes and points show paired per-head differences ({num_images} images × "
        f"{num_heads} heads per layer); heads within an image are descriptive repeats,\n"
        "not independent confidence replicates.  "
        r"Positive $\Delta$ means real attention has the larger rank proxy.",
        ha="center",
        fontsize=8,
        linespacing=1.35,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.92))
    paths = save_figure(fig, output_dir / "vit_rank_proxy_differences_by_layer")
    return paths, image_summary


def plot_vit_lsd_distances(
    image_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    plot_data = metrics.assign(
        layer=(metrics["layer_idx"].astype(int) + 1).astype(str)
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, distance in zip(axes, LSD_DISTANCE_METRICS, strict=True):
        sns.boxplot(
            data=plot_data,
            x="layer",
            y=distance,
            color=LSD_DISTANCE_COLORS[distance],
            width=0.62,
            whis=1.5,
            showfliers=False,
            ax=ax,
        )
        sns.stripplot(
            data=plot_data,
            x="layer",
            y=distance,
            color="0.20",
            size=1.6,
            alpha=0.24,
            jitter=0.20,
            ax=ax,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Distance")
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(LSD_DISTANCE_LABELS[distance])
        if distance in ("total_variation_distance", "kolmogorov_distance"):
            ax.set_ylim(-0.02, 1.02)

    fig.suptitle(
        "ViT-B/16 real vs. spectral-canonical per-head LSD distances",
        y=0.99,
        fontsize=14,
    )

    num_images = int(image_summary["sample_idx"].nunique())
    num_heads = int(metrics["head_idx"].nunique())
    fig.text(
        0.5,
        0.01,
        f"Boxes and points show paired per-head distances ({num_images} images × "
        f"{num_heads} heads per layer); heads within an image are descriptive repeats.\n"
        rf"TV uses {TV_NUM_BINS} fixed log10 bins on [{TV_LOG10_MIN:g}, {TV_LOG10_MAX:g}]; "
        r"Kolmogorov and $W_1$ use the raw empirical spectra.",
        ha="center",
        fontsize=8,
        linespacing=1.35,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    return save_figure(fig, output_dir / "vit_lsd_distances_by_layer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare real and spectral-canonical attention rank proxies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", choices=("all", "gpt2", "vit"), default="all")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help="Reuse existing head-metric CSV files and only regenerate summaries/plots.",
    )
    parser.add_argument("--max-layers", type=int, default=None, help="Smoke-test layer limit.")
    parser.add_argument("--max-heads", type=int, default=None, help="Smoke-test head limit.")
    parser.add_argument(
        "--gpt2-seq-lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 384, 512, 768, 1024],
        help="Nested prefix lengths evaluated for every selected WikiText document.",
    )
    parser.add_argument(
        "--num-text-samples",
        type=int,
        default=12,
        help="Deterministic common WikiText document cohort size; 0 uses all eligible documents.",
    )
    parser.add_argument("--wikitext-arrow", type=Path, default=None)
    parser.add_argument("--imagenette-root", type=Path, default=DEFAULT_IMAGENETTE_ROOT)
    parser.add_argument(
        "--num-image-samples",
        type=int,
        default=20,
        help="Class-balanced Imagenette validation subset size; 0 uses all images.",
    )
    args = parser.parse_args()

    if not args.gpt2_seq_lengths or any(length <= 0 for length in args.gpt2_seq_lengths):
        parser.error("--gpt2-seq-lengths must contain positive integers.")
    if args.num_text_samples < 0:
        parser.error("--num-text-samples must be nonnegative.")
    if args.num_image_samples < 0:
        parser.error("--num-image-samples must be nonnegative.")
    if args.max_layers is not None and args.max_layers <= 0:
        parser.error("--max-layers must be positive.")
    if args.max_heads is not None and args.max_heads <= 0:
        parser.error("--max-heads must be positive.")
    return args


def validate_reused_metrics(
    metrics: pd.DataFrame,
    model_name: str,
    args: argparse.Namespace,
    previous_metadata: dict[str, Any],
) -> None:
    """Reject stale, partial, or numerically invalid CSVs before plotting."""

    if int(previous_metadata.get("seed", -1)) != args.seed:
        raise ValueError("Reused metrics were generated with a different random seed.")

    common_columns = {
        "sample_idx",
        "seq_len",
        "layer_idx",
        "head_idx",
        "real_stable_rank",
        "canonical_stable_rank",
        "delta_stable_rank",
        "real_erank_1",
        "canonical_erank_1",
        "delta_erank_1",
        "real_erank_2",
        "canonical_erank_2",
        "delta_erank_2",
        *ALL_LSD_DISTANCE_METRICS,
        "attention_reconstruction_max_abs_error",
    }
    missing_common = sorted(common_columns.difference(metrics.columns))
    if missing_common:
        raise ValueError(f"Reused {model_name} CSV is missing columns: {missing_common}")
    if model_name == "gpt2":
        required = common_columns | {"row_start", "row_end"}
        key_columns = ["sample_idx", "seq_len", "layer_idx", "head_idx"]
        model_metadata = previous_metadata.get("gpt2", {})
        expected_lengths = sorted({int(x) for x in args.gpt2_seq_lengths})
        actual_lengths = sorted(int(x) for x in metrics["seq_len"].unique())
        if actual_lengths != expected_lengths:
            raise ValueError(
                f"Reused GPT-2 CSV has lengths {actual_lengths}, expected {expected_lengths}."
            )
        selected_documents = int(model_metadata.get("selected_documents", -1))
        eligible_documents = int(model_metadata.get("eligible_documents", -2))
        if args.num_text_samples > 0:
            expected_documents = min(args.num_text_samples, eligible_documents)
            if selected_documents != expected_documents:
                raise ValueError("Reused GPT-2 CSV was generated with a different document count.")
        elif selected_documents != eligible_documents:
            raise ValueError("--num-text-samples 0 requires a CSV containing all eligible documents.")
        if args.max_layers is None:
            if int(model_metadata.get("num_layers", -1)) != int(model_metadata.get("num_layers_available", -2)):
                raise ValueError("Reused GPT-2 CSV is partial; specify --max-layers to reuse it explicitly.")
        elif int(model_metadata.get("num_layers", -1)) != min(
            args.max_layers, int(model_metadata.get("num_layers_available", -2))
        ):
            raise ValueError("Reused GPT-2 CSV has a different layer limit.")
        if args.max_heads is None:
            if int(model_metadata.get("num_heads", -1)) != int(model_metadata.get("num_heads_available", -2)):
                raise ValueError("Reused GPT-2 CSV is partial; specify --max-heads to reuse it explicitly.")
        elif int(model_metadata.get("num_heads", -1)) != min(
            args.max_heads, int(model_metadata.get("num_heads_available", -2))
        ):
            raise ValueError("Reused GPT-2 CSV has a different head limit.")
    elif model_name == "vit":
        required = common_columns | {"dataset_idx", "sample_id", "class_synset"}
        key_columns = ["sample_idx", "layer_idx", "head_idx"]
        model_metadata = previous_metadata.get("vit", {})
        selected_images = int(model_metadata.get("selected_images", -1))
        validation_images = int(model_metadata.get("validation_images", -2))
        if args.num_image_samples > 0:
            expected_images = min(args.num_image_samples, validation_images)
            if selected_images != expected_images:
                raise ValueError("Reused ViT CSV was generated with a different image count.")
        elif selected_images != validation_images:
            raise ValueError("--num-image-samples 0 requires a CSV containing all validation images.")
        if args.max_layers is None:
            if int(model_metadata.get("num_layers", -1)) != int(model_metadata.get("num_layers_available", -2)):
                raise ValueError("Reused ViT CSV is partial; specify --max-layers to reuse it explicitly.")
        elif int(model_metadata.get("num_layers", -1)) != min(
            args.max_layers, int(model_metadata.get("num_layers_available", -2))
        ):
            raise ValueError("Reused ViT CSV has a different layer limit.")
        if args.max_heads is None:
            if int(model_metadata.get("num_heads", -1)) != int(model_metadata.get("num_heads_available", -2)):
                raise ValueError("Reused ViT CSV is partial; specify --max-heads to reuse it explicitly.")
        elif int(model_metadata.get("num_heads", -1)) != min(
            args.max_heads, int(model_metadata.get("num_heads_available", -2))
        ):
            raise ValueError("Reused ViT CSV has a different head limit.")
    else:
        raise ValueError(model_name)

    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Reused {model_name} CSV is missing columns: {missing}")
    numeric = metrics.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"Reused {model_name} CSV contains non-finite values.")
    if metrics.duplicated(key_columns).any():
        raise ValueError(f"Reused {model_name} CSV contains duplicate sample/layer/head keys.")
    expected_rows = (
        int(metrics["sample_idx"].nunique())
        * int(metrics["seq_len"].nunique())
        * int(metrics["layer_idx"].nunique())
        * int(metrics["head_idx"].nunique())
    )
    if len(metrics) != expected_rows:
        raise ValueError(f"Reused {model_name} CSV does not contain full Cartesian coverage.")
    for metric in RANK_METRICS:
        for prefix in ("real", "canonical"):
            values = metrics[f"{prefix}_{metric}"]
            if (values < -1e-8).any() or (values > metrics["seq_len"] + 1e-5).any():
                raise ValueError(f"Reused {model_name} CSV has out-of-range {prefix}_{metric} values.")
        delta_error = (
            metrics[f"delta_{metric}"]
            - metrics[f"real_{metric}"]
            + metrics[f"canonical_{metric}"]
        ).abs()
        if float(delta_error.max()) > 1e-6:
            raise ValueError(f"Reused {model_name} CSV has inconsistent {metric} deltas.")
    if (metrics["total_variation_distance"].between(0, 1) == False).any():
        raise ValueError(f"Reused {model_name} CSV has invalid total-variation distances.")
    if (metrics["kolmogorov_distance"].between(0, 1) == False).any():
        raise ValueError(f"Reused {model_name} CSV has invalid Kolmogorov distances.")
    if (metrics["wasserstein_distance"] < -1e-8).any():
        raise ValueError(f"Reused {model_name} CSV has negative Wasserstein distances.")
    if (metrics["wasserstein_2_distance"] < -1e-8).any():
        raise ValueError(f"Reused {model_name} CSV has negative Wasserstein-2 distances.")
    if (metrics["wasserstein_distance"] > metrics["wasserstein_2_distance"] + 1e-8).any():
        raise ValueError(f"Reused {model_name} CSV violates W1 <= W2.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_kl_unit_checks()
    configure_plot_style()
    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    gpt2_csv = output_dir / "gpt2_head_rank_proxy_metrics.csv"
    vit_csv = output_dir / "vit_head_rank_proxy_metrics.csv"
    metadata_path = output_dir / "experiment_metadata.json"
    previous_metadata: dict[str, Any] = {}
    if args.reuse_data:
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Cannot validate reused CSVs without the manifest: {metadata_path}"
            )
        previous_metadata = json.loads(metadata_path.read_text())
    metadata: dict[str, Any] = {
        "seed": args.seed,
        "device": str(device),
        "configuration": {
            "models": args.models,
            "gpt2_sequence_lengths": sorted({int(x) for x in args.gpt2_seq_lengths}),
            "num_text_samples": args.num_text_samples,
            "num_image_samples": args.num_image_samples,
            "max_layers": args.max_layers,
            "max_heads": args.max_heads,
        },
        "canonical_diagonal_order": "descending score singular values",
        "difference_convention": "real minus canonical",
        "lsd_distance_definitions": {
            "total_variation_distance": (
                f"TV of common {TV_NUM_BINS}-bin log10 histograms on "
                f"[{TV_LOG10_MIN:g}, {TV_LOG10_MAX:g}] with values clipped to the range"
            ),
            "kolmogorov_distance": "supremum absolute gap between raw empirical singular-value CDFs",
            "wasserstein_distance": (
                "mean absolute difference between paired descending raw attention singular values"
            ),
            "wasserstein_2_distance": (
                "root mean squared difference between paired descending raw attention singular values"
            ),
        },
        "kl_divergence_definitions": {
            "empirical_measure_note": (
                "Exact KL between floating-point empirical spectral measures is generally "
                "infinite; reported values are finite fixed-log-histogram divergences."
            ),
            "log10_range": [KL_LOG10_MIN, KL_LOG10_MAX],
            "primary_num_bins": KL_PRIMARY_NUM_BINS,
            "primary_uniform_contamination": KL_PRIMARY_UNIFORM_CONTAMINATION,
            "sensitivity_num_bins": list(KL_SENSITIVITY_NUM_BINS),
            "sensitivity_uniform_contaminations": list(
                KL_SENSITIVITY_UNIFORM_CONTAMINATIONS
            ),
            "smoothing": (
                "p_tilde = (1 - eta) * p + eta / num_bins, independently for "
                "the real and canonical normalized histograms"
            ),
            "units": "nats (natural logarithm)",
            "directional_columns": {
                "kl_real_to_canonical": "KL(real || canonical)",
                "kl_canonical_to_real": "KL(canonical || real)",
            },
            "symmetric_column": (
                "symmetric_kl_divergence = 0.5 * "
                "(kl_real_to_canonical + kl_canonical_to_real)"
            ),
            "clipping_fraction_columns": list(KL_CLIPPING_METRICS),
            "sensitivity_columns": {
                column: {
                    "num_bins": num_bins,
                    "uniform_contamination": uniform_contamination,
                }
                for column, num_bins, uniform_contamination in KL_SENSITIVITY_SPECS
            },
            "reuse_note": (
                "KL columns are additive in newly collected CSVs and are not required "
                "when validating legacy reused CSVs."
            ),
        },
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "transformers_version": transformers.__version__,
    }
    if args.reuse_data:
        for model_key in ("gpt2", "vit"):
            if model_key in previous_metadata:
                metadata[model_key] = previous_metadata[model_key]
    generated_paths: list[Path] = []

    if args.models in ("all", "gpt2"):
        if args.reuse_data:
            if not gpt2_csv.exists():
                raise FileNotFoundError(f"Cannot reuse missing file: {gpt2_csv}")
            gpt2_metrics = pd.read_csv(gpt2_csv)
            validate_reused_metrics(gpt2_metrics, "gpt2", args, previous_metadata)
            gpt2_metadata = {
                **previous_metadata.get("gpt2", {}),
                "reused_csv": str(gpt2_csv),
                "head_rows": len(gpt2_metrics),
            }
        else:
            print("Collecting GPT-2 / WikiText rank proxies...")
            gpt2_metrics, gpt2_metadata = collect_gpt2_metrics(
                device=device,
                output_csv=gpt2_csv,
                sequence_lengths=args.gpt2_seq_lengths,
                num_text_samples=args.num_text_samples,
                seed=args.seed,
                arrow_path=args.wikitext_arrow,
                max_layers=args.max_layers,
                max_heads=args.max_heads,
            )
        metadata["gpt2"] = gpt2_metadata
        generated_paths.append(gpt2_csv)

        gpt2_sequence_summaries: list[pd.DataFrame] = []
        gpt2_document_scope_medians: list[pd.DataFrame] = []
        for scope in ("all", "first", "last"):
            figure_paths, summary = plot_gpt2_rank_proxy_scope(
                metrics=gpt2_metrics,
                scope=scope,
                output_dir=output_dir,
            )
            generated_paths.extend(figure_paths)
            gpt2_sequence_summaries.append(summary)
            gpt2_document_scope_medians.append(
                sample_level_gpt2_summary(gpt2_metrics, scope).assign(layer_scope=scope)
            )
        lsd_distance_paths, lsd_distance_summary = plot_gpt2_lsd_distances(
            metrics=gpt2_metrics,
            output_dir=output_dir,
        )
        generated_paths.extend(lsd_distance_paths)

        gpt2_binned_path = output_dir / "gpt2_rank_proxy_sequence_summary.csv"
        gpt2_lsd_distance_path = output_dir / "gpt2_lsd_distance_sequence_summary.csv"
        gpt2_document_median_path = output_dir / "gpt2_document_scope_medians.csv"
        pd.concat(gpt2_sequence_summaries, ignore_index=True).to_csv(gpt2_binned_path, index=False)
        lsd_distance_summary.to_csv(gpt2_lsd_distance_path, index=False)
        pd.concat(gpt2_document_scope_medians, ignore_index=True).to_csv(
            gpt2_document_median_path,
            index=False,
        )
        generated_paths.extend(
            [gpt2_binned_path, gpt2_lsd_distance_path, gpt2_document_median_path]
        )

    if args.models in ("all", "vit"):
        if args.reuse_data:
            if not vit_csv.exists():
                raise FileNotFoundError(f"Cannot reuse missing file: {vit_csv}")
            vit_metrics = pd.read_csv(vit_csv)
            validate_reused_metrics(vit_metrics, "vit", args, previous_metadata)
            vit_metadata = {
                **previous_metadata.get("vit", {}),
                "reused_csv": str(vit_csv),
                "head_rows": len(vit_metrics),
            }
        else:
            print("Collecting ViT / Imagenette rank proxies...")
            vit_metrics, vit_metadata = collect_vit_metrics(
                device=device,
                output_csv=vit_csv,
                imagenette_root=args.imagenette_root,
                num_image_samples=args.num_image_samples,
                seed=args.seed,
                max_layers=args.max_layers,
                max_heads=args.max_heads,
            )
        metadata["vit"] = vit_metadata
        generated_paths.append(vit_csv)

        vit_figure_paths, image_summary = plot_vit_rank_proxy_differences(
            metrics=vit_metrics,
            output_dir=output_dir,
        )
        generated_paths.extend(vit_figure_paths)
        generated_paths.extend(plot_vit_lsd_distances(image_summary, vit_metrics, output_dir))
        vit_summary_path = output_dir / "vit_image_layer_summary.csv"
        image_summary.to_csv(vit_summary_path, index=False)
        generated_paths.append(vit_summary_path)

    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    generated_paths.append(metadata_path)

    print("Generated:")
    for path in generated_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
