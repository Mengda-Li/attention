"""Plot per-layer score-matrix singular-value histograms.

For every attention head, this script reconstructs the trained-model score
matrix

    S_h = Q_h K_h^T / sqrt(d_h)

and computes its nonzero singular values without forming the full token-by-token
matrix.  A compact QR factorization reduces the SVD to a d_h-by-d_h core, so
the decomposition is exact up to floating-point error even for a 1024-token
GPT-2 input.

The layer panels pool the independently computed per-head singular values only
for visualization; no head-averaged score matrix is ever decomposed.  Because
rank(S_h) <= d_h, the ell-d_h structural zeros per head are reported separately
instead of allowing them to swamp the positive-value histogram.

The default run produces three 3x4 figures for GPT-2 on a 1024-token prefix of
the longest WikiText-103 validation document and for SWAG ViT-B/16 on a
deterministic Imagenette validation image:

1. a pure log-x histogram with no finite spike/bulk annotations;
2. a log-x histogram whose high-value tail is colored after a descriptive,
   layer-specific empirical eigengap; and
3. a pure linear-x histogram.

The empirical split is deliberately not identified with the paper's asymptotic
logarithmic separation.  For each layer, it is obtained from the median ordered
per-head spectrum: among indices 1,...,floor(d_h/2), select the largest adjacent
ratio m_k/m_{k+1}, then place the threshold at the geometric midpoint
sqrt(m_k m_{k+1}).  Its only purpose is to make the visible finite-sample
high-value group easier to read.  PNG/SVG figures, raw per-head singular values,
summaries, cached arrays, and metadata are saved under ``rebuttal``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import datasets as vision_datasets
from torchvision.models import ViT_B_16_Weights
from transformers import AutoTokenizer, GPT2Model

from plot_real_vs_canonical_attention_lsd import (
    resolve_device,
    select_longest_wikitext_document,
    set_seed,
)


SEED = 233
DEFAULT_OUTPUT_DIR = Path("rebuttal") / "score_bulk_spike"
DEFAULT_DATA_DIR = Path("rebuttal") / "score_svd_spike_bulk_data"
DEFAULT_IMAGENETTE_ROOT = Path.home() / "dataset" / "imagenette2"
DEFAULT_GPT2_SEQ_LEN = 1024
DEFAULT_VIT_SAMPLE_IDX = 1
DEFAULT_NUM_BINS = 42

HISTOGRAM_COLOR = "#3F4854"
SPIKE_COLOR = "#E69F00"
THRESHOLD_COLOR = "#262626"


@dataclass(frozen=True)
class ScoreSpectra:
    slug: str
    display_name: str
    dataset_name: str
    sample_label: str
    seq_len: int
    head_dim: int
    singular_values: np.ndarray
    metadata: dict[str, Any]

    @property
    def num_layers(self) -> int:
        return int(self.singular_values.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self.singular_values.shape[1])


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.55,
        }
    )


def make_capture_input_hook(layer_idx: int, storage: dict[int, torch.Tensor]):
    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        _ = module
        storage[layer_idx] = inputs[0].detach()

    return hook


def make_capture_output_hook(layer_idx: int, storage: dict[int, torch.Tensor]):
    def hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        _ = module
        _ = inputs
        storage[layer_idx] = output.detach()

    return hook


@torch.inference_mode()
def nonzero_score_singular_values(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Return the exact nonzero singular values of ``scale * Q K^T`` per head.

    ``query`` and ``key`` have shape ``(heads, seq_len, head_dim)``.  If
    ``Q = U_q R_q`` and ``K = U_k R_k`` are reduced QR factorizations, then the
    nonzero singular values of ``Q K^T`` equal those of ``R_q R_k^T``.
    Sensitive factorizations are intentionally performed in float64 on CPU.
    """

    query_cpu = query.detach().to(device="cpu", dtype=torch.float64)
    key_cpu = key.detach().to(device="cpu", dtype=torch.float64)
    _, query_r = torch.linalg.qr(query_cpu, mode="reduced")
    _, key_r = torch.linalg.qr(key_cpu, mode="reduced")
    core = torch.matmul(query_r, key_r.transpose(-1, -2)) * float(scale)
    singular_values = torch.linalg.svdvals(core)
    if not torch.isfinite(singular_values).all():
        raise RuntimeError("Score SVD produced non-finite singular values.")
    return torch.clamp(singular_values, min=0.0)


def validate_spectra(spectra: ScoreSpectra) -> None:
    values = spectra.singular_values
    expected_shape = (spectra.num_layers, spectra.num_heads, spectra.head_dim)
    if values.shape != expected_shape:
        raise AssertionError(f"Expected spectra shape {expected_shape}, got {values.shape}.")
    if spectra.seq_len < spectra.head_dim:
        raise AssertionError("This experiment expects seq_len >= head_dim.")
    if not np.isfinite(values).all():
        raise AssertionError("Score spectra contain non-finite values.")
    if np.any(values < 0.0):
        raise AssertionError("Score spectra contain negative singular values.")
    descending_violation = np.max(np.diff(values, axis=-1), initial=0.0)
    tolerance = max(float(np.max(values)), 1.0) * 1e-10
    if descending_violation > tolerance:
        raise AssertionError(
            f"Score singular values are not descending; max violation={descending_violation:.3e}."
        )


@torch.inference_mode()
def collect_gpt2_score_spectra(
    device: torch.device,
    seq_len: int,
    wikitext_arrow: Path | None,
    max_layers: int | None,
    max_heads: int | None,
) -> ScoreSpectra:
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    model = GPT2Model.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()

    context_length = int(model.config.n_positions)
    if seq_len > context_length:
        raise ValueError(
            f"GPT-2 supports at most {context_length} positions, but {seq_len} were requested."
        )

    input_ids, sample_metadata = select_longest_wikitext_document(
        tokenizer=tokenizer,
        seq_len=seq_len,
        arrow_path=wikitext_arrow,
    )

    n_layers_available = len(model.h)
    n_layers = n_layers_available if max_layers is None else min(max_layers, n_layers_available)
    n_heads_available = int(model.config.num_attention_heads)
    n_heads = n_heads_available if max_heads is None else min(max_heads, n_heads_available)
    head_dim = int(model.h[0].attn.head_dim)

    captured_inputs: dict[int, torch.Tensor] = {}
    hooks = [
        model.h[layer_idx].attn.register_forward_pre_hook(
            make_capture_input_hook(layer_idx, captured_inputs)
        )
        for layer_idx in range(n_layers)
    ]

    model.to(device)
    tensor_input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    try:
        model(input_ids=tensor_input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    singular_values = np.empty((n_layers, n_heads, head_dim), dtype=np.float64)
    for layer_idx in range(n_layers):
        if layer_idx not in captured_inputs:
            raise RuntimeError(f"Missing captured GPT-2 attention input for layer {layer_idx}.")

        hidden_states = captured_inputs[layer_idx]
        attention_module = model.h[layer_idx].attn
        qkv = attention_module.c_attn(hidden_states)
        query, key, _ = qkv.split(attention_module.split_size, dim=2)
        query = query.view(1, seq_len, n_heads_available, head_dim).transpose(1, 2)[
            0, :n_heads
        ]
        key = key.view(1, seq_len, n_heads_available, head_dim).transpose(1, 2)[
            0, :n_heads
        ]
        layer_singular_values = nonzero_score_singular_values(
            query=query,
            key=key,
            scale=float(attention_module.scaling),
        )
        singular_values[layer_idx] = layer_singular_values.numpy()
        print(
            f"GPT-2 layer {layer_idx + 1}/{n_layers}: "
            f"top score singular value range "
            f"[{singular_values[layer_idx, :, 0].min():.3g}, "
            f"{singular_values[layer_idx, :, 0].max():.3g}]"
        )

    sample_label = (
        f"WikiText rows {sample_metadata['row_start']}--{sample_metadata['row_end']}; "
        f"first {seq_len} tokens"
    )
    metadata = {
        **sample_metadata,
        "model": "gpt2",
        "model_display_name": "GPT-2",
        "dataset": "WikiText-103 validation",
        "causal": True,
        "score_definition": "Q K^T / sqrt(head_dim), before the causal mask",
        "context_length": context_length,
        "num_layers": n_layers,
        "num_layers_available": n_layers_available,
        "num_heads": n_heads,
        "num_heads_available": n_heads_available,
        "head_dim": head_dim,
        "structural_zeros_per_head": seq_len - head_dim,
        "device": str(device),
    }

    del model, tensor_input_ids, captured_inputs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    spectra = ScoreSpectra(
        slug="gpt2_score_svd_spike_bulk_by_layer",
        display_name="GPT-2",
        dataset_name="WikiText-103 validation",
        sample_label=sample_label,
        seq_len=seq_len,
        head_dim=head_dim,
        singular_values=singular_values,
        metadata=metadata,
    )
    validate_spectra(spectra)
    return spectra


@torch.inference_mode()
def collect_vit_score_spectra(
    device: torch.device,
    imagenette_root: Path,
    sample_idx: int,
    max_layers: int | None,
    max_heads: int | None,
) -> ScoreSpectra:
    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    validation_root = imagenette_root.expanduser() / "val"
    if not validation_root.is_dir():
        raise FileNotFoundError(f"Imagenette validation directory not found: {validation_root}")

    imagenette = vision_datasets.ImageFolder(
        root=validation_root,
        transform=weights.transforms(),
    )
    if not 0 <= sample_idx < len(imagenette):
        raise IndexError(
            f"Imagenette sample index {sample_idx} is outside [0, {len(imagenette) - 1}]."
        )

    sample_image, sample_label_idx = imagenette[sample_idx]
    sample_path = Path(imagenette.samples[sample_idx][0])
    class_synset = imagenette.classes[sample_label_idx]

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
            make_capture_output_hook(layer_idx, captured_ln1)
        )
        for layer_idx in range(n_layers)
    ]

    image_batch = sample_image.unsqueeze(0).to(device)
    try:
        model(image_batch)
    finally:
        for hook in hooks:
            hook.remove()

    if 0 not in captured_ln1:
        raise RuntimeError("Failed to capture the ViT LayerNorm output.")
    seq_len = int(captured_ln1[0].shape[1])
    singular_values = np.empty((n_layers, n_heads, head_dim), dtype=np.float64)

    for layer_idx in range(n_layers):
        if layer_idx not in captured_ln1:
            raise RuntimeError(f"Missing captured ViT LayerNorm output for layer {layer_idx}.")

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
        layer_singular_values = nonzero_score_singular_values(
            query=query,
            key=key,
            scale=1.0 / math.sqrt(float(head_dim)),
        )
        singular_values[layer_idx] = layer_singular_values.numpy()
        print(
            f"ViT layer {layer_idx + 1}/{n_layers}: "
            f"top score singular value range "
            f"[{singular_values[layer_idx, :, 0].min():.3g}, "
            f"{singular_values[layer_idx, :, 0].max():.3g}]"
        )

    sample_label = f"{sample_path.name} ({class_synset}); {seq_len} tokens"
    metadata = {
        "model": "vit_b_16",
        "model_display_name": "ViT-B/16",
        "weights": weights.name,
        "dataset": "Imagenette validation",
        "imagenette_root": str(imagenette_root.expanduser()),
        "sample_index": sample_idx,
        "sample_path": str(sample_path),
        "class_synset": class_synset,
        "causal": False,
        "score_definition": "Q K^T / sqrt(head_dim)",
        "seq_len": seq_len,
        "num_layers": n_layers,
        "num_layers_available": n_layers_available,
        "num_heads": n_heads,
        "num_heads_available": n_heads_available,
        "head_dim": head_dim,
        "structural_zeros_per_head": seq_len - head_dim,
        "device": str(device),
    }

    del model, image_batch, captured_ln1
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    spectra = ScoreSpectra(
        slug="vit_score_svd_spike_bulk_by_layer",
        display_name="ViT-B/16",
        dataset_name="Imagenette validation",
        sample_label=sample_label,
        seq_len=seq_len,
        head_dim=head_dim,
        singular_values=singular_values,
        metadata=metadata,
    )
    validate_spectra(spectra)
    return spectra


def spectra_to_frames(spectra: ScoreSpectra) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zero_count = spectra.seq_len - spectra.head_dim
    value_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []

    for layer_idx in range(spectra.num_layers):
        gap_rank, empirical_threshold, gap_ratio = empirical_gap_split(
            spectra.singular_values[layer_idx]
        )
        layer_spike_counts: list[int] = []
        for head_idx in range(spectra.num_heads):
            values = spectra.singular_values[layer_idx, head_idx]
            spike_mask = values >= empirical_threshold
            spike_count = int(np.count_nonzero(spike_mask))
            bulk_positive_count = int(spectra.head_dim - spike_count)
            layer_spike_counts.append(spike_count)

            for singular_idx, singular_value in enumerate(values):
                regime = (
                    "high-value group"
                    if singular_value >= empirical_threshold
                    else "bulk-side"
                )
                value_rows.append(
                    {
                        "model": spectra.display_name,
                        "dataset": spectra.dataset_name,
                        "sample": spectra.sample_label,
                        "seq_len": spectra.seq_len,
                        "layer_idx": layer_idx,
                        "head_idx": head_idx,
                        "singular_value_idx": singular_idx,
                        "singular_value": float(singular_value),
                        "empirical_group": regime,
                        "empirical_gap_rank": gap_rank,
                        "empirical_threshold": empirical_threshold,
                        "empirical_gap_ratio": gap_ratio,
                    }
                )

            head_rows.append(
                {
                    "model": spectra.display_name,
                    "dataset": spectra.dataset_name,
                    "sample": spectra.sample_label,
                    "seq_len": spectra.seq_len,
                    "layer_idx": layer_idx,
                    "head_idx": head_idx,
                    "head_dim": spectra.head_dim,
                    "structural_zero_count": zero_count,
                    "bulk_positive_count": bulk_positive_count,
                    "bulk_total_count": zero_count + bulk_positive_count,
                    "spike_count": spike_count,
                    "has_spike": spike_count > 0,
                    "top_singular_value": float(values[0]),
                    "empirical_gap_rank": gap_rank,
                    "empirical_threshold": empirical_threshold,
                    "empirical_gap_ratio": gap_ratio,
                }
            )

        spike_counts = np.asarray(layer_spike_counts, dtype=float)
        layer_rows.append(
            {
                "model": spectra.display_name,
                "dataset": spectra.dataset_name,
                "sample": spectra.sample_label,
                "seq_len": spectra.seq_len,
                "layer_idx": int(layer_idx),
                "num_heads": spectra.num_heads,
                "heads_with_spike": int(np.count_nonzero(spike_counts > 0)),
                "median_spikes_per_head": float(np.median(spike_counts)),
                "q1_spikes_per_head": float(np.quantile(spike_counts, 0.25)),
                "q3_spikes_per_head": float(np.quantile(spike_counts, 0.75)),
                "min_spikes_per_head": int(np.min(spike_counts)),
                "max_spikes_per_head": int(np.max(spike_counts)),
                "spike_fraction_of_nonzero_values": float(
                    np.sum(spike_counts) / (spectra.num_heads * spectra.head_dim)
                ),
                "spike_fraction_of_full_spectrum": float(
                    np.sum(spike_counts) / (spectra.num_heads * spectra.seq_len)
                ),
                "structural_zero_fraction": float(zero_count / spectra.seq_len),
                "empirical_gap_rank": gap_rank,
                "empirical_threshold": empirical_threshold,
                "empirical_gap_ratio": gap_ratio,
            }
        )
    return pd.DataFrame(value_rows), pd.DataFrame(head_rows), pd.DataFrame(layer_rows)


def empirical_gap_split(layer_values: np.ndarray) -> tuple[int, float, float]:
    """Return a descriptive upper-spectrum eigengap split for one layer.

    The ordered singular values are first median-aggregated across heads.  We
    search the upper half of that median scree curve for the largest adjacent
    multiplicative gap.  The returned rank is one-based, and the threshold is
    the geometric midpoint across the selected gap.
    """

    if layer_values.ndim != 2:
        raise ValueError(
            f"Expected a heads-by-head_dim array, got shape {layer_values.shape}."
        )
    positive = layer_values[np.isfinite(layer_values) & (layer_values > 0.0)]
    if positive.size == 0:
        raise ValueError("Cannot estimate an empirical eigengap without positive values.")

    numerical_floor = float(np.min(positive)) * 1e-12
    median_spectrum = np.median(
        np.maximum(layer_values, numerical_floor),
        axis=0,
    )
    max_candidate_rank = max(1, layer_values.shape[1] // 2)
    adjacent_ratios = (
        median_spectrum[:max_candidate_rank]
        / median_spectrum[1 : max_candidate_rank + 1]
    )
    gap_rank = int(np.argmax(adjacent_ratios)) + 1
    gap_ratio = float(adjacent_ratios[gap_rank - 1])
    threshold = float(
        math.sqrt(median_spectrum[gap_rank - 1] * median_spectrum[gap_rank])
    )
    return gap_rank, threshold, gap_ratio


def logarithmic_histogram_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError("Cannot make a logarithmic histogram without positive values.")
    lower_exponent = math.floor(math.log10(float(np.min(positive))))
    upper_exponent = math.ceil(math.log10(float(np.max(positive))))
    if lower_exponent == upper_exponent:
        upper_exponent += 1
    return np.geomspace(10.0**lower_exponent, 10.0**upper_exponent, num_bins + 1)


def plot_layer_histograms(
    spectra: ScoreSpectra,
    output_dir: Path,
    num_bins: int,
    variant: str,
) -> list[Path]:
    supported_variants = {"log_pure", "log_empirical_gap", "linear_pure"}
    if variant not in supported_variants:
        raise ValueError(f"Unsupported histogram variant: {variant}")

    x_scale = "linear" if variant == "linear_pure" else "log"
    empirical_splits = [
        empirical_gap_split(spectra.singular_values[layer_idx])
        for layer_idx in range(spectra.num_layers)
    ]
    histogram_edges_by_layer: list[np.ndarray] = []
    x_limits_by_layer: list[tuple[float, float]] = []
    if x_scale == "log":
        shared_edges = logarithmic_histogram_edges(
            spectra.singular_values.reshape(-1),
            num_bins=num_bins,
        )
        shared_limits = (float(shared_edges[0]), float(shared_edges[-1]))
        for layer_idx in range(spectra.num_layers):
            if variant == "log_empirical_gap":
                threshold = empirical_splits[layer_idx][1]
                layer_edges = np.unique(np.append(shared_edges, threshold))
                histogram_edges_by_layer.append(np.sort(layer_edges))
            else:
                histogram_edges_by_layer.append(shared_edges)
        x_limits_by_layer = [shared_limits] * spectra.num_layers
    else:
        for layer_idx in range(spectra.num_layers):
            values = spectra.singular_values[layer_idx].reshape(-1)
            layer_max = float(np.max(values)) * 1.025
            histogram_edges_by_layer.append(np.linspace(0.0, layer_max, num_bins + 1))
            x_limits_by_layer.append((0.0, layer_max))

    n_cols = 4
    n_rows = math.ceil(spectra.num_layers / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16.0, 10.2),
        sharex=x_scale == "log",
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    max_bin_count = 0.0
    for layer_idx in range(spectra.num_layers):
        counts, _ = np.histogram(
            spectra.singular_values[layer_idx].reshape(-1),
            bins=histogram_edges_by_layer[layer_idx],
        )
        max_bin_count = max(max_bin_count, float(np.max(counts)))

    for panel_idx, ax in enumerate(flat_axes):
        if panel_idx >= spectra.num_layers:
            ax.axis("off")
            continue

        layer_idx = panel_idx
        values = spectra.singular_values[layer_idx].reshape(-1)
        histogram_edges = histogram_edges_by_layer[layer_idx]
        x_min, x_max = x_limits_by_layer[layer_idx]
        if variant == "log_empirical_gap":
            _, threshold, _ = empirical_splits[layer_idx]
            ax.hist(
                [values[values < threshold], values[values >= threshold]],
                bins=histogram_edges,
                stacked=True,
                color=[HISTOGRAM_COLOR, SPIKE_COLOR],
                alpha=0.86,
                edgecolor="none",
                zorder=2,
            )
            ax.axvline(
                threshold,
                color=THRESHOLD_COLOR,
                linestyle=(0, (4, 2)),
                linewidth=1.15,
                zorder=3,
            )
        else:
            ax.hist(
                values,
                bins=histogram_edges,
                color=HISTOGRAM_COLOR,
                alpha=0.82,
                edgecolor="none",
                zorder=2,
            )

        if x_scale == "log":
            ax.set_xscale("log")
        else:
            ax.ticklabel_format(
                axis="x",
                style="sci",
                scilimits=(-2, 3),
                useMathText=True,
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, max_bin_count * 1.14)
        ax.set_title(f"Layer {layer_idx + 1}")

    for row_idx in range(n_rows):
        y_label = "Count per log bin" if x_scale == "log" else "Count per linear bin"
        axes[row_idx, 0].set_ylabel(y_label)
    for col_idx in range(n_cols):
        axes[-1, col_idx].set_xlabel(r"Nonzero singular value $\sigma_j(S_h)$")

    if variant == "log_empirical_gap":
        title_suffix = "log x-axis; empirical high-value split"
    else:
        title_suffix = f"{x_scale} x-axis"
    fig.suptitle(
        f"{spectra.display_name}: pre-softmax score singular values by layer "
        f"({title_suffix})",
        y=0.992,
        fontsize=13,
    )
    fig.text(
        0.5,
        0.954,
        spectra.sample_label,
        ha="center",
        va="top",
        fontsize=9,
    )
    if variant == "log_empirical_gap":
        handles = [
            Patch(
                facecolor=HISTOGRAM_COLOR,
                alpha=0.86,
                label="Below empirical gap",
            ),
            Patch(
                facecolor=SPIKE_COLOR,
                alpha=0.86,
                label="Above empirical gap (spike-side)",
            ),
            Line2D(
                [0],
                [0],
                color=THRESHOLD_COLOR,
                linestyle=(0, (4, 2)),
                linewidth=1.15,
                label="Layer-specific empirical eigengap",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=3,
            frameon=False,
        )

    zero_count = spectra.seq_len - spectra.head_dim
    zero_fraction = zero_count / spectra.seq_len
    if variant == "log_pure":
        footer = (
            f"Each panel pools {spectra.num_heads} heads only after independent per-head "
            f"SVDs ({spectra.head_dim} nonzero values/head). The {zero_count} structural "
            f"zeros/head ({zero_fraction:.1%}) are omitted from the log axis; no finite "
            "spike/bulk threshold is imposed."
        )
        layout_top = 0.92
    elif variant == "log_empirical_gap":
        footer = (
            f"Each panel pools {spectra.num_heads} heads only after independent per-head "
            f"SVDs ({spectra.head_dim} nonzero values/head); {zero_count} structural "
            "zeros/head are omitted. Orange values lie above the largest adjacent gap in "
            "the layer's median per-head scree curve (searched over its upper half). This "
            r"is a descriptive finite-sample split, not the asymptotic $\log \ell$ cutoff."
        )
        layout_top = 0.885
    else:
        footer = (
            f"Each panel pools {spectra.num_heads} heads after independent per-head SVDs "
            f"({spectra.head_dim} nonzero values/head). The {zero_count} structural zeros/head "
            f"are omitted; each panel shows its full layer-specific linear range."
        )
        layout_top = 0.92
    fig.text(
        0.5,
        0.008,
        footer,
        ha="center",
        va="bottom",
        fontsize=7.7,
    )
    fig.tight_layout(rect=(0.045, 0.055, 0.995, layout_top), h_pad=1.15, w_pad=0.9)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    variant_suffix = {
        "log_pure": "_log_pure",
        "log_empirical_gap": "_log_empirical_gap",
        "linear_pure": "_linear",
    }[variant]
    for suffix in (".png", ".svg"):
        path = output_dir / f"{spectra.slug}{variant_suffix}{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def save_spectra_outputs(
    spectra: ScoreSpectra,
    data_dir: Path,
) -> tuple[list[Path], pd.DataFrame]:
    values_frame, heads_frame, layer_frame = spectra_to_frames(spectra)
    data_dir.mkdir(parents=True, exist_ok=True)

    total_heads = int(heads_frame.shape[0])
    spike_count = int(heads_frame["spike_count"].sum())
    total_singular_values = total_heads * spectra.seq_len
    layer_splits = [
        {
            "layer_idx": int(row.layer_idx),
            "empirical_gap_rank": int(row.empirical_gap_rank),
            "empirical_threshold": float(row.empirical_threshold),
            "empirical_gap_ratio": float(row.empirical_gap_ratio),
        }
        for row in layer_frame.itertuples(index=False)
    ]
    legacy_annotation_keys = {
        "finite_length_regime_choice",
        "bulk_threshold",
        "spike_threshold",
        "finite_length_summary",
        "interpretation_warning",
    }
    base_metadata = {
        key: value
        for key, value in spectra.metadata.items()
        if key not in legacy_annotation_keys
    }
    metadata = {
        **base_metadata,
        "sample_label": spectra.sample_label,
        "seq_len": spectra.seq_len,
        "histogram_scope": (
            "nonzero score singular values pooled within layer after independent per-head SVD"
        ),
        "structural_zeros_in_histogram": False,
        "empirical_split": {
            "method": (
                "For each layer, median the ordered per-head spectra; search ranks "
                "1 through floor(head_dim/2) for the largest adjacent ratio; use the "
                "geometric midpoint across that gap as the descriptive threshold."
            ),
            "layer_splits": layer_splits,
            "total_heads": total_heads,
            "spike_fraction_of_nonzero_values": spike_count
            / (total_heads * spectra.head_dim),
            "spike_fraction_of_full_spectrum": spike_count / total_singular_values,
        },
        "interpretation_warning": (
            "The colored empirical split is a descriptive finite-sample eigengap and is not "
            "identified with the asymptotic logarithmic separation in Assumption 2. One "
            "sample also does not estimate population frequency."
        ),
    }

    npz_path = data_dir / f"{spectra.slug}.npz"
    values_path = data_dir / f"{spectra.slug}_values.csv"
    heads_path = data_dir / f"{spectra.slug}_head_summary.csv"
    layers_path = data_dir / f"{spectra.slug}_layer_summary.csv"
    metadata_path = data_dir / f"{spectra.slug}_metadata.json"
    np.savez_compressed(
        npz_path,
        singular_values=spectra.singular_values,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    values_frame.to_csv(values_path, index=False)
    heads_frame.to_csv(heads_path, index=False)
    layer_frame.to_csv(layers_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return [npz_path, values_path, heads_path, layers_path, metadata_path], layer_frame


def load_cached_spectra(path: Path) -> ScoreSpectra:
    with np.load(path, allow_pickle=False) as payload:
        singular_values = np.asarray(payload["singular_values"], dtype=np.float64)
        metadata = json.loads(str(payload["metadata_json"].item()))
    spectra = ScoreSpectra(
        slug=path.stem,
        display_name=str(metadata["model_display_name"]),
        dataset_name=str(metadata["dataset"]),
        sample_label=str(metadata["sample_label"]),
        seq_len=int(metadata["seq_len"]),
        head_dim=int(metadata["head_dim"]),
        singular_values=singular_values,
        metadata=metadata,
    )
    validate_spectra(spectra)
    return spectra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pure and empirically split per-layer score singular-value histograms."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", choices=("all", "gpt2", "vit"), default="all")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--gpt2-seq-len", type=int, default=DEFAULT_GPT2_SEQ_LEN)
    parser.add_argument("--wikitext-arrow", type=Path, default=None)
    parser.add_argument("--imagenette-root", type=Path, default=DEFAULT_IMAGENETTE_ROOT)
    parser.add_argument("--vit-sample-idx", type=int, default=DEFAULT_VIT_SAMPLE_IDX)
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--max-heads", type=int, default=None)
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help="Reuse cached NPZ arrays in --data-dir and redraw figures.",
    )
    args = parser.parse_args()
    if args.gpt2_seq_len <= 0:
        parser.error("--gpt2-seq-len must be positive.")
    if args.num_bins <= 0:
        parser.error("--num-bins must be positive.")
    if args.max_layers is not None and args.max_layers <= 0:
        parser.error("--max-layers must be positive when specified.")
    if args.max_heads is not None and args.max_heads <= 0:
        parser.error("--max-heads must be positive when specified.")
    return args


def print_summary(spectra: ScoreSpectra) -> None:
    for layer_idx in range(spectra.num_layers):
        gap_rank, threshold, gap_ratio = empirical_gap_split(
            spectra.singular_values[layer_idx]
        )
        high_value_count = int(
            np.count_nonzero(spectra.singular_values[layer_idx] >= threshold)
        )
        print(
            f"{spectra.display_name} layer {layer_idx + 1}: empirical gap after median "
            f"rank {gap_rank}, ratio={gap_ratio:.3g}, threshold={threshold:.3g}, "
            f"high-value count={high_value_count}/{spectra.num_heads * spectra.head_dim}."
        )


def main() -> None:
    args = parse_args()
    configure_plot_style()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    generated_paths: list[Path] = []
    requested_models = ("gpt2", "vit") if args.models == "all" else (args.models,)
    for model_name in requested_models:
        if model_name == "gpt2":
            cache_path = args.data_dir / "gpt2_score_svd_spike_bulk_by_layer.npz"
            if args.reuse_data:
                if not cache_path.exists():
                    raise FileNotFoundError(f"Cannot reuse missing cache: {cache_path}")
                spectra = load_cached_spectra(cache_path)
            else:
                spectra = collect_gpt2_score_spectra(
                    device=device,
                    seq_len=args.gpt2_seq_len,
                    wikitext_arrow=args.wikitext_arrow,
                    max_layers=args.max_layers,
                    max_heads=args.max_heads,
                )
        else:
            cache_path = args.data_dir / "vit_score_svd_spike_bulk_by_layer.npz"
            if args.reuse_data:
                if not cache_path.exists():
                    raise FileNotFoundError(f"Cannot reuse missing cache: {cache_path}")
                spectra = load_cached_spectra(cache_path)
            else:
                spectra = collect_vit_score_spectra(
                    device=device,
                    imagenette_root=args.imagenette_root,
                    sample_idx=args.vit_sample_idx,
                    max_layers=args.max_layers,
                    max_heads=args.max_heads,
                )

        if args.reuse_data:
            data_paths: list[Path] = []
        else:
            data_paths, _ = save_spectra_outputs(spectra, args.data_dir)
        figure_paths: list[Path] = []
        for variant in ("log_pure", "log_empirical_gap", "linear_pure"):
            figure_paths.extend(
                plot_layer_histograms(
                    spectra=spectra,
                    output_dir=args.output_dir,
                    num_bins=args.num_bins,
                    variant=variant,
                )
            )
        generated_paths.extend(data_paths)
        generated_paths.extend(figure_paths)
        print_summary(spectra)

    print("Generated:")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
