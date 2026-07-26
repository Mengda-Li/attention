"""Compare real and spectral-canonical attention spectra per head and layer.

For every attention head, this script constructs the trained-model score matrix

    S = Q K^T / sqrt(d_k),

its spectral canonical representative Sigma = diag(svdvals(S)), and the two
attention matrices

    A_real = softmax(S + M),
    A_canonical = softmax(Sigma + M),

where M is the causal mask for GPT-2 and is zero for ViT.  Singular values are
computed separately for every head.  Only then are the per-head spectra pooled
within a layer to estimate the empirical singular-value distribution (LSD).

The default run produces four figure contents (each saved as PNG and SVG):

1. GPT-2 layer-wise pooled LSD comparison.
2. GPT-2 layer-wise per-head ranked spectra.
3. ViT layer-wise pooled LSD comparison.
4. ViT layer-wise per-head ranked spectra.

It also saves the raw spectra and head/layer discrepancy metrics used in the
plots.  The defaults are fully offline and use the locally cached pretrained
weights and datasets described in local_advice.md.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
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
import torch
import torch.nn.functional as F
import torchvision
import transformers
from datasets import Dataset
from scipy.stats import ks_2samp
from torch import nn
from torchvision import datasets as vision_datasets
from torchvision.models import ViT_B_16_Weights
from transformers import AutoTokenizer, GPT2Model, PreTrainedTokenizerBase


SEED = 233
SPECTRAL_FLOOR = 1e-8
ATTENTION_RECONSTRUCTION_TOLERANCE = 5e-5
REAL_COLOR = "#0072B2"
CANONICAL_COLOR = "#D55E00"
DEFAULT_OUTPUT_DIR = Path("results/real_vs_canonical_attention_lsd")
DEFAULT_IMAGENETTE_ROOT = Path.home() / "dataset" / "imagenette2"
HF_DATASETS_CACHE = Path.home() / ".cache" / "huggingface" / "datasets"


@dataclass
class SpectralComparison:
    """Per-layer, per-head spectra for one model/sample pair."""

    slug: str
    display_name: str
    dataset_name: str
    sample_id: str
    seq_len: int
    real_singular_values: np.ndarray
    canonical_singular_values: np.ndarray
    score_singular_values: np.ndarray
    reconstruction_error: np.ndarray
    metadata: dict[str, Any]

    @property
    def num_layers(self) -> int:
        return int(self.real_singular_values.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self.real_singular_values.shape[1])


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_cached_wikitext_validation_arrow() -> Path:
    matches = sorted(
        HF_DATASETS_CACHE.glob(
            "Salesforce___wikitext/wikitext-103-raw-v1/*/*/wikitext-validation.arrow"
        )
    )
    if not matches:
        raise FileNotFoundError(
            "Could not find the cached WikiText-103 validation Arrow file under "
            "~/.cache/huggingface/datasets."
        )
    return matches[-1]


def iter_wikitext_documents(validation: Dataset) -> Iterable[tuple[int, int, str]]:
    """Yield contiguous WikiText documents separated by blank rows."""

    row_indices: list[int] = []
    pieces: list[str] = []
    for row_idx, example in enumerate(validation):
        text = str(example["text"])
        if text.strip():
            row_indices.append(row_idx)
            pieces.append(text.strip())
        elif pieces:
            yield row_indices[0], row_indices[-1], "\n".join(pieces)
            row_indices = []
            pieces = []

    if pieces:
        yield row_indices[0], row_indices[-1], "\n".join(pieces)


def select_longest_wikitext_document(
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    arrow_path: Path | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Select the longest validation document and take its first ``seq_len`` tokens."""

    resolved_arrow = arrow_path or find_cached_wikitext_validation_arrow()
    validation = Dataset.from_file(str(resolved_arrow))

    original_model_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = 10**9  # Avoid warnings while measuring full documents.
    best: tuple[int, int, str, list[int]] | None = None
    try:
        for row_start, row_end, text in iter_wikitext_documents(validation):
            input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if best is None or len(input_ids) > len(best[3]):
                best = (row_start, row_end, text, input_ids)
    finally:
        tokenizer.model_max_length = original_model_max_length

    if best is None:
        raise RuntimeError("No non-empty WikiText validation document was found.")

    row_start, row_end, text, full_input_ids = best
    if len(full_input_ids) < seq_len:
        raise RuntimeError(
            f"The longest WikiText validation document has only {len(full_input_ids)} "
            f"tokens, fewer than the requested {seq_len}."
        )

    metadata = {
        "arrow_path": str(resolved_arrow),
        "row_start": int(row_start),
        "row_end": int(row_end),
        "full_document_token_count": int(len(full_input_ids)),
        "used_token_count": int(seq_len),
        "text_preview": " ".join(text.split())[:200],
    }
    return full_input_ids[:seq_len], metadata


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


def causal_additive_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype, device=device)
    forbidden = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1
    )
    return mask.masked_fill(forbidden, torch.finfo(dtype).min)


def row_softmax(score_matrix: torch.Tensor, causal: bool) -> torch.Tensor:
    if causal:
        seq_len = int(score_matrix.shape[-1])
        forbidden = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=score_matrix.device),
            diagonal=1,
        )
        score_matrix = score_matrix.masked_fill(forbidden, torch.finfo(score_matrix.dtype).min)
    return torch.softmax(score_matrix, dim=-1)


def svdvals_cpu(matrix: torch.Tensor) -> torch.Tensor:
    """Compute stable float32 singular values on CPU and return them descending."""

    return torch.linalg.svdvals(matrix.detach().to(device="cpu", dtype=torch.float32))


@torch.inference_mode()
def compare_one_head(
    score_matrix: torch.Tensor,
    real_attention: torch.Tensor,
    causal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return real, canonical, and score spectra plus reconstruction error."""

    score_cpu = score_matrix.detach().to(device="cpu", dtype=torch.float32)
    real_cpu = real_attention.detach().to(device="cpu", dtype=torch.float32)

    manual_real = row_softmax(score_cpu, causal=causal)
    reconstruction_error = float(torch.max(torch.abs(real_cpu - manual_real)).item())

    score_singular_values = svdvals_cpu(score_cpu)
    sigma = torch.diag(score_singular_values)
    canonical_attention = row_softmax(sigma, causal=causal)

    real_singular_values = svdvals_cpu(real_cpu)
    canonical_singular_values = svdvals_cpu(canonical_attention)

    return (
        np.maximum(real_singular_values.numpy(), 0.0),
        np.maximum(canonical_singular_values.numpy(), 0.0),
        np.maximum(score_singular_values.numpy(), 0.0),
        reconstruction_error,
    )


@torch.inference_mode()
def collect_gpt2_comparison(
    device: torch.device,
    seq_len: int,
    max_layers: int | None,
    max_heads: int | None,
    wikitext_arrow: Path | None,
) -> SpectralComparison:
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    model = GPT2Model.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()

    model_max_length = int(model.config.n_positions)
    if seq_len > model_max_length:
        raise ValueError(
            f"GPT-2 supports at most {model_max_length} positions, but {seq_len} were requested."
        )

    input_ids, sample_metadata = select_longest_wikitext_document(
        tokenizer=tokenizer,
        seq_len=seq_len,
        arrow_path=wikitext_arrow,
    )

    n_layers = len(model.h) if max_layers is None else min(max_layers, len(model.h))
    n_heads_available = int(model.config.num_attention_heads)
    n_heads = n_heads_available if max_heads is None else min(max_heads, n_heads_available)

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
        with torch.inference_mode():
            model(input_ids=tensor_input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    real_spectra = np.empty((n_layers, n_heads, seq_len), dtype=np.float32)
    canonical_spectra = np.empty_like(real_spectra)
    score_spectra = np.empty_like(real_spectra)
    reconstruction_error = np.empty((n_layers, n_heads), dtype=np.float64)

    for layer_idx in range(n_layers):
        if layer_idx not in captured_inputs:
            raise RuntimeError(f"Missing captured GPT-2 attention input for layer {layer_idx}.")

        hidden_states = captured_inputs[layer_idx]
        attention_module = model.h[layer_idx].attn
        qkv = attention_module.c_attn(hidden_states)
        query, key, _ = qkv.split(attention_module.split_size, dim=2)
        query = query.view(1, seq_len, n_heads_available, attention_module.head_dim).transpose(1, 2)
        key = key.view(1, seq_len, n_heads_available, attention_module.head_dim).transpose(1, 2)
        score = torch.matmul(query, key.transpose(-1, -2)) * float(attention_module.scaling)

        mask = causal_additive_mask(seq_len, hidden_states.dtype, hidden_states.device)
        with torch.inference_mode():
            _, actual_attention = attention_module(hidden_states, attention_mask=mask)

        for head_idx in range(n_heads):
            real_sv, canonical_sv, score_sv, error = compare_one_head(
                score_matrix=score[0, head_idx],
                real_attention=actual_attention[0, head_idx],
                causal=True,
            )
            real_spectra[layer_idx, head_idx] = real_sv
            canonical_spectra[layer_idx, head_idx] = canonical_sv
            score_spectra[layer_idx, head_idx] = score_sv
            reconstruction_error[layer_idx, head_idx] = error

        print(
            f"GPT-2 layer {layer_idx + 1}/{n_layers}: "
            f"max attention reconstruction error="
            f"{reconstruction_error[layer_idx].max():.3e}"
        )
        if reconstruction_error[layer_idx].max() > ATTENTION_RECONSTRUCTION_TOLERANCE:
            raise RuntimeError(
                f"GPT-2 layer {layer_idx} reconstruction error exceeds "
                f"{ATTENTION_RECONSTRUCTION_TOLERANCE:g}; check score scaling or masking."
            )

    sample_id = f"wikitext_rows_{sample_metadata['row_start']}-{sample_metadata['row_end']}"
    metadata = {
        **sample_metadata,
        "model": "gpt2",
        "causal": True,
        "num_layers": int(n_layers),
        "num_heads": int(n_heads),
        "num_heads_available": int(n_heads_available),
        "head_dim": int(model.h[0].attn.head_dim),
        "device": str(device),
    }

    del model, tensor_input_ids, captured_inputs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return SpectralComparison(
        slug="gpt2_wikitext",
        display_name="GPT-2",
        dataset_name="WikiText-103 validation",
        sample_id=sample_id,
        seq_len=seq_len,
        real_singular_values=real_spectra,
        canonical_singular_values=canonical_spectra,
        score_singular_values=score_spectra,
        reconstruction_error=reconstruction_error,
        metadata=metadata,
    )


@torch.inference_mode()
def collect_vit_comparison(
    device: torch.device,
    imagenette_root: Path,
    sample_idx: int,
    max_layers: int | None,
    max_heads: int | None,
) -> SpectralComparison:
    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    validation_root = imagenette_root.expanduser() / "val"
    if not validation_root.is_dir():
        raise FileNotFoundError(
            f"Imagenette validation directory not found: {validation_root}"
        )

    imagenette = vision_datasets.ImageFolder(
        root=validation_root,
        transform=weights.transforms(),
    )
    if not 0 <= sample_idx < len(imagenette):
        raise IndexError(
            f"Imagenette sample index {sample_idx} is outside [0, {len(imagenette) - 1}]."
        )

    sample_image, sample_label = imagenette[sample_idx]
    sample_path = Path(imagenette.samples[sample_idx][0])
    class_name = imagenette.classes[sample_label]

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
        with torch.inference_mode():
            model(image_batch)
    finally:
        for hook in hooks:
            hook.remove()

    if 0 not in captured_ln1:
        raise RuntimeError("Failed to capture the ViT LayerNorm output.")
    seq_len = int(captured_ln1[0].shape[1])

    real_spectra = np.empty((n_layers, n_heads, seq_len), dtype=np.float32)
    canonical_spectra = np.empty_like(real_spectra)
    score_spectra = np.empty_like(real_spectra)
    reconstruction_error = np.empty((n_layers, n_heads), dtype=np.float64)

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
        query = query.view(1, seq_len, n_heads_available, head_dim).transpose(1, 2)
        key = key.view(1, seq_len, n_heads_available, head_dim).transpose(1, 2)
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(float(head_dim))

        with torch.inference_mode():
            _, actual_attention = attention_module(
                hidden_states,
                hidden_states,
                hidden_states,
                need_weights=True,
                average_attn_weights=False,
            )

        for head_idx in range(n_heads):
            real_sv, canonical_sv, score_sv, error = compare_one_head(
                score_matrix=score[0, head_idx],
                real_attention=actual_attention[0, head_idx],
                causal=False,
            )
            real_spectra[layer_idx, head_idx] = real_sv
            canonical_spectra[layer_idx, head_idx] = canonical_sv
            score_spectra[layer_idx, head_idx] = score_sv
            reconstruction_error[layer_idx, head_idx] = error

        print(
            f"ViT layer {layer_idx + 1}/{n_layers}: "
            f"max attention reconstruction error="
            f"{reconstruction_error[layer_idx].max():.3e}"
        )
        if reconstruction_error[layer_idx].max() > ATTENTION_RECONSTRUCTION_TOLERANCE:
            raise RuntimeError(
                f"ViT layer {layer_idx} reconstruction error exceeds "
                f"{ATTENTION_RECONSTRUCTION_TOLERANCE:g}; check Q/K projections or scaling."
            )

    metadata = {
        "model": "vit_b_16",
        "weights": weights.name,
        "causal": False,
        "imagenette_root": str(imagenette_root.expanduser()),
        "sample_index": int(sample_idx),
        "sample_path": str(sample_path),
        "class_synset": class_name,
        "num_layers": int(n_layers),
        "num_heads": int(n_heads),
        "num_heads_available": int(n_heads_available),
        "head_dim": int(head_dim),
        "device": str(device),
    }

    del model, image_batch, captured_ln1
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return SpectralComparison(
        slug="vit_imagenette",
        display_name="ViT-B/16",
        dataset_name="Imagenette validation",
        sample_id=sample_path.name,
        seq_len=seq_len,
        real_singular_values=real_spectra,
        canonical_singular_values=canonical_spectra,
        score_singular_values=score_spectra,
        reconstruction_error=reconstruction_error,
        metadata=metadata,
    )


def effective_rank(singular_values: np.ndarray) -> float:
    total = float(np.sum(singular_values))
    if total <= 0.0:
        return 0.0
    probabilities = singular_values / total
    positive = probabilities[probabilities > 0.0]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def stable_rank(singular_values: np.ndarray) -> float:
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        return 0.0
    return float(np.sum(np.square(singular_values)) / singular_values[0] ** 2)


def energy_rank(singular_values: np.ndarray, fraction: float = 0.99) -> int:
    energy = np.square(singular_values)
    total = float(np.sum(energy))
    if total <= 0.0:
        return 0
    return int(np.searchsorted(np.cumsum(energy) / total, fraction) + 1)


def spectrum_metrics(real_sv: np.ndarray, canonical_sv: np.ndarray) -> dict[str, float]:
    real_sorted = np.sort(real_sv)
    canonical_sorted = np.sort(canonical_sv)
    wasserstein = float(np.mean(np.abs(real_sorted - canonical_sorted)))
    ks_distance = float(ks_2samp(real_sv, canonical_sv, method="auto").statistic)
    denominator = max(float(np.linalg.norm(real_sv)), 1e-12)
    relative_l2 = float(np.linalg.norm(real_sv - canonical_sv) / denominator)
    return {
        "wasserstein_distance": wasserstein,
        "ks_distance": ks_distance,
        "relative_ranked_spectrum_l2": relative_l2,
    }


def build_metric_frames(
    comparisons: list[SpectralComparison],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    head_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []

    for comparison in comparisons:
        log_seq_len = math.log(float(comparison.seq_len))
        for layer_idx in range(comparison.num_layers):
            layer_head_rows: list[dict[str, Any]] = []
            for head_idx in range(comparison.num_heads):
                real_sv = comparison.real_singular_values[layer_idx, head_idx]
                canonical_sv = comparison.canonical_singular_values[layer_idx, head_idx]
                score_sv = comparison.score_singular_values[layer_idx, head_idx]
                row: dict[str, Any] = {
                    "model": comparison.display_name,
                    "dataset": comparison.dataset_name,
                    "sample_id": comparison.sample_id,
                    "seq_len": comparison.seq_len,
                    "layer_idx": layer_idx,
                    "head_idx": head_idx,
                    **spectrum_metrics(real_sv, canonical_sv),
                    "real_effective_rank": effective_rank(real_sv),
                    "canonical_effective_rank": effective_rank(canonical_sv),
                    "real_stable_rank": stable_rank(real_sv),
                    "canonical_stable_rank": stable_rank(canonical_sv),
                    "real_energy_rank_99": energy_rank(real_sv, 0.99),
                    "canonical_energy_rank_99": energy_rank(canonical_sv, 0.99),
                    "real_num_sv_gt_0_1": int(np.count_nonzero(real_sv > 0.1)),
                    "canonical_num_sv_gt_0_1": int(np.count_nonzero(canonical_sv > 0.1)),
                    "real_num_sv_gt_0_5": int(np.count_nonzero(real_sv > 0.5)),
                    "canonical_num_sv_gt_0_5": int(np.count_nonzero(canonical_sv > 0.5)),
                    "real_num_sv_gt_1": int(np.count_nonzero(real_sv > 1.0)),
                    "canonical_num_sv_gt_1": int(np.count_nonzero(canonical_sv > 1.0)),
                    "real_top_singular_value": float(real_sv[0]),
                    "canonical_top_singular_value": float(canonical_sv[0]),
                    "score_effective_rank": effective_rank(score_sv),
                    "score_stable_rank": stable_rank(score_sv),
                    "score_top_singular_value": float(score_sv[0]),
                    "score_top_over_log_seq_len": float(score_sv[0] / log_seq_len),
                    "score_num_sv_gt_log_seq_len": int(np.count_nonzero(score_sv > log_seq_len)),
                    "score_num_sv_gt_2log_seq_len": int(
                        np.count_nonzero(score_sv > 2.0 * log_seq_len)
                    ),
                    "attention_reconstruction_max_abs_error": float(
                        comparison.reconstruction_error[layer_idx, head_idx]
                    ),
                }
                head_rows.append(row)
                layer_head_rows.append(row)

            pooled_real = comparison.real_singular_values[layer_idx].reshape(-1)
            pooled_canonical = comparison.canonical_singular_values[layer_idx].reshape(-1)
            per_head_w1 = np.array(
                [row["wasserstein_distance"] for row in layer_head_rows], dtype=float
            )
            per_head_ks = np.array([row["ks_distance"] for row in layer_head_rows], dtype=float)
            per_head_rel_l2 = np.array(
                [row["relative_ranked_spectrum_l2"] for row in layer_head_rows], dtype=float
            )
            layer_rows.append(
                {
                    "model": comparison.display_name,
                    "dataset": comparison.dataset_name,
                    "sample_id": comparison.sample_id,
                    "seq_len": comparison.seq_len,
                    "layer_idx": layer_idx,
                    "num_heads": comparison.num_heads,
                    "pooled_wasserstein_distance": spectrum_metrics(
                        pooled_real, pooled_canonical
                    )["wasserstein_distance"],
                    "pooled_ks_distance": spectrum_metrics(pooled_real, pooled_canonical)[
                        "ks_distance"
                    ],
                    "head_wasserstein_median": float(np.median(per_head_w1)),
                    "head_wasserstein_q25": float(np.quantile(per_head_w1, 0.25)),
                    "head_wasserstein_q75": float(np.quantile(per_head_w1, 0.75)),
                    "head_wasserstein_max": float(np.max(per_head_w1)),
                    "head_ks_median": float(np.median(per_head_ks)),
                    "head_ks_max": float(np.max(per_head_ks)),
                    "head_relative_l2_median": float(np.median(per_head_rel_l2)),
                    "score_num_sv_gt_log_seq_len_median": float(
                        np.median(
                            [row["score_num_sv_gt_log_seq_len"] for row in layer_head_rows]
                        )
                    ),
                    "score_top_over_log_seq_len_median": float(
                        np.median(
                            [row["score_top_over_log_seq_len"] for row in layer_head_rows]
                        )
                    ),
                    "attention_reconstruction_max_abs_error": float(
                        np.max(comparison.reconstruction_error[layer_idx])
                    ),
                }
            )

    return pd.DataFrame(head_rows), pd.DataFrame(layer_rows)


def subplot_grid(num_layers: int) -> tuple[int, int]:
    if num_layers == 12:
        return 3, 4
    n_cols = min(4, max(1, num_layers))
    return int(math.ceil(num_layers / n_cols)), n_cols


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def layer_head_wasserstein(comparison: SpectralComparison, layer_idx: int) -> np.ndarray:
    return np.array(
        [
            spectrum_metrics(
                comparison.real_singular_values[layer_idx, head_idx],
                comparison.canonical_singular_values[layer_idx, head_idx],
            )["wasserstein_distance"]
            for head_idx in range(comparison.num_heads)
        ],
        dtype=float,
    )


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs: list[Path] = []
    for suffix in (".png", ".svg"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_layerwise_lsd(
    comparison: SpectralComparison,
    output_dir: Path,
    num_bins: int,
) -> list[Path]:
    """Plot the per-layer LSD after pooling separately computed head spectra."""

    n_rows, n_cols = subplot_grid(comparison.num_layers)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.1 * n_cols, 3.15 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    log_real_all = np.log10(
        np.clip(comparison.real_singular_values.reshape(-1), SPECTRAL_FLOOR, None)
    )
    log_canonical_all = np.log10(
        np.clip(comparison.canonical_singular_values.reshape(-1), SPECTRAL_FLOOR, None)
    )
    log_max = max(0.1, float(np.max(log_real_all)), float(np.max(log_canonical_all)))
    bin_edges = np.linspace(math.log10(SPECTRAL_FLOOR), log_max, num_bins + 1)
    if comparison.metadata["causal"]:
        real_label = r"Real $A_h=\mathrm{softmax}(S_h+M)$"
        canonical_label = r"Canonical $A_{\Sigma,h}=\mathrm{softmax}(\Sigma_h+M)$"
    else:
        real_label = r"Real $A_h=\mathrm{softmax}(S_h)$"
        canonical_label = r"Canonical $A_{\Sigma,h}=\mathrm{softmax}(\Sigma_h)$"

    for layer_idx, ax in enumerate(flat_axes):
        if layer_idx >= comparison.num_layers:
            ax.axis("off")
            continue

        real = comparison.real_singular_values[layer_idx].reshape(-1)
        canonical = comparison.canonical_singular_values[layer_idx].reshape(-1)
        log_real = np.log10(np.clip(real, SPECTRAL_FLOOR, None))
        log_canonical = np.log10(np.clip(canonical, SPECTRAL_FLOOR, None))

        ax.hist(
            log_real,
            bins=bin_edges,
            density=True,
            histtype="stepfilled",
            color=REAL_COLOR,
            edgecolor=REAL_COLOR,
            linewidth=1.2,
            alpha=0.24,
            label=real_label,
        )
        ax.hist(
            log_canonical,
            bins=bin_edges,
            density=True,
            histtype="step",
            color=CANONICAL_COLOR,
            linewidth=1.6,
            label=canonical_label,
        )
        head_w1 = layer_head_wasserstein(comparison, layer_idx)
        ax.text(
            0.97,
            0.94,
            rf"median head $W_1$={np.median(head_w1):.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        ax.set_title(f"Layer {layer_idx + 1}")
        ax.grid(axis="y", alpha=0.22, linewidth=0.6)

    for row_idx in range(n_rows):
        axes[row_idx, 0].set_ylabel(r"Density of $\log_{10}\sigma$")
    for col_idx in range(n_cols):
        axes[-1, col_idx].set_xlabel(r"$\log_{10}$ singular value")

    mask_text = "causal mask applied to both" if comparison.metadata["causal"] else "bidirectional"
    fig.suptitle(
        f"{comparison.display_name}: pooled per-head attention log-spectrum by layer",
        y=0.99,
        fontsize=13,
    )
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=2)
    fig.text(
        0.5,
        0.005,
        f"{comparison.dataset_name}; {comparison.seq_len} tokens; {mask_text}.  "
        f"Each layer pools {comparison.num_heads} spectra computed independently per head "
        f"({comparison.num_heads * comparison.seq_len:,} singular values per curve); "
        f"values below {SPECTRAL_FLOOR:g} are shown at the left boundary.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.86))
    return save_figure(fig, output_dir / f"{comparison.slug}_layerwise_lsd")


def plot_layerwise_head_spectra(
    comparison: SpectralComparison,
    output_dir: Path,
) -> list[Path]:
    """Plot each head's ranked spectrum, with layer-wise median and IQR."""

    n_rows, n_cols = subplot_grid(comparison.num_layers)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.1 * n_cols, 3.15 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    rank_fraction = np.arange(1, comparison.seq_len + 1, dtype=float) / comparison.seq_len
    global_max = max(
        float(np.max(comparison.real_singular_values)),
        float(np.max(comparison.canonical_singular_values)),
    )

    for layer_idx, ax in enumerate(flat_axes):
        if layer_idx >= comparison.num_layers:
            ax.axis("off")
            continue

        real = np.clip(
            comparison.real_singular_values[layer_idx], SPECTRAL_FLOOR, None
        )
        canonical = np.clip(
            comparison.canonical_singular_values[layer_idx], SPECTRAL_FLOOR, None
        )
        for head_idx in range(comparison.num_heads):
            ax.plot(
                rank_fraction,
                real[head_idx],
                color=REAL_COLOR,
                linewidth=0.55,
                alpha=0.17,
            )
            ax.plot(
                rank_fraction,
                canonical[head_idx],
                color=CANONICAL_COLOR,
                linestyle="--",
                linewidth=0.55,
                alpha=0.17,
            )

        real_q25, real_median, real_q75 = np.quantile(real, [0.25, 0.5, 0.75], axis=0)
        canonical_q25, canonical_median, canonical_q75 = np.quantile(
            canonical, [0.25, 0.5, 0.75], axis=0
        )
        ax.fill_between(
            rank_fraction,
            real_q25,
            real_q75,
            color=REAL_COLOR,
            alpha=0.13,
            linewidth=0,
        )
        ax.fill_between(
            rank_fraction,
            canonical_q25,
            canonical_q75,
            color=CANONICAL_COLOR,
            alpha=0.10,
            linewidth=0,
        )
        ax.plot(
            rank_fraction,
            real_median,
            color=REAL_COLOR,
            linewidth=1.8,
            label="Real median (heads)",
        )
        ax.plot(
            rank_fraction,
            canonical_median,
            color=CANONICAL_COLOR,
            linestyle="--",
            linewidth=1.8,
            label="Canonical median (heads)",
        )

        head_w1 = layer_head_wasserstein(comparison, layer_idx)
        ax.text(
            0.97,
            0.80,
            rf"head $W_1$: {np.median(head_w1):.3g} med, {np.max(head_w1):.3g} max",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        ax.set_title(f"Layer {layer_idx + 1}")
        ax.set_yscale("log")
        ax.set_ylim(SPECTRAL_FLOOR, global_max * 1.15)
        ax.grid(which="both", alpha=0.2, linewidth=0.55)

    for row_idx in range(n_rows):
        axes[row_idx, 0].set_ylabel(r"Raw singular value $\sigma_j(A_h)$")
    for col_idx in range(n_cols):
        axes[-1, col_idx].set_xlabel(r"Normalized singular-value index $j/\ell$")

    fig.suptitle(
        f"{comparison.display_name}: head-resolved real vs. spectral-canonical spectra",
        y=0.99,
        fontsize=13,
    )
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=2)
    fig.text(
        0.5,
        0.005,
        f"{comparison.dataset_name}; {comparison.seq_len} tokens.  "
        "Thin curves are individual heads; thick curves and bands are the head-wise median and IQR. "
        r"The horizontal coordinate is the ordered index $j/\ell$, not a rank proxy.  "
        f"Values below {SPECTRAL_FLOOR:g} are clipped for the log axis.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.86))
    return save_figure(fig, output_dir / f"{comparison.slug}_layerwise_head_spectra")


def save_raw_comparison(comparison: SpectralComparison, output_dir: Path) -> Path:
    path = output_dir / f"{comparison.slug}_spectra.npz"
    np.savez_compressed(
        path,
        real_singular_values=comparison.real_singular_values,
        canonical_singular_values=comparison.canonical_singular_values,
        score_singular_values=comparison.score_singular_values,
        reconstruction_error=comparison.reconstruction_error,
        metadata_json=np.array(json.dumps(comparison.metadata, sort_keys=True)),
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per-head spectra of real and spectral-canonical attention.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        choices=("all", "gpt2", "vit"),
        default="all",
        help="Which pretrained model experiment to run.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-bins", type=int, default=80)
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Optional layer limit for smoke tests; omitted uses all 12 layers.",
    )
    parser.add_argument(
        "--max-heads",
        type=int,
        default=None,
        help="Optional head limit for smoke tests; omitted uses all 12 heads.",
    )
    parser.add_argument(
        "--gpt2-seq-len",
        type=int,
        default=1024,
        help="Number of tokens taken from the longest WikiText validation document.",
    )
    parser.add_argument(
        "--wikitext-arrow",
        type=Path,
        default=None,
        help="Optional explicit cached WikiText validation Arrow path.",
    )
    parser.add_argument(
        "--imagenette-root",
        type=Path,
        default=DEFAULT_IMAGENETTE_ROOT,
        help="Imagenette root containing the val/ directory.",
    )
    parser.add_argument("--imagenette-sample-index", type=int, default=1)
    args = parser.parse_args()

    if args.num_bins < 10:
        parser.error("--num-bins must be at least 10.")
    if args.gpt2_seq_len <= 0:
        parser.error("--gpt2-seq-len must be positive.")
    if args.max_layers is not None and args.max_layers <= 0:
        parser.error("--max-layers must be positive.")
    if args.max_heads is not None and args.max_heads <= 0:
        parser.error("--max-heads must be positive.")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_plot_style()
    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    comparisons: list[SpectralComparison] = []
    generated_paths: list[Path] = []

    if args.models in ("all", "gpt2"):
        print("Collecting GPT-2 / WikiText spectra...")
        gpt2_comparison = collect_gpt2_comparison(
            device=device,
            seq_len=args.gpt2_seq_len,
            max_layers=args.max_layers,
            max_heads=args.max_heads,
            wikitext_arrow=args.wikitext_arrow,
        )
        comparisons.append(gpt2_comparison)

    if args.models in ("all", "vit"):
        print("Collecting ViT / Imagenette spectra...")
        vit_comparison = collect_vit_comparison(
            device=device,
            imagenette_root=args.imagenette_root,
            sample_idx=args.imagenette_sample_index,
            max_layers=args.max_layers,
            max_heads=args.max_heads,
        )
        comparisons.append(vit_comparison)

    for comparison in comparisons:
        generated_paths.append(save_raw_comparison(comparison, output_dir))
        generated_paths.extend(plot_layerwise_lsd(comparison, output_dir, args.num_bins))
        generated_paths.extend(plot_layerwise_head_spectra(comparison, output_dir))

    head_metrics, layer_metrics = build_metric_frames(comparisons)
    head_metrics_path = output_dir / "real_vs_canonical_head_metrics.csv"
    layer_metrics_path = output_dir / "real_vs_canonical_layer_summary.csv"
    head_metrics.to_csv(head_metrics_path, index=False)
    layer_metrics.to_csv(layer_metrics_path, index=False)
    generated_paths.extend([head_metrics_path, layer_metrics_path])

    metadata_path = output_dir / "experiment_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "device": str(device),
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "transformers_version": transformers.__version__,
                "spectral_floor_for_plots": SPECTRAL_FLOOR,
                "attention_reconstruction_tolerance": ATTENTION_RECONSTRUCTION_TOLERANCE,
                "canonical_diagonal_order": "descending score singular values",
                "aggregation": (
                    "Compute singular values independently for every attention head, then "
                    "pool all head singular values with equal mass within each layer."
                ),
                "comparisons": [comparison.metadata for comparison in comparisons],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    generated_paths.append(metadata_path)

    print("Generated:")
    for path in generated_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
