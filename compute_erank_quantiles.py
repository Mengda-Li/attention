import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from datasets import load_dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer


SINGULAR_VALUE_QUANTILES = (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def set_seed(seed: int = 233) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def effective_rank_entropy(p: np.ndarray) -> float:
    p_nonzero = p[p > 0]
    if p_nonzero.size == 0:
        return 0.0
    H = -float(np.sum(p_nonzero * np.log(p_nonzero)))
    return float(np.exp(H))


def quantile_rank(p: np.ndarray, thresholds: Sequence[float]) -> dict:
    c = np.cumsum(p)
    out = {}
    for t in thresholds:
        idx = int(np.searchsorted(c, t) + 1)
        out[f"q_rank_{int(t * 100)}"] = idx
    return out


def singular_value_quantiles(singular_values: np.ndarray, quantiles: Sequence[float]) -> dict:
    if singular_values.size == 0:
        return {f"singular_value_q{int(q * 100)}": 0.0 for q in quantiles}
    return {
        f"singular_value_q{int(q * 100)}": float(np.quantile(singular_values, q))
        for q in quantiles
    }


def compute_singular_metrics(
    score_matrix: np.ndarray,
    thresholds: Sequence[float],
    eps: float = 1e-12,
) -> dict:
    singular_values = np.linalg.svd(score_matrix, compute_uv=False)
    singular_values = np.maximum(singular_values, 0.0)

    max_sv = float(np.max(singular_values)) if singular_values.size > 0 else 0.0
    mean_sv = float(np.mean(singular_values)) if singular_values.size > 0 else 0.0
    median_sv = float(np.median(singular_values)) if singular_values.size > 0 else 0.0
    fro_norm = float(np.linalg.norm(score_matrix, ord="fro"))
    nuc_norm = float(np.sum(singular_values))

    denom = max(max_sv * max_sv, eps)
    stable_rank = float((fro_norm * fro_norm) / denom)

    sum_s = float(np.sum(singular_values))
    if sum_s <= 0:
        p1 = np.zeros_like(singular_values)
    else:
        p1 = singular_values / sum_s
    erank1_entropy = effective_rank_entropy(p1)
    q_ranks_1 = quantile_rank(p1, thresholds)

    energy = singular_values * singular_values
    sum_sq = float(np.sum(energy))
    if sum_sq <= 0:
        p2 = np.zeros_like(singular_values)
    else:
        p2 = energy / sum_sq
    erank2_entropy = effective_rank_entropy(p2)
    q_ranks_2 = quantile_rank(p2, thresholds)

    trace = float(np.trace(score_matrix))
    seq_len = score_matrix.shape[0] if score_matrix.ndim == 2 else 0
    trace_per_token = float(trace / max(seq_len, 1))

    metrics = {
        "max_singular_value": max_sv,
        "mean_singular_value": mean_sv,
        "median_singular_value": median_sv,
        "stable_rank": stable_rank,
        "erank1_entropy": erank1_entropy,
        "erank2_entropy": erank2_entropy,
        "trace": trace,
        "trace_per_token": trace_per_token,
        "frobenius_norm": fro_norm,
        "nuclear_norm": nuc_norm,
    }
    metrics.update(singular_value_quantiles(singular_values, SINGULAR_VALUE_QUANTILES))
    for k, v in q_ranks_1.items():
        metrics[f"{k}_erank1"] = int(v)
    for k, v in q_ranks_2.items():
        metrics[f"{k}_erank2"] = int(v)

    return metrics


def flatten_columns(columns: pd.Index) -> list[str]:
    flat_cols: list[str] = []
    for col in columns:
        if isinstance(col, tuple):
            left, right = col
            if right == "":
                flat_cols.append(str(left))
            else:
                flat_cols.append(f"{left}_{right}")
        else:
            flat_cols.append(str(col))
    return flat_cols


def build_summary(df: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    metric_cols = [
        "max_singular_value",
        "mean_singular_value",
        "median_singular_value",
        "stable_rank",
        "erank1_entropy",
        "erank2_entropy",
        "trace",
        "trace_per_token",
        "frobenius_norm",
        "nuclear_norm",
    ]
    metric_cols.extend(
        [f"singular_value_q{int(q * 100)}" for q in SINGULAR_VALUE_QUANTILES]
    )
    metric_cols.extend([f"q_rank_{int(t * 100)}_erank1" for t in thresholds])
    metric_cols.extend([f"q_rank_{int(t * 100)}_erank2" for t in thresholds])

    layer_head_summary = (
        df.groupby(["model", "dataset", "layer_idx", "head_idx"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    layer_head_summary.columns = flatten_columns(layer_head_summary.columns)
    layer_head_counts = (
        df.groupby(["model", "dataset", "layer_idx", "head_idx"])
        .size()
        .reset_index(name="row_count")
    )
    layer_head_summary = layer_head_summary.merge(
        layer_head_counts,
        on=["model", "dataset", "layer_idx", "head_idx"],
        how="left",
    )
    layer_head_summary.insert(0, "aggregation_level", "layer_head")

    layer_summary = (
        df.groupby(["model", "dataset", "layer_idx"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    layer_summary.columns = flatten_columns(layer_summary.columns)
    layer_counts = (
        df.groupby(["model", "dataset", "layer_idx"])
        .size()
        .reset_index(name="row_count")
    )
    layer_summary = layer_summary.merge(
        layer_counts,
        on=["model", "dataset", "layer_idx"],
        how="left",
    )
    layer_summary["head_idx"] = -1
    layer_summary.insert(0, "aggregation_level", "layer")

    return pd.concat([layer_head_summary, layer_summary], ignore_index=True)


def make_hidden_capture_hook(layer_idx: int, captured: dict[int, torch.Tensor]):
    def hook(module, inputs, output):
        _ = module
        _ = output
        captured[layer_idx] = inputs[0].detach()

    return hook


def compute_text_rows(
    device: str,
    num_samples: int | None,
    max_length: int | None,
    thresholds: Sequence[float],
) -> list[dict]:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    validation = dataset["validation"]
    texts = [ex["text"] for ex in validation if len(ex["text"].strip()) > 100]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    resolved_max_length = int(tokenizer.model_max_length) if max_length is None else max_length

    model = GPT2Model.from_pretrained("gpt2", attn_implementation="eager")
    model.to(device)
    model.eval()

    n_layers = len(model.h)
    n_heads = int(model.config.num_attention_heads)
    d_k = int(model.config.hidden_size // n_heads)

    selected = list(enumerate(texts if num_samples is None else texts[:num_samples]))

    captured_inputs: dict[int, torch.Tensor] = {}
    hooks = [
        model.h[layer_idx].attn.register_forward_hook(
            make_hidden_capture_hook(layer_idx, captured_inputs)
        )
        for layer_idx in range(n_layers)
    ]

    rows: list[dict] = []
    try:
        for sample_idx, text in tqdm(selected, desc="Text samples", unit="sample"):
            try:
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=resolved_max_length,
                )
                input_ids = encoded["input_ids"].to(device)
                seq_len = int(input_ids.shape[1])

                captured_inputs.clear()
                with torch.no_grad():
                    model(input_ids=input_ids)

                for layer_idx in range(n_layers):
                    hidden_states = captured_inputs[layer_idx]
                    attn_module = model.h[layer_idx].attn
                    qkv = attn_module.c_attn(hidden_states)
                    query, key, _ = torch.split(qkv, attn_module.split_size, dim=2)

                    query = query.view(1, seq_len, n_heads, d_k).permute(0, 2, 1, 3)
                    key = key.view(1, seq_len, n_heads, d_k).permute(0, 2, 1, 3)

                    for head_idx in range(n_heads):
                        q_h = query[0, head_idx].detach().float().cpu().numpy()
                        k_h = key[0, head_idx].detach().float().cpu().numpy()
                        score = (q_h @ k_h.T) / np.sqrt(float(d_k))

                        row = {
                            "model": "gpt2",
                            "dataset": "wikitext-103-validation",
                            "sample_idx": int(sample_idx),
                            "sample_id": f"article_{sample_idx}",
                            "layer_idx": int(layer_idx),
                            "head_idx": int(head_idx),
                            "seq_len": int(seq_len),
                            "d_k": int(d_k),
                        }
                        row.update(compute_singular_metrics(score, thresholds))
                        rows.append(row)
            except Exception as exc:
                tqdm.write(f"Skipping text sample {sample_idx} due to error: {exc}")
    finally:
        for hook in hooks:
            hook.remove()

    return rows


def compute_image_rows(
    device: str,
    num_samples: int | None,
    max_layers: int,
    thresholds: Sequence[float],
) -> list[dict]:
    transform = transforms.Compose(
        [
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    imagenette = datasets.Imagenette(root="./data", split="val", transform=transform)

    num_selected = len(imagenette) if num_samples is None else min(num_samples, len(imagenette))

    model = torchvision.models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")
    model.to(device)
    model.eval()

    n_layers_available = len(model.encoder.layers)
    n_layers = min(max_layers, n_layers_available)
    attn_probe = model.encoder.layers[0].self_attention
    n_heads = int(attn_probe.num_heads)
    embed_dim = int(attn_probe.embed_dim)
    d_k = int(embed_dim // n_heads)

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            _ = module
            _ = inputs
            captured[layer_idx] = output.detach()

        return hook

    hooks = [
        model.encoder.layers[layer_idx].ln_1.register_forward_hook(make_hook(layer_idx))
        for layer_idx in range(n_layers)
    ]

    rows: list[dict] = []
    try:
        for sample_idx in tqdm(range(num_selected), desc="Image samples", unit="sample"):
            try:
                image, _ = imagenette[sample_idx]
                image_batch = image.unsqueeze(0).to(device)

                captured.clear()
                with torch.no_grad():
                    model(image_batch)

                sample_path = ""
                if hasattr(imagenette, "samples") and sample_idx < len(imagenette.samples):
                    sample_path = str(imagenette.samples[sample_idx][0])

                for layer_idx in range(n_layers):
                    x_ln1 = captured[layer_idx]
                    attn = model.encoder.layers[layer_idx].self_attention

                    w_qkv = attn.in_proj_weight
                    b_qkv = attn.in_proj_bias
                    q = F.linear(x_ln1, w_qkv[:embed_dim], b_qkv[:embed_dim])
                    k = F.linear(x_ln1, w_qkv[embed_dim : 2 * embed_dim], b_qkv[embed_dim : 2 * embed_dim])

                    seq_len = int(q.shape[1])
                    q = q.view(1, seq_len, n_heads, d_k).permute(0, 2, 1, 3)
                    k = k.view(1, seq_len, n_heads, d_k).permute(0, 2, 1, 3)

                    for head_idx in range(n_heads):
                        q_h = q[0, head_idx].detach().float().cpu().numpy()
                        k_h = k[0, head_idx].detach().float().cpu().numpy()
                        score = (q_h @ k_h.T) / np.sqrt(float(d_k))

                        row = {
                            "model": "vit_b_16",
                            "dataset": "imagenette-val",
                            "sample_idx": int(sample_idx),
                            "sample_id": sample_path if sample_path else f"image_{sample_idx}",
                            "layer_idx": int(layer_idx),
                            "head_idx": int(head_idx),
                            "seq_len": int(seq_len),
                            "d_k": int(d_k),
                        }
                        row.update(compute_singular_metrics(score, thresholds))
                        rows.append(row)
            except Exception as exc:
                tqdm.write(f"Skipping image sample {sample_idx} due to error: {exc}")
    finally:
        for hook in hooks:
            hook.remove()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute effective-rank and quantile-rank metrics directly over text and image attention scores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (cpu or cuda)")
    parser.add_argument(
        "--num_text_samples",
        type=int,
        default=None,
        help="Number of WikiText validation samples to analyze. If omitted, use all filtered samples.",
    )
    parser.add_argument(
        "--num_image_samples",
        type=int,
        default=None,
        help="Number of Imagenette validation samples to analyze. If omitted, use all samples.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum GPT-2 token length per text sample.",
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=12,
        help="Maximum number of ViT encoder layers to analyze.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/erank_quantiles_detailed.csv",
        help="Detailed combined output CSV path.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="results/erank_quantiles_summary.csv",
        help="Summary CSV path.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.99, 0.95, 0.90],
        help="Thresholds for quantile rank computation.",
    )
    args = parser.parse_args()

    if args.max_layers <= 0:
        raise ValueError("--max_layers must be at least 1.")

    thresholds = tuple(args.thresholds)
    set_seed(233)

    print("[1/4] Computing text metrics...")
    text_rows = compute_text_rows(
        device=args.device,
        num_samples=args.num_text_samples,
        max_length=args.max_length,
        thresholds=thresholds,
    )
    print(f"  Text rows: {len(text_rows)}")

    print("[2/4] Computing image metrics...")
    image_rows = compute_image_rows(
        device=args.device,
        num_samples=args.num_image_samples,
        max_layers=args.max_layers,
        thresholds=thresholds,
    )
    print(f"  Image rows: {len(image_rows)}")

    rows = text_rows + image_rows
    if not rows:
        raise RuntimeError("No rows were produced; check dataset/model availability and arguments.")

    print("[3/4] Building outputs...")
    detailed = pd.DataFrame(rows)
    summary = build_summary(detailed, thresholds)

    output_path = Path(args.output_csv)
    summary_path = Path(args.summary_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    detailed.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("[4/4] Done.")
    print(f"Wrote detailed CSV: {output_path} ({len(detailed)} rows)")
    print(f"Wrote summary CSV: {summary_path} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
