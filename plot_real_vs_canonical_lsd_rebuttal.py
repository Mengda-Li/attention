"""Rebuttal-focused LSD comparisons for real and spectral-canonical attention.

The primary Wasserstein figure uses the exact one-dimensional W1 distance
between the two equal-mass empirical singular-value distributions. Exact KL
between those finite atomic measures is generically infinite, so the KL figure
uses explicitly labeled, fixed log-histogram distributions with uniform
contamination smoothing. Every metric is computed for one matched
sample-layer-head tuple before any aggregation.
"""

from __future__ import annotations

import argparse
import json
import os
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
import torchvision
import transformers

from plot_real_vs_canonical_rank_proxies import (
    ALL_KL_METRICS,
    DEFAULT_IMAGENETTE_ROOT,
    KL_CLIPPING_METRICS,
    KL_LOG10_MAX,
    KL_LOG10_MIN,
    KL_PRIMARY_METRICS,
    KL_PRIMARY_NUM_BINS,
    KL_PRIMARY_UNIFORM_CONTAMINATION,
    KL_SENSITIVITY_METRICS,
    KL_SENSITIVITY_SPECS,
    SEED,
    collect_gpt2_metrics,
    collect_vit_metrics,
    configure_plot_style,
    resolve_device,
    run_kl_unit_checks,
    save_figure,
    sequence_quantile_summary,
    set_seed,
    validate_reused_metrics,
)


DEFAULT_OUTPUT_DIR = Path("results/real_vs_canonical_lsd_rebuttal")
DEFAULT_FIGURE_DIR = Path("rebuttal")
WASSERSTEIN_1 = "wasserstein_distance"
SYMMETRIC_KL = "symmetric_kl_divergence"
PRIMARY_SENSITIVITY_KL = (
    "symmetric_kl_divergence_bins_100_eta_1e_3"
)
SUMMARY_METRICS = (
    WASSERSTEIN_1,
    *KL_PRIMARY_METRICS,
    *KL_CLIPPING_METRICS,
    *KL_SENSITIVITY_METRICS,
)
WASSERSTEIN_COLOR = "#2A9D8F"
KL_COLOR = "#6A4C93"


def draw_iqr_curve(
    ax: plt.Axes,
    summary: pd.DataFrame,
    column: str,
    color: str,
) -> None:
    """Draw the head-level median, IQR, and descriptive 10--90% range."""

    x = summary["seq_len"].to_numpy(dtype=float)
    median = summary[f"{column}_median"].to_numpy(dtype=float)
    q25 = summary[f"{column}_q25"].to_numpy(dtype=float)
    q75 = summary[f"{column}_q75"].to_numpy(dtype=float)
    q10 = summary[f"{column}_q10"].to_numpy(dtype=float)
    q90 = summary[f"{column}_q90"].to_numpy(dtype=float)
    ax.fill_between(x, q10, q90, color=color, alpha=0.07, linewidth=0)
    ax.fill_between(x, q25, q75, color=color, alpha=0.22, linewidth=0)
    ax.plot(
        x,
        median,
        color=color,
        linewidth=2.1,
        marker="o",
        markersize=4.0,
    )


def plot_gpt2_metric(
    metrics: pd.DataFrame,
    metric: str,
    color: str,
    ylabel: str,
    stem: str,
    figure_dir: Path,
) -> list[Path]:
    """Create one all-layer GPT-2 sequence-length panel."""

    summary = sequence_quantile_summary(metrics, [metric])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    draw_iqr_curve(ax, summary, metric, color)
    observed_lengths = summary["seq_len"].astype(int).tolist()
    ax.set_xticks(observed_lengths)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    num_layers = int(metrics["layer_idx"].nunique())
    layer_scope = (
        "all 12 layers" if num_layers == 12 else f"{num_layers} evaluated layers"
    )
    ax.set_title(f"GPT-2 / WikiText-103 ({layer_scope})")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.margins(x=0.025)
    fig.tight_layout(pad=0.7)
    return save_figure(fig, figure_dir / stem)


def plot_gpt2_all_and_last_layer_w1(
    metrics: pd.DataFrame,
    figure_dir: Path,
) -> Path:
    """Create the rebuttal 1x2 W1 comparison for all and last layers."""

    last_layer_idx = int(metrics["layer_idx"].max())
    panels = (
        ("All 12 layers", metrics),
        (
            f"Last layer (Layer {last_layer_idx + 1})",
            metrics[metrics["layer_idx"] == last_layer_idx],
        ),
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.2),
        sharex=True,
        sharey=True,
    )
    for ax, (title, panel_metrics) in zip(axes, panels, strict=True):
        summary = sequence_quantile_summary(panel_metrics, [WASSERSTEIN_1])
        draw_iqr_curve(ax, summary, WASSERSTEIN_1, WASSERSTEIN_COLOR)
        ax.set_xticks(summary["seq_len"].astype(int).tolist())
        ax.set_xlabel("Sequence length")
        ax.set_title(title)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)
        ax.margins(x=0.025)

    axes[0].set_ylabel(
        r"Per-head $W_1$ between empirical singular-value distributions"
    )
    fig.suptitle("GPT-2 / WikiText-103", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.7)
    output_path = figure_dir / "gpt2_w1_all_and_last_layer_vs_seq_len.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_vit_metric(
    metrics: pd.DataFrame,
    metric: str,
    color: str,
    ylabel: str,
    stem: str,
    figure_dir: Path,
) -> list[Path]:
    """Create one ViT layerwise boxplot with all image-head observations."""

    plot_data = metrics.assign(
        layer=(metrics["layer_idx"].astype(int) + 1).astype(str)
    )
    layer_order = [
        str(int(layer_idx) + 1)
        for layer_idx in sorted(metrics["layer_idx"].unique())
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.boxplot(
        data=plot_data,
        x="layer",
        y=metric,
        color=color,
        order=layer_order,
        width=0.62,
        whis=1.5,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_data,
        x="layer",
        y=metric,
        color="0.20",
        order=layer_order,
        size=1.2,
        alpha=0.15,
        jitter=0.20,
        ax=ax,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title("ViT-B/16 / Imagenette")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=0.7)
    return save_figure(fig, figure_dir / stem)


def grouped_quantile_summary(
    metrics: pd.DataFrame,
    group_column: str,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Summarize raw observations by a categorical grouping column."""

    rows: list[dict[str, Any]] = []
    for group_value, group in metrics.groupby(group_column, sort=True):
        row: dict[str, Any] = {
            group_column: int(group_value),
            "sample_count": int(group["sample_idx"].nunique()),
            "observation_count": int(len(group)),
            "head_count": int(group["head_idx"].nunique()),
        }
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_median"] = float(np.quantile(values, 0.50))
            row[f"{column}_q25"] = float(np.quantile(values, 0.25))
            row[f"{column}_q75"] = float(np.quantile(values, 0.75))
            row[f"{column}_q10"] = float(np.quantile(values, 0.10))
            row[f"{column}_q90"] = float(np.quantile(values, 0.90))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_column)


def validate_kl_columns(metrics: pd.DataFrame, model_name: str) -> None:
    """Validate the finite histogram-KL schema and its defining identities."""

    missing = sorted(set(ALL_KL_METRICS).difference(metrics.columns))
    if missing:
        raise ValueError(f"{model_name} metrics are missing KL columns: {missing}")
    values = metrics[list(ALL_KL_METRICS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{model_name} metrics contain non-finite KL values.")

    divergence_columns = [*KL_PRIMARY_METRICS, *KL_SENSITIVITY_METRICS]
    if (metrics[divergence_columns] < -1e-10).any().any():
        raise ValueError(f"{model_name} metrics contain negative KL values.")
    clipping = metrics[list(KL_CLIPPING_METRICS)]
    if ((clipping < -1e-12) | (clipping > 1.0 + 1e-12)).any().any():
        raise ValueError(f"{model_name} metrics contain invalid clipping fractions.")

    symmetric_error = np.abs(
        metrics[SYMMETRIC_KL]
        - 0.5
        * (
            metrics["kl_real_to_canonical"]
            + metrics["kl_canonical_to_real"]
        )
    )
    if float(symmetric_error.max()) > 1e-10:
        raise ValueError(f"{model_name} metrics violate the symmetric-KL identity.")
    if not np.allclose(
        metrics[SYMMETRIC_KL],
        metrics[PRIMARY_SENSITIVITY_KL],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"{model_name} primary KL differs from its matching sensitivity column."
        )


def kl_estimator_metadata() -> dict[str, Any]:
    return {
        "raw_empirical_kl": (
            "generically infinite because the finite empirical spectral measures "
            "have nonmatching floating-point atoms"
        ),
        "reported_name": "smoothed symmetrized fixed-log-histogram KL",
        "spectrum_transform": (
            f"clip log10(max(sigma, 10^{KL_LOG10_MIN:g})) to "
            f"[{KL_LOG10_MIN:g}, {KL_LOG10_MAX:g}]"
        ),
        "primary_num_bins": KL_PRIMARY_NUM_BINS,
        "primary_uniform_contamination": KL_PRIMARY_UNIFORM_CONTAMINATION,
        "smoothing": "p_tilde=(1-eta)*p+eta/B for every histogram",
        "directional_columns": [
            "kl_real_to_canonical",
            "kl_canonical_to_real",
        ],
        "main_statistic": (
            "symmetric_kl_divergence=0.5*(KL(real||canonical)+"
            "KL(canonical||real))"
        ),
        "probability_ratio_log": "natural logarithm; divergences are in nats",
        "sensitivity_specs": [
            {
                "column": column,
                "num_bins": num_bins,
                "uniform_contamination": eta,
            }
            for column, num_bins, eta in KL_SENSITIVITY_SPECS
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create rebuttal W1 and histogram-KL attention LSD figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", choices=("all", "gpt2", "vit"), default="all")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help="Reuse validated per-head CSVs and regenerate summaries/figures.",
    )
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--max-heads", type=int, default=None)
    parser.add_argument(
        "--gpt2-seq-lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 384, 512, 768, 1024],
    )
    parser.add_argument("--num-text-samples", type=int, default=12)
    parser.add_argument("--wikitext-arrow", type=Path, default=None)
    parser.add_argument("--imagenette-root", type=Path, default=DEFAULT_IMAGENETTE_ROOT)
    parser.add_argument("--num-image-samples", type=int, default=20)
    args = parser.parse_args()

    if not args.gpt2_seq_lengths or any(x <= 0 for x in args.gpt2_seq_lengths):
        parser.error("--gpt2-seq-lengths must contain positive integers.")
    if args.num_text_samples < 0 or args.num_image_samples < 0:
        parser.error("Sample counts must be nonnegative.")
    if args.max_layers is not None and args.max_layers <= 0:
        parser.error("--max-layers must be positive.")
    if args.max_heads is not None and args.max_heads <= 0:
        parser.error("--max-heads must be positive.")
    return args


def main() -> None:
    args = parse_args()
    run_kl_unit_checks()
    set_seed(args.seed)
    configure_plot_style()
    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser()
    figure_dir = args.figure_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    gpt2_csv = output_dir / "gpt2_head_lsd_metrics.csv"
    vit_csv = output_dir / "vit_head_lsd_metrics.csv"
    metadata_path = output_dir / "experiment_metadata.json"
    previous_metadata: dict[str, Any] = {}
    expected_kl_metadata = kl_estimator_metadata()
    if args.reuse_data:
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Cannot validate reused CSVs without the manifest: {metadata_path}"
            )
        previous_metadata = json.loads(metadata_path.read_text())
        if previous_metadata.get("schema_version") != 1:
            raise ValueError("Reused data have an incompatible rebuttal schema.")
        if previous_metadata.get("kl_estimator") != expected_kl_metadata:
            raise ValueError("Reused data use a different KL estimator configuration.")

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "seed": args.seed,
        "device": str(device),
        "configuration": {
            "models": args.models,
            "gpt2_sequence_lengths": sorted(
                {int(x) for x in args.gpt2_seq_lengths}
            ),
            "num_text_samples": args.num_text_samples,
            "num_image_samples": args.num_image_samples,
            "max_layers": args.max_layers,
            "max_heads": args.max_heads,
        },
        "canonical_diagonal_order": "descending score singular values",
        "aggregation": (
            "metrics are computed for matched sample-layer-head tuples before "
            "the descriptive GPT quantiles or ViT layer boxplots"
        ),
        "wasserstein_1_definition": (
            "mean absolute difference between matched descending raw attention "
            "singular values"
        ),
        "kl_estimator": expected_kl_metadata,
        "figure_directory": str(figure_dir),
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
            print("Collecting GPT-2 / WikiText rebuttal LSD metrics...")
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
        validate_kl_columns(gpt2_metrics, "GPT-2")
        metadata["gpt2"] = gpt2_metadata
        generated_paths.append(gpt2_csv)

        generated_paths.extend(
            plot_gpt2_metric(
                gpt2_metrics,
                WASSERSTEIN_1,
                WASSERSTEIN_COLOR,
                r"Per-head $W_1$ between empirical singular-value distributions",
                "gpt2_w1_all_layers_vs_seq_len",
                figure_dir,
            )
        )
        generated_paths.append(
            plot_gpt2_all_and_last_layer_w1(gpt2_metrics, figure_dir)
        )
        generated_paths.extend(
            plot_gpt2_metric(
                gpt2_metrics,
                SYMMETRIC_KL,
                KL_COLOR,
                "Per-head symmetrized histogram KL (nats)",
                "gpt2_histogram_symmetric_kl_all_layers_vs_seq_len",
                figure_dir,
            )
        )

        gpt2_summary = sequence_quantile_summary(gpt2_metrics, SUMMARY_METRICS)
        gpt2_summary_path = output_dir / "gpt2_sequence_summary.csv"
        gpt2_summary.to_csv(gpt2_summary_path, index=False)
        generated_paths.append(gpt2_summary_path)

        document_keys = ["sample_idx", "row_start", "row_end", "seq_len"]
        document_summary = (
            gpt2_metrics.groupby(document_keys, as_index=False)[list(SUMMARY_METRICS)]
            .median()
            .sort_values(["seq_len", "sample_idx"])
        )
        document_summary_path = output_dir / "gpt2_document_medians.csv"
        document_summary.to_csv(document_summary_path, index=False)
        generated_paths.append(document_summary_path)

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
            print("Collecting ViT / Imagenette rebuttal LSD metrics...")
            vit_metrics, vit_metadata = collect_vit_metrics(
                device=device,
                output_csv=vit_csv,
                imagenette_root=args.imagenette_root,
                num_image_samples=args.num_image_samples,
                seed=args.seed,
                max_layers=args.max_layers,
                max_heads=args.max_heads,
            )
        validate_kl_columns(vit_metrics, "ViT")
        metadata["vit"] = vit_metadata
        generated_paths.append(vit_csv)

        generated_paths.extend(
            plot_vit_metric(
                vit_metrics,
                WASSERSTEIN_1,
                WASSERSTEIN_COLOR,
                r"Per-head $W_1$ between empirical singular-value distributions",
                "vit_w1_by_layer",
                figure_dir,
            )
        )
        generated_paths.extend(
            plot_vit_metric(
                vit_metrics,
                SYMMETRIC_KL,
                KL_COLOR,
                "Per-head symmetrized histogram KL (nats)",
                "vit_histogram_symmetric_kl_by_layer",
                figure_dir,
            )
        )

        vit_layer_summary = grouped_quantile_summary(
            vit_metrics,
            "layer_idx",
            SUMMARY_METRICS,
        )
        vit_layer_summary_path = output_dir / "vit_layer_summary.csv"
        vit_layer_summary.to_csv(vit_layer_summary_path, index=False)
        generated_paths.append(vit_layer_summary_path)

        image_keys = [
            "sample_idx",
            "dataset_idx",
            "sample_id",
            "class_synset",
            "seq_len",
            "layer_idx",
        ]
        image_summary = (
            vit_metrics.groupby(image_keys, as_index=False)[list(SUMMARY_METRICS)]
            .median()
            .sort_values(["layer_idx", "sample_idx"])
        )
        image_summary_path = output_dir / "vit_image_layer_medians.csv"
        image_summary.to_csv(image_summary_path, index=False)
        generated_paths.append(image_summary_path)

    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    generated_paths.append(metadata_path)
    print("Generated:")
    for path in generated_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
