from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from transformers import GPT2Model, GPT2Tokenizer


SEED = 233
NUM_BINS = 100
RESULTS_DIR = Path("results/per_sample")
OUTPUT_PATH = RESULTS_DIR / "gpt2_longest_spectral_histogram.svg"
HF_DATASETS_CACHE = Path.home() / ".cache/huggingface/datasets"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cached_wikitext_validation() -> Dataset:
    matches = sorted(
        HF_DATASETS_CACHE.glob(
            "Salesforce___wikitext/wikitext-103-raw-v1/*/*/wikitext-validation.arrow"
        )
    )
    if not matches:
        raise FileNotFoundError(
            "Could not find cached WikiText-103 validation Arrow file under ~/.cache/huggingface/datasets."
        )
    return Dataset.from_file(str(matches[-1]))


def select_longest_text(
    tokenizer: GPT2Tokenizer,
) -> tuple[list[int], dict[str, str | int]]:
    validation = load_cached_wikitext_validation()
    model_max_length = int(tokenizer.model_max_length)

    best_token_count = -1
    best_row_idx = -1
    best_text = ""
    best_input_ids: list[int] = []

    for row_idx, example in enumerate(validation):
        text = example["text"]
        if not text.strip():
            continue

        input_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=model_max_length,
        )["input_ids"]
        token_count = len(input_ids)

        if token_count > best_token_count:
            best_token_count = token_count
            best_row_idx = row_idx
            best_text = text
            best_input_ids = input_ids

    if best_row_idx < 0 or not best_input_ids:
        raise RuntimeError("No usable WikiText validation sample was found.")

    metadata: dict[str, str | int] = {
        "row_idx": best_row_idx,
        "token_count": best_token_count,
        "text_preview": " ".join(best_text.strip().split())[:120],
    }
    return best_input_ids, metadata


def singular_values_from_attention_matrix(attention_matrix: torch.Tensor) -> np.ndarray:
    singular_values = torch.linalg.svdvals(attention_matrix).detach().cpu().numpy()
    return np.maximum(singular_values, 0.0)


def collect_layerwise_singular_values(
    model: GPT2Model,
    input_ids: list[int],
    device: str,
) -> dict[int, np.ndarray]:
    tensor_input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=tensor_input_ids, output_attentions=True)

    if outputs.attentions is None:
        raise RuntimeError("GPT-2 forward did not return attention weights.")

    layerwise_singular_values: dict[int, np.ndarray] = {}
    for layer_idx, attention_weights in enumerate(outputs.attentions):
        if attention_weights.ndim != 4:
            raise RuntimeError(
                f"Expected GPT-2 attention weights with 4 dims, got shape {tuple(attention_weights.shape)}."
            )

        attn_avg = attention_weights[0].mean(dim=0)
        layerwise_singular_values[layer_idx] = singular_values_from_attention_matrix(attn_avg)

    return layerwise_singular_values


def plot_layerwise_histograms(
    layerwise_singular_values: dict[int, np.ndarray],
    save_path: Path,
    row_idx: int,
    token_count: int,
) -> None:
    n_layers = len(layerwise_singular_values)
    n_rows = 3
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), dpi=150)
    axes = axes.ravel()

    for layer_idx, ax in enumerate(axes):
        if layer_idx >= n_layers:
            ax.axis("off")
            continue

        singular_values = layerwise_singular_values[layer_idx]
        ax.hist(
            singular_values,
            bins=NUM_BINS,
            color="#4c72b0",
            edgecolor="black",
            alpha=0.8,
        )
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Singular value")
        ax.set_ylabel("Count")

    fig.suptitle(
        "GPT-2 attention matrix singular values from averaged model attention output\n"
        f"WikiText validation row {row_idx}, seq_len={token_count}",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(SEED)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        local_files_only=True,
    )
    model.to(DEVICE)
    model.eval()

    input_ids, metadata = select_longest_text(tokenizer)
    layerwise_singular_values = collect_layerwise_singular_values(model, input_ids, DEVICE)
    plot_layerwise_histograms(
        layerwise_singular_values,
        OUTPUT_PATH,
        row_idx=int(metadata["row_idx"]),
        token_count=int(metadata["token_count"]),
    )

    print(f"Selected validation row {metadata['row_idx']} with {metadata['token_count']} tokens.")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
