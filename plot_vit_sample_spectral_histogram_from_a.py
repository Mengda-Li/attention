from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms


SEED = 233
SAMPLE_IDX = 1
NUM_BINS = 100
RESULTS_DIR = Path("results/per_sample")
OUTPUT_PATH = RESULTS_DIR / f"vit_sample{SAMPLE_IDX}_spectral_histogram_from_a.svg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_ln1_hook(storage: list[torch.Tensor]):
    def hook(module, inputs, output):
        _ = module
        _ = inputs
        storage.append(output.detach())

    return hook


def collect_layer_norm_outputs(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> list[torch.Tensor]:
    layer_norm_outputs: list[torch.Tensor] = []
    hooks = [
        encoder_layer.ln_1.register_forward_hook(make_ln1_hook(layer_norm_outputs))
        for encoder_layer in model.encoder.layers
    ]
    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        for hook in hooks:
            hook.remove()
    return layer_norm_outputs


def compute_layerwise_singular_values_from_a_style_attention(
    model: torch.nn.Module,
    layer_norm_outputs: list[torch.Tensor],
) -> dict[int, np.ndarray]:
    layerwise_singular_values: dict[int, np.ndarray] = {}

    for layer_idx, encoder_layer in enumerate(model.encoder.layers):
        if layer_idx >= len(layer_norm_outputs):
            raise RuntimeError(f"Missing LayerNorm output for layer {layer_idx}.")

        attention_layer = encoder_layer.self_attention
        in_proj_weight = attention_layer.in_proj_weight
        embed_dim = attention_layer.embed_dim

        w_q = in_proj_weight[:embed_dim, :]
        w_k = in_proj_weight[embed_dim : 2 * embed_dim, :]
        w_v = in_proj_weight[2 * embed_dim :, :]

        x_normalized = layer_norm_outputs[layer_idx][0].unsqueeze(0)

        with torch.no_grad():
            q = x_normalized @ w_q.T
            k = x_normalized @ w_k.T
            v = x_normalized @ w_v.T

            _, attn_output_weights = attention_layer(
                q,
                k,
                v,
                need_weights=True,
            )

        if attn_output_weights.dim() == 4:
            attn_avg = attn_output_weights[0].mean(dim=0)
        elif attn_output_weights.dim() == 3:
            attn_avg = attn_output_weights[0]
        else:
            raise RuntimeError(
                "Unexpected attention weight shape "
                f"{tuple(attn_output_weights.shape)} at layer {layer_idx}."
            )

        singular_values = torch.linalg.svdvals(attn_avg).detach().cpu().numpy()
        layerwise_singular_values[layer_idx] = np.maximum(singular_values, 0.0)

    return layerwise_singular_values


def plot_layerwise_histograms(
    layerwise_singular_values: dict[int, np.ndarray],
    save_path: Path,
    sample_name: str,
    class_name: str,
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

    # fig.suptitle(
    #     "ViT-B/16 attention matrix singular values from a.py-style calculation\n"
    #     f"Imagenette val sample {SAMPLE_IDX} ({sample_name}, class={class_name})",
    #     y=0.98,
    # )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(SEED)

    transform = transforms.Compose(
        [
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    imagenette = datasets.Imagenette(root="./data", split="val", transform=transform)
    sample_image, sample_label = imagenette[SAMPLE_IDX]
    sample_path = ""
    if hasattr(imagenette, "samples") and SAMPLE_IDX < len(imagenette.samples):
        sample_path = str(imagenette.samples[SAMPLE_IDX][0])
    sample_name = Path(sample_path).name if sample_path else f"sample_{SAMPLE_IDX}"
    class_name = imagenette.classes[sample_label] if hasattr(imagenette, "classes") else str(sample_label)

    model = torchvision.models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")
    model.to(DEVICE)
    model.eval()

    input_tensor = sample_image.unsqueeze(0).to(DEVICE)
    layer_norm_outputs = collect_layer_norm_outputs(model, input_tensor)
    layerwise_singular_values = compute_layerwise_singular_values_from_a_style_attention(
        model,
        layer_norm_outputs,
    )
    plot_layerwise_histograms(layerwise_singular_values, OUTPUT_PATH, sample_name, class_name)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
