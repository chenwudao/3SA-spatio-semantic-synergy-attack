"""
Action 1: Attention Panel Visualization for Stage 1.1 Calibration.

Generates a 1x4 panel for each sampled image:
  [Original] [LLaVA Heatmap] [InternVL Heatmap] [DINOv2 Heatmap]

Purpose: Visual sanity check — ensure attention maps are spatially aligned
and actually highlight meaningful regions (not floating or mosaic artifacts).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

workspace_tmp = Path(__file__).resolve().parents[1] / ".tmp"
workspace_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(workspace_tmp))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from threesa.config import ExperimentConfig
from threesa.data import ImageTextDataset, build_dataloader
from threesa.models import LLaVASurrogate, DINOv2Surrogate, InternVLSurrogate


def overlay_heatmap(image_np: np.ndarray, attention_np: np.ndarray,
                    alpha: float = 0.45, cmap: str = "magma") -> np.ndarray:
    """Overlay a continuous attention heatmap on the original image.

    Args:
        image_np: HxWx3 uint8 RGB image.
        attention_np: HxW float array, values in [0, 1].
        alpha: Overlay transparency.
        cmap: Matplotlib colormap name.

    Returns:
        HxWx3 uint8 RGB composite image.
    """
    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    # attention → RGBA via colormap
    heatmap_rgba = cmap_obj(Normalize(0, 1)(attention_np))  # (H, W, 4)
    heatmap_rgb = heatmap_rgba[:, :, :3]

    # Blend
    composite = (1 - alpha) * image_np.astype(np.float32) / 255.0 + alpha * heatmap_rgb
    composite = np.clip(composite * 255, 0, 255).astype(np.uint8)
    return composite


def build_panel(image_tensor: torch.Tensor, attention_maps: dict[str, torch.Tensor],
                prompt: str, image_id: str) -> matplotlib.figure.Figure:
    """Build a 1x4 figure: [Original, LLaVA, InternVL, DINOv2]."""
    # Convert tensor to numpy RGB
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
    fig.suptitle(f"Image: {image_id}\nPrompt: {prompt}", fontsize=11, y=0.98)

    # Panel 0: Original
    axes[0].imshow(img_np)
    axes[0].set_title("Original Input", fontsize=10)
    axes[0].axis("off")

    # Panels 1-3: Heatmap overlays
    model_labels = ["llava-1.5-7b-hf", "InternVL2-1B", "dinov2_vits14"]
    display_names = ["LLaVA-1.5 (CLIP)", "InternVL2-1B", "DINOv2-S/14"]
    cmaps = ["magma", "viridis", "plasma"]

    for idx, (label, display, cmap_name) in enumerate(zip(model_labels, display_names, cmaps), start=1):
        attn = attention_maps.get(label)
        if attn is not None:
            attn_np = attn.squeeze().cpu().numpy()  # ensure (H, W)
            composite = overlay_heatmap(img_np, attn_np, cmap=cmap_name)
            axes[idx].imshow(composite)
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=Normalize(0, 1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes[idx], fraction=0.046, pad=0.04)
            cbar.set_label("Attention", fontsize=8)
        else:
            axes[idx].text(0.5, 0.5, f"{display}\n(attention not available)",
                           ha="center", va="center", transform=axes[idx].transAxes,
                           fontsize=10, color="red")
        axes[idx].set_title(display, fontsize=10)
        axes[idx].axis("off")

    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate attention heatmap panels for visual inspection.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prompt", default="Describe the weapon.")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--num-samples", type=int, default=5, help="Number of images to visualize")
    parser.add_argument("--output-dir", type=str, default="results/stage1_panels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(image_size=args.image_size)
    dataset = ImageTextDataset(args.data_dir, image_size=config.image_size, default_prompt=args.prompt)
    dataloader = build_dataloader(dataset, batch_size=1, num_workers=0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading surrogates...")
    surrogates = [
        LLaVASurrogate(),
        InternVLSurrogate(),
        DINOv2Surrogate(),
    ]

    from threesa.models.attention import combine_attention_maps

    processed = 0
    for batch in dataloader:
        if processed >= args.num_samples:
            break

        image = batch["image"]
        image_id = batch["image_id"][0]
        prompt = batch["prompt"][0] if batch["prompt"][0] else args.prompt

        print(f"[{processed + 1}/{args.num_samples}] Processing {image_id}...")

        _, attention_maps = combine_attention_maps(surrogates, image, prompt, strategy="weighted_topk")

        fig = build_panel(image[0], attention_maps, prompt, image_id)
        safe_id = image_id.replace("/", "__").replace("\\", "__")
        save_path = output_dir / f"{safe_id}_panel.png"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  Saved → {save_path}")
        processed += 1

    print(f"\nDone. {processed} panels saved to {output_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
