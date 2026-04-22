"""
Attention Panel Visualization for Stage 1.1 Calibration (Sampled from Mixed Experiment).

Generates a 1x4 panel for each sampled image from the mixed experiment:
  [Original] [LLaVA Heatmap] [InternVL Heatmap] [DINOv2 Heatmap]

Samples are stratified across all 13 categories with L1/L2/L3 prompt coverage.
Purpose: Visual sanity check — ensure attention maps are spatially aligned
and actually highlight meaningful regions (not floating or mosaic artifacts).
"""
from __future__ import annotations

import argparse
import csv
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
from threesa.models.attention import combine_attention_maps


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
                prompt: str, image_id: str, level: str) -> matplotlib.figure.Figure:
    """Build a 1x4 figure: [Original, LLaVA, InternVL, DINOv2]."""
    # Convert tensor to numpy RGB
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
    fig.suptitle(f"Image: {image_id}  |  Prompt Level: {level}\nPrompt: {prompt}", 
                 fontsize=11, y=0.98)

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


def load_sample_list(sample_file: Path) -> list[tuple[str, str]]:
    """Load image_id,prompt pairs from sample file."""
    samples = []
    with sample_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                image_id = row[0].strip()
                prompt = ",".join(row[1:]).strip()  # prompt may contain commas
                samples.append((image_id, prompt))
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate attention heatmap panels for visual inspection (sampled from mixed experiment)."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sample-file", required=True, 
                        help="Path to sample file (image_id,prompt per line)")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--output-dir", type=str, default="results/stage1_attention_panels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(image_size=args.image_size)
    
    # Load samples from file
    sample_file = Path(args.sample_file)
    if not sample_file.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_file}")
    
    samples = load_sample_list(sample_file)
    print(f"Loaded {len(samples)} samples from {sample_file}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading surrogates...")
    surrogates = [
        LLaVASurrogate(),
        InternVLSurrogate(),
        DINOv2Surrogate(),
    ]

    # Classify prompt level for display
    def classify_prompt(prompt: str) -> str:
        prompt_lower = prompt.lower()
        if any(k in prompt_lower for k in ['step-by-step', 'instructions', 'implement', 'carry out', 'cause the harm', 
                                            'inflict', 'execute', 'scam', 'explicit', 'propaganda', 
                                            'private information', 'legal advice', 'recommendation', 
                                            'medical advice', 'disinformation', 'persuasive', 'working implementation']):
            return 'L3'
        elif any(k in prompt_lower for k in ['depicted', 'shown', 'illegal activity', 'speech', 'harm', 
                                              'fraud', 'political', 'privacy', 'legal', 'financial', 
                                              'health', 'government', 'economic', 'adult content',
                                              'malware', 'code', 'software', 'deception', 'criminal']):
            return 'L2'
        else:
            return 'L1'

    processed = 0
    for image_id, prompt in samples:
        print(f"[{processed + 1}/{len(samples)}] Processing {image_id}...")
        
        # Load single image with specific prompt
        from PIL import Image
        from torchvision import transforms
        
        # Find image path
        image_path = Path(args.data_dir) / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = Path(args.data_dir) / f"{image_id}.png"
        if not image_path.exists():
            # Try to find in subdirectories
            image_path = None
            for p in Path(args.data_dir).rglob(f"{Path(image_id).name}.*"):
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    image_path = p
                    break
        
        if image_path is None or not image_path.exists():
            print(f"  WARNING: Image not found: {image_id}, skipping...")
            continue
        
        # Load and transform image
        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
        ])
        
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Extract attention maps
        _, attention_maps = combine_attention_maps(surrogates, image_tensor, prompt, strategy="weighted_topk")

        # Classify prompt level
        level = classify_prompt(prompt)
        
        # Build and save panel
        fig = build_panel(image_tensor[0], attention_maps, prompt, image_id, level)
        safe_id = image_id.replace("/", "__").replace("\\", "__")
        save_path = output_dir / f"{safe_id}_{level}_panel.png"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  Saved → {save_path}")
        processed += 1

    print(f"\nDone. {processed} panels saved to {output_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
