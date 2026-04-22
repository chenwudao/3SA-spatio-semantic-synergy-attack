from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

workspace_tmp = Path(__file__).resolve().parents[1] / ".tmp"
workspace_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(workspace_tmp))
os.environ.setdefault("TEMP", str(workspace_tmp))
os.environ.setdefault("TMP", str(workspace_tmp))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(workspace_tmp / "torchinductor"))

import torch
import numpy as np
import pandas as pd
import cv2

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from threesa.config import ExperimentConfig
from threesa.data import ImageTextDataset, build_dataloader
from threesa.models import MockVisionLanguageSurrogate
from threesa.models.attention import combine_attention_maps, compute_mask_iou, topk_binary_mask


def sanitize_image_id(image_id: str) -> str:
    return image_id.replace("/", "__").replace("\\", "__").replace(":", "_")


def save_heatmap(image_tensor: torch.Tensor, attention_tensor: torch.Tensor, save_path: Path) -> None:
    # Convert image tensor (C, H, W) to numpy (H, W, C)
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Convert attention tensor (1, H, W) to numpy (H, W)
    attn_np = attention_tensor[0].cpu().numpy()

    # Normalize attention and apply color map
    attn_norm = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
    attn_heat = np.uint8(255 * attn_norm)
    heatmap = cv2.applyColorMap(attn_heat, cv2.COLORMAP_JET)

    # Superimpose heatmap on image
    superimposed_img = cv2.addWeighted(heatmap, 0.4, img_bgr, 0.6, 0)

    # Save image
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), superimposed_img)


def save_mask(mask_tensor: torch.Tensor, save_path: Path) -> None:
    mask_np = (mask_tensor[0].cpu().numpy() * 255).astype(np.uint8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), mask_np)


def build_surrogates(use_mock: bool):
    if use_mock:
        return [
            MockVisionLanguageSurrogate("clip_like"),
            MockVisionLanguageSurrogate("internvl_like"),
            MockVisionLanguageSurrogate("dinov2_like"),
        ]

    from threesa.models import DINOv2Surrogate, InternVLSurrogate, LLaVASurrogate

    return [
        LLaVASurrogate(model_id="llava-hf/llava-1.5-7b-hf"),
        DINOv2Surrogate(name="dinov2_vits14"),
        InternVLSurrogate(model_id="OpenGVLab/InternVL2-1B"),
    ]


def pairwise_iou(masks: dict[str, torch.Tensor]) -> dict[str, float]:
    names = list(masks.keys())
    results: dict[str, float] = {}
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            value = compute_mask_iou(masks[left_name], masks[right_name]).item()
            results[f"{left_name}__{right_name}"] = value
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-paradigm attention IoU analysis.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--topk-ratio", type=float, default=0.10)
    parser.add_argument("--save-dir", type=str, default="results/stage1")
    parser.add_argument("--use-mock", action="store_true", help="Use mock surrogates instead of real ones")
    parser.add_argument("--max-images", type=int, default=0, help="Limit the number of images for a quick sanity run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(image_size=args.image_size)
    dataset = ImageTextDataset(args.data_dir, image_size=config.image_size, default_prompt=args.prompt)
    dataloader = build_dataloader(dataset, batch_size=1, num_workers=0)
    
    save_dir = Path(args.save_dir)
    heatmap_dir = save_dir / "heatmaps"
    mask_dir = save_dir / "masks"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    surrogates = build_surrogates(args.use_mock)

    results = []
    processed = 0

    for batch in dataloader:
        if args.max_images and processed >= args.max_images:
            break
        image = batch["image"]
        image_id = batch["image_id"][0]
        prompt = batch["prompt"][0] if batch["prompt"][0] else args.prompt
        safe_image_id = sanitize_image_id(image_id)
        print(f"Processing {image_id}...")
        
        _, attention_maps = combine_attention_maps(surrogates, image, prompt, strategy="weighted_topk")
        
        masks: dict[str, torch.Tensor] = {}
        for name, attn in attention_maps.items():
            # Save individual heatmaps
            save_heatmap(image[0], attn[0], heatmap_dir / f"{safe_image_id}_{name}_heatmap.jpg")
            mask = topk_binary_mask(attn, args.topk_ratio)
            save_mask(mask[0], mask_dir / f"{safe_image_id}_{name}_mask.png")
            masks[name] = mask
        
        iou = compute_mask_iou(*masks.values())
        pairwise = pairwise_iou(masks)
        
        print(f"image_id={image_id} prompt={prompt!r} iou={iou.item():.4f}")
        row = {
            "image_id": image_id,
            "relative_path": batch["relative_path"][0],
            "prompt": prompt,
            "iou": iou.item(),
        }
        row.update(pairwise)
        results.append(row)
        processed += 1

    df = pd.DataFrame(results)
    csv_path = save_dir / "stage1_iou_results.csv"
    df.to_csv(csv_path, index=False)
    summary = {
        "num_images": int(len(df)),
        "mean_iou": float(df["iou"].mean()) if not df.empty else 0.0,
        "std_iou": float(df["iou"].std(ddof=0)) if not df.empty else 0.0,
        "topk_ratio": args.topk_ratio,
        "use_mock": bool(args.use_mock),
        "surrogates": [surrogate.name for surrogate in surrogates],
    }
    pairwise_columns = [column for column in df.columns if "__" in column]
    for column in pairwise_columns:
        summary[f"mean_{column}"] = float(df[column].mean())

    summary_path = save_dir / "stage1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results to {csv_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
