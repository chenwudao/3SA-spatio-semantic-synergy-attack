"""
Action 2 & 3: Threshold Sweeping and Soft Correlation for Stage 1.1 Calibration.

Action 2: Sweep top-k thresholds [5%, 10%, 15%, 20%, 30%] and compute
          pairwise + triple IoU for each threshold.

Action 3: Compute Spearman rank correlation and Cosine similarity between
          continuous (non-binarized) attention maps.

Outputs:
  - CSV with per-image IoU at each threshold
  - Threshold sweeping line plot (X=threshold%, Y=IoU)
  - Soft correlation matrix heatmap
  - Summary JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

workspace_tmp = Path(__file__).resolve().parents[1] / ".tmp"
workspace_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(workspace_tmp))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from threesa.config import ExperimentConfig
from threesa.data import ImageTextDataset, build_dataloader
from threesa.models import LLaVASurrogate, DINOv2Surrogate, InternVLSurrogate
from threesa.models.attention import combine_attention_maps, compute_mask_iou, topk_binary_mask


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30]
MODEL_LABELS_REAL = ["llava-1.5-7b-hf", "InternVL2-1B", "dinov2_vits14"]
MODEL_LABELS_MOCK = ["clip_like", "internvl_like", "dinov2_like"]
PAIR_NAMES_REAL = [
    ("llava-1.5-7b-hf", "InternVL2-1B"),
    ("llava-1.5-7b-hf", "dinov2_vits14"),
    ("InternVL2-1B", "dinov2_vits14"),
]
PAIR_NAMES_MOCK = [
    ("clip_like", "internvl_like"),
    ("clip_like", "dinov2_like"),
    ("internvl_like", "dinov2_like"),
]


def compute_iou_at_threshold(
    attention_maps: dict[str, torch.Tensor],
    ratio: float,
    model_labels: list[str],
    pair_names: list[tuple[str, str]],
) -> dict[str, float]:
    """Compute pairwise and triple IoU at a given top-k ratio."""
    masks = {name: topk_binary_mask(attn, ratio) for name, attn in attention_maps.items()}

    results = {}
    # Pairwise
    for left, right in pair_names:
        key = f"{left}__{right}"
        results[key] = compute_mask_iou(masks[left], masks[right]).item()

    # Triple intersection
    triple_mask = masks[model_labels[0]].bool() & masks[model_labels[1]].bool() & masks[model_labels[2]].bool()
    union = masks[model_labels[0]].bool() | masks[model_labels[1]].bool() | masks[model_labels[2]].bool()
    results["triple_iou"] = (triple_mask.sum() / union.sum().clamp_min(1)).item()
    return results


def compute_soft_correlations(
    attention_maps: dict[str, torch.Tensor],
    model_labels: list[str],
    pair_names: list[tuple[str, str]],
) -> dict[str, dict[str, float]]:
    """Compute Spearman rank correlation and Cosine similarity between continuous maps."""
    # Flatten each map to 1D
    flat_maps = {name: attn.flatten().cpu().numpy() for name, attn in attention_maps.items()}

    results = {}
    for left, right in pair_names:
        key = f"{left}__{right}"
        spearman_val, _ = spearmanr(flat_maps[left], flat_maps[right])
        cos_sim = float(np.dot(flat_maps[left], flat_maps[right]) /
                        (np.linalg.norm(flat_maps[left]) * np.linalg.norm(flat_maps[right]) + 1e-12))
        results[key] = {"spearman": float(spearman_val), "cosine": cos_sim}
    return results


def plot_threshold_sweep(
    sweep_data: list[dict],
    output_path: Path,
    model_labels: list[str],
    pair_names: list[tuple[str, str]],
) -> None:
    """Plot IoU vs threshold for all pairs and triple."""
    df = pd.DataFrame(sweep_data)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # Generate colors dynamically
    import matplotlib.cm as cm
    colors = cm.tab10(range(len(pair_names)))

    for i, (left, right) in enumerate(pair_names):
        pair_name = f"{left}__{right}"
        ax.plot(df["threshold_pct"], df[pair_name], marker="o", linewidth=2,
                label=f"{pair_name}", color=colors[i])

    ax.plot(df["threshold_pct"], df["triple_iou"], marker="s", linewidth=2,
            label="Triple Intersection", color="red", linestyle="--")

    ax.set_xlabel("Top-K Threshold (%)", fontsize=12)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title("Cross-Paradigm Attention IoU vs Threshold", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df["threshold_pct"])
    ax.set_xticklabels([f"{int(t)}" for t in df["threshold_pct"]])

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_soft_correlation_matrix(
    soft_data: list[dict],
    output_path: Path,
    model_labels: list[str],
    pair_names: list[tuple[str, str]],
) -> None:
    """Plot Spearman and Cosine correlation matrices."""
    # Average across images
    avg_spearman = {}
    avg_cosine = {}
    for entry in soft_data:
        for key, vals in entry.items():
            if key == "image_id":
                continue
            avg_spearman.setdefault(key, []).append(vals["spearman"])
            avg_cosine.setdefault(key, []).append(vals["cosine"])

    avg_spearman = {k: np.mean(v) for k, v in avg_spearman.items()}
    avg_cosine = {k: np.mean(v) for k, v in avg_cosine.items()}

    # Build symmetric matrices
    names = model_labels
    n = len(names)
    spearman_mat = np.eye(n)
    cosine_mat = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            key = f"{names[i]}__{names[j]}"
            spearman_mat[i, j] = spearman_mat[j, i] = avg_spearman.get(key, 0)
            cosine_mat[i, j] = cosine_mat[j, i] = avg_cosine.get(key, 0)

    # Short names for display
    short_names = [name.split("-")[-1].split("_")[0][:8] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    for ax, mat, title in zip(axes, [spearman_mat, cosine_mat],
                               ["Spearman Rank Correlation", "Cosine Similarity"]):
        sns.heatmap(mat, annot=True, fmt=".3f", xticklabels=short_names,
                    yticklabels=short_names, cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold sweeping and soft correlation for Stage 1.1.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images")
    parser.add_argument("--save-dir", type=str, default="results/stage1_calibration")
    parser.add_argument("--topk-ratios", nargs="+", type=float, default=None,
                        help="Override default thresholds (default: 0.05 0.10 0.15 0.20 0.30)")
    parser.add_argument("--use-mock", action="store_true", help="Use mock surrogates for quick testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(image_size=args.image_size)
    dataset = ImageTextDataset(args.data_dir, image_size=config.image_size, default_prompt=args.prompt)
    dataloader = build_dataloader(dataset, batch_size=1, num_workers=0)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Override thresholds if specified
    thresholds = args.topk_ratios if args.topk_ratios else THRESHOLDS

    print("Loading surrogates...")
    if args.use_mock:
        from threesa.models import MockVisionLanguageSurrogate
        surrogates = [
            MockVisionLanguageSurrogate("clip_like"),
            MockVisionLanguageSurrogate("internvl_like"),
            MockVisionLanguageSurrogate("dinov2_like"),
        ]
        model_labels = MODEL_LABELS_MOCK
        pair_names = PAIR_NAMES_MOCK
    else:
        from threesa.models import LLaVASurrogate, DINOv2Surrogate, InternVLSurrogate
        surrogates = [
            LLaVASurrogate(),
            InternVLSurrogate(),
            DINOv2Surrogate(),
        ]
        model_labels = MODEL_LABELS_REAL
        pair_names = PAIR_NAMES_REAL

    sweep_rows = []  # per-image, per-threshold IoU
    soft_rows = []   # per-image soft correlations
    processed = 0

    for batch in dataloader:
        if args.max_images and processed >= args.max_images:
            break

        image = batch["image"]
        image_id = batch["image_id"][0]
        prompt = batch["prompt"][0] if batch["prompt"][0] else args.prompt

        print(f"[{processed + 1}] {image_id} ...")

        _, attention_maps = combine_attention_maps(surrogates, image, prompt, strategy="weighted_topk")

        # Action 2: Threshold sweeping
        for ratio in thresholds:
            iou_results = compute_iou_at_threshold(attention_maps, ratio, model_labels, pair_names)
            row = {
                "image_id": image_id,
                "threshold_pct": ratio * 100,
                **iou_results,
            }
            sweep_rows.append(row)

        # Action 3: Soft correlations
        soft_results = compute_soft_correlations(attention_maps, model_labels, pair_names)
        soft_rows.append({"image_id": image_id, **soft_results})

        processed += 1

    # --- Save and plot ---
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(save_dir / "threshold_sweep.csv", index=False)

    # Average sweep across images (exclude non-numeric columns)
    numeric_cols = ["threshold_pct"] + [f"{l}__{r}" for l, r in pair_names] + ["triple_iou"]
    sweep_avg = sweep_df[numeric_cols].groupby("threshold_pct", as_index=False).mean()
    plot_threshold_sweep(sweep_avg, save_dir / "threshold_sweep_plot.png", model_labels, pair_names)

    # Soft correlations
    soft_df = pd.DataFrame(soft_rows)
    soft_df.to_csv(save_dir / "soft_correlations.csv", index=False)
    plot_soft_correlation_matrix(soft_rows, save_dir / "soft_correlation_matrix.png", model_labels, pair_names)

    # Summary
    pair_keys = [f"{l}__{r}" for l, r in pair_names]
    summary = {
        "num_images": processed,
        "thresholds": thresholds,
        "model_labels": model_labels,
        "mean_iou_at_thresholds": sweep_avg.set_index("threshold_pct").to_dict(),
        "mean_soft_correlations": {
            key: {
                "spearman": float(np.mean([r[key]["spearman"] for r in soft_rows])),
                "cosine": float(np.mean([r[key]["cosine"] for r in soft_rows])),
            }
            for key in pair_keys
        },
    }
    (save_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nSaved to {save_dir}/")
    print(f"  - threshold_sweep.csv")
    print(f"  - threshold_sweep_plot.png")
    print(f"  - soft_correlations.csv")
    print(f"  - soft_correlation_matrix.png")
    print(f"  - calibration_summary.json")

    # Print key results
    print("\n=== Threshold Sweep (Mean IoU) ===")
    print(sweep_avg.to_string(index=False))
    print("\n=== Soft Correlations (Mean) ===")
    for key, vals in summary["mean_soft_correlations"].items():
        print(f"  {key}: Spearman={vals['spearman']:.4f}, Cosine={vals['cosine']:.4f}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
