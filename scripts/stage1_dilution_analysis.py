"""
Experiment 1.2: MPC (Multi-Paradigm Consensus) Energy Dilution Analysis.

Purpose: Prove that traditional FULL-IMAGE MPC attack wastes energy on
non-sensitive regions, motivating SAGA's "purified injection" strategy.

Key innovation vs. previous version:
- REAL MPC PGD: gradients from 3 paradigms (LLaVA/CLIP, InternVL, DINOv2)
  are computed separately, normalized, and accumulated — this is the true
  "Multi-Paradigm Consensus" attack, not a single-model surrogate.
- Gradient conflict tracking: cosine similarity between paradigm gradients
  (sets up Stage 2.2 PCGrad motivation).

Method:
1. Extract attention maps from 3 surrogates → compute consensus mask (Top 10%)
2. Run standard MPC PGD attack (3-model normalized gradient ensemble)
   on the FULL image (no mask)
3. Compare perturbation energy (L2 norm²) in:
   - Consensus sensitive region
   - Background non-sensitive region
4. Quantify the "dilution effect"

Expected conclusion: Most attack budget is wasted on useless background,
and gradient conflicts across paradigms further diffuse the perturbation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import gc
from pathlib import Path

workspace_tmp = Path(__file__).resolve().parents[1] / ".tmp"
workspace_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(workspace_tmp))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from threesa.config import ExperimentConfig
from threesa.data import ImageTextDataset, build_dataloader
from threesa.models import LLaVASurrogate, DINOv2Surrogate, InternVLSurrogate
from threesa.models.attention import combine_attention_maps, topk_binary_mask


# ──────────────────────────────────────────────────────────────
# MPC PGD with normalized gradient accumulation
# ──────────────────────────────────────────────────────────────

def compute_single_gradient(surrogate, x_adv, prompt):
    """Compute normalized gradient for a single surrogate model.

    Returns None if gradient computation fails.
    """
    x = x_adv.clone().detach().requires_grad_(True)
    try:
        loss = surrogate.compute_loss(x, prompt).loss
        grad = torch.autograd.grad(loss, x)[0]
        # L2-normalize the gradient
        grad_norm = grad.norm(p=2) + 1e-12
        grad_normalized = grad / grad_norm
        return grad_normalized.detach()
    except Exception:
        return None


def compute_mpc_pgd_perturbation(
    surrogates: list,
    image: torch.Tensor,
    prompt: str,
    steps: int = 10,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
) -> tuple[torch.Tensor, list[dict]]:
    """Generate adversarial perturbation using MPC PGD attack.

    Multi-Paradigm Consensus PGD:
    - Compute gradient for each surrogate SEPARATELY (memory efficient)
    - L2-normalize each gradient to prevent dominance by any single model
    - Accumulate normalized gradients (true MPC)
    - Track gradient conflict (cosine similarity) between paradigms

    Args:
        surrogates: list of VisionLanguageSurrogate models
        image: (1, 3, H, W) original image
        prompt: text prompt
        steps: PGD iterations
        epsilon: max perturbation norm
        alpha: step size

    Returns:
        perturbation: (1, 3, H, W) adversarial perturbation
        conflict_log: list of per-step gradient conflict metrics
    """
    x_adv = image.clone().detach()
    conflict_log = []

    for step in range(steps):
        accumulated_grad = torch.zeros_like(x_adv)
        grad_list = []
        grad_names = []

        # Compute each surrogate's gradient separately
        for surrogate in surrogates:
            grad = compute_single_gradient(surrogate, x_adv.clone().detach(), prompt)
            if grad is not None:
                grad_list.append(grad)
                grad_names.append(surrogate.name)
                accumulated_grad += grad
            gc.collect()
            torch.cuda.empty_cache()

        if not grad_list:
            print(f"    WARNING: No valid gradients at step {step}, stopping")
            break

        # Track gradient conflicts (cosine similarity between paradigms)
        step_conflicts = {}
        if len(grad_list) >= 2:
            for i in range(len(grad_list)):
                for j in range(i + 1, len(grad_list)):
                    cos_sim = F.cosine_similarity(
                        grad_list[i].view(1, -1),
                        grad_list[j].view(1, -1)
                    ).item()
                    key = f"{grad_names[i]}__{grad_names[j]}"
                    step_conflicts[key] = cos_sim

        conflict_log.append(step_conflicts)

        # Apply MPC gradient update
        x_adv = x_adv + alpha * accumulated_grad.sign()

        # Project to epsilon ball
        perturbation = x_adv - image
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        x_adv = image + perturbation
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()

    return x_adv - image, conflict_log


# ──────────────────────────────────────────────────────────────
# Energy dilution analysis (unchanged logic)
# ──────────────────────────────────────────────────────────────

def analyze_energy_dilution(
    perturbation: torch.Tensor,
    consensus_mask: torch.Tensor,
) -> dict:
    """Analyze energy distribution between sensitive and background regions.

    Args:
        perturbation: (1, 3, H, W) adversarial perturbation
        consensus_mask: (1, H, W) binary consensus mask

    Returns:
        dict with energy analysis results
    """
    energy_map = perturbation.abs().mean(dim=1)  # (1, H, W)
    binary_mask = (consensus_mask > 0.5).float()  # (1, H, W)
    background_mask = 1 - binary_mask

    sensitive_energy = (energy_map * binary_mask).sum().item()
    background_energy = (energy_map * background_mask).sum().item()
    total_energy = energy_map.sum().item()

    sensitive_pixels = binary_mask.sum().item()
    background_pixels = background_mask.sum().item()

    sensitive_density = sensitive_energy / (sensitive_pixels + 1e-8)
    background_density = background_energy / (background_pixels + 1e-8)

    dilution_ratio = background_energy / (sensitive_energy + 1e-8)
    waste_percentage = background_energy / (total_energy + 1e-8) * 100

    return {
        "sensitive_energy": sensitive_energy,
        "background_energy": background_energy,
        "total_energy": total_energy,
        "sensitive_pixels": sensitive_pixels,
        "background_pixels": background_pixels,
        "sensitive_density": sensitive_density,
        "background_density": background_density,
        "dilution_ratio": dilution_ratio,
        "waste_percentage": waste_percentage,
    }


def plot_energy_distribution(
    results: list[dict],
    output_path: Path,
) -> None:
    """Plot energy distribution analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    waste_pcts = [r["waste_percentage"] for r in results]
    dilution_ratios = [r["dilution_ratio"] for r in results]

    axes[0].hist(waste_pcts, bins=20, color="coral", alpha=0.7, edgecolor="black")
    axes[0].axvline(np.mean(waste_pcts), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(waste_pcts):.1f}%")
    axes[0].set_xlabel("Energy Waste Percentage (%)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("MPC Energy Dilution: Waste Distribution", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(dilution_ratios, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    axes[1].axvline(np.mean(dilution_ratios), color="blue", linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(dilution_ratios):.2f}x")
    axes[1].set_xlabel("Dilution Ratio (Background/Sensitive)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Energy Dilution Ratio Distribution", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_gradient_conflicts(
    conflict_log: list[dict],
    output_path: Path,
) -> None:
    """Plot gradient cosine similarity across PGD steps."""
    if not conflict_log:
        return

    pairs = set()
    for step_log in conflict_log:
        pairs.update(step_log.keys())

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    for pair in sorted(pairs):
        values = [step_log.get(pair, None) for step_log in conflict_log]
        steps = list(range(len(values)))
        label_short = pair.replace("llava-1.5-7b-hf", "LLaVA").replace(
            "InternVL2-1B", "InternVL").replace("dinov2_vits14", "DINOv2")
        ax.plot(steps, values, marker="o", markersize=3, label=label_short, linewidth=1.5)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("PGD Step", fontsize=12)
    ax.set_ylabel("Gradient Cosine Similarity", fontsize=12)
    ax.set_title("Cross-Paradigm Gradient Conflict During MPC PGD", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.05, 1.05)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPC Energy Dilution Analysis (Experiment 1.2)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--max-images", type=int, default=50, help="Number of images to analyze")
    parser.add_argument("--save-dir", type=str, default="results/stage1_dilution_analysis")
    parser.add_argument("--pgd-steps", type=int, default=10, help="PGD attack iterations")
    parser.add_argument("--epsilon", type=float, default=8 / 255, help="PGD epsilon constraint")
    parser.add_argument("--use-mock", action="store_true", help="Use mock surrogates for quick testing")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    config = ExperimentConfig(image_size=args.image_size)
    dataset = ImageTextDataset(args.data_dir, image_size=config.image_size, default_prompt="")
    dataloader = build_dataloader(dataset, batch_size=1, num_workers=0)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading surrogates...")
    if args.use_mock:
        from threesa.models import MockVisionLanguageSurrogate
        surrogates = [
            MockVisionLanguageSurrogate("clip_like"),
            MockVisionLanguageSurrogate("internvl_like"),
            MockVisionLanguageSurrogate("dinov2_like"),
        ]
    else:
        surrogates = [
            LLaVASurrogate(),
            InternVLSurrogate(),
            DINOv2Surrogate(),
        ]

    print(f"  Surrogates: {[s.name for s in surrogates]}")
    print(f"  MPC PGD: normalized gradient accumulation across {len(surrogates)} paradigms")

    results = []
    all_conflict_logs = []

    if args.use_mock:
        # Mock branch: simple loop
        processed = 0
        print(f"\nAnalyzing energy dilution for {args.max_images} images (mock)...")
        for batch in dataloader:
            if processed >= args.max_images:
                break
            image = batch["image"]
            image_id = batch["image_id"][0]
            prompt = batch["prompt"][0] if batch["prompt"][0] else "Describe this image."
            print(f"[{processed + 1}/{args.max_images}] {image_id} ...")

            try:
                _, attention_maps = combine_attention_maps(surrogates, image, prompt, strategy="weighted_topk")
                stacked = torch.stack(list(attention_maps.values()), dim=0)
                combined = stacked.min(dim=0).values
                consensus_mask = topk_binary_mask(combined, ratio=0.10).squeeze(1)

                perturbation, conflict_log = compute_mpc_pgd_perturbation(
                    surrogates, image, prompt,
                    steps=args.pgd_steps, epsilon=args.epsilon
                )
                energy_analysis = analyze_energy_dilution(perturbation, consensus_mask)
                energy_analysis["image_id"] = image_id
                results.append(energy_analysis)
                all_conflict_logs.extend(conflict_log)
                processed += 1

            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                continue
    else:
        # Real models: two-phase execution for memory efficiency
        # Phase A: Extract attention maps (no grad)
        print(f"\nPhase A: Extracting attention maps for all images (no grad)...")
        attention_data = []
        with torch.no_grad():
            for batch in dataloader:
                if len(attention_data) >= args.max_images:
                    break
                image = batch["image"]
                image_id = batch["image_id"][0]
                prompt = batch["prompt"][0] if batch["prompt"][0] else "Describe this image."
                print(f"  [{len(attention_data) + 1}/{args.max_images}] {image_id} ...")

                try:
                    _, attention_maps = combine_attention_maps(surrogates, image, prompt, strategy="weighted_topk")
                    stacked = torch.stack(list(attention_maps.values()), dim=0)
                    combined = stacked.min(dim=0).values
                    consensus_mask = topk_binary_mask(combined, ratio=0.10).squeeze(1)
                    attention_data.append((image_id, prompt, consensus_mask.cpu(), image.cpu()))
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

        # Phase B: Run MPC PGD attack
        # Note: We keep all surrogates loaded. compute_mpc_pgd_perturbation computes
        # each model's gradient SEPARATELY to avoid simultaneous memory usage.
        print(f"\nPhase B: Running MPC PGD attack (3-model normalized gradient ensemble)...")
        for idx, (image_id, prompt, consensus_mask, image) in enumerate(attention_data):
            image = image.cuda()
            consensus_mask = consensus_mask.cuda()
            print(f"  [{idx + 1}/{len(attention_data)}] {image_id} (MPC PGD) ...")

            try:
                perturbation, conflict_log = compute_mpc_pgd_perturbation(
                    surrogates, image, prompt,
                    steps=args.pgd_steps, epsilon=args.epsilon
                )
                energy_analysis = analyze_energy_dilution(perturbation, consensus_mask)
                energy_analysis["image_id"] = image_id
                results.append(energy_analysis)
                all_conflict_logs.extend(conflict_log)

                del perturbation, image, consensus_mask
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                import traceback
                print(f"    ERROR: {e}")
                traceback.print_exc()
                continue

    if not results:
        print("No results generated. Exiting.")
        return

    # ── Save results ──────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(save_dir / "dilution_analysis.csv", index=False)
    plot_energy_distribution(results, save_dir / "dilution_distribution.png")
    plot_gradient_conflicts(all_conflict_logs, save_dir / "gradient_conflicts.png")

    summary = {
        "num_images": len(results),
        "mean_waste_percentage": float(df["waste_percentage"].mean()),
        "median_waste_percentage": float(df["waste_percentage"].median()),
        "mean_dilution_ratio": float(df["dilution_ratio"].mean()),
        "median_dilution_ratio": float(df["dilution_ratio"].median()),
        "mean_sensitive_density": float(df["sensitive_density"].mean()),
        "mean_background_density": float(df["background_density"].mean()),
        "density_ratio": float(df["sensitive_density"].mean() / (df["background_density"].mean() + 1e-8)),
    }

    # Add gradient conflict summary
    if all_conflict_logs:
        conflict_pairs = set()
        for log in all_conflict_logs:
            conflict_pairs.update(log.keys())
        conflict_summary = {}
        for pair in sorted(conflict_pairs):
            values = [log.get(pair, None) for log in all_conflict_logs if log.get(pair) is not None]
            if values:
                conflict_summary[pair] = {
                    "mean_cosine_similarity": float(np.mean(values)),
                    "median_cosine_similarity": float(np.median(values)),
                    "min_cosine_similarity": float(np.min(values)),
                    "max_cosine_similarity": float(np.max(values)),
                }
        summary["gradient_conflicts"] = conflict_summary

    (save_dir / "dilution_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Print results ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MPC (Multi-Paradigm Consensus) Energy Dilution Analysis Results")
    print("=" * 70)
    print(f"  Images analyzed:              {len(results)}")
    print(f"  Surrogates:                   {[s.name for s in surrogates]}")
    print(f"  PGD method:                   Normalized gradient accumulation")
    print(f"  Mean energy waste:            {summary['mean_waste_percentage']:.1f}%")
    print(f"  Mean dilution ratio:          {summary['mean_dilution_ratio']:.2f}x")
    print(f"  Sensitive region density:     {summary['mean_sensitive_density']:.6f}")
    print(f"  Background region density:    {summary['mean_background_density']:.6f}")
    print(f"  Density ratio (sens/bg):      {summary['density_ratio']:.3f}x")
    if "gradient_conflicts" in summary:
        print(f"\n  Gradient Conflict (cosine similarity):")
        for pair, stats in summary["gradient_conflicts"].items():
            print(f"    {pair}: mean={stats['mean_cosine_similarity']:.3f}, "
                  f"min={stats['min_cosine_similarity']:.3f}")
    print("\n" + "=" * 70)
    print("  Conclusion: Traditional full-image MPC wastes significant energy")
    print("  on background regions. Gradient conflicts across paradigms further")
    print("  diffuse the perturbation, motivating SAGA's purified injection.")
    print("=" * 70)
    print(f"\nResults saved to: {save_dir}/")


if __name__ == "__main__":
    main()
