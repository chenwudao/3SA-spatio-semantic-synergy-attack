from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torchvision.transforms import functional as TF
from skimage.metrics import structural_similarity

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from threesa.attacks import ThreeSAAttack
from threesa.config import ExperimentConfig
from threesa.data import ImageTextDataset, build_dataloader
from threesa.models import DINOv2Surrogate, InternVLSurrogate, LLaVASurrogate, MockVisionLanguageSurrogate


SURROGATE_CHOICES = ("llava", "internvl", "dinov2")


def sanitize_image_id(image_id: str) -> str:
    return image_id.replace("/", "__").replace("\\", "__").replace(":", "_")


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_float_list(values: list[str] | None, expected_length: int) -> list[float]:
    if values is None:
        return [1.0] * expected_length
    parsed = [float(value) for value in values]
    if len(parsed) != expected_length:
        raise ValueError(f"Expected {expected_length} surrogate weights, got {len(parsed)}")
    return parsed


def build_surrogates(
    surrogate_names: list[str],
    weights: list[float],
    *,
    use_mock: bool,
    device: str,
):
    surrogates = []
    for name, weight in zip(surrogate_names, weights):
        if use_mock:
            mock_name = {
                "llava": "clip_like",
                "internvl": "internvl_like",
                "dinov2": "dinov2_like",
            }[name]
            surrogates.append(MockVisionLanguageSurrogate(mock_name, weight=weight))
            continue

        if name == "llava":
            surrogates.append(LLaVASurrogate(model_id="llava-hf/llava-1.5-7b-hf", weight=weight, device=device))
        elif name == "internvl":
            surrogates.append(InternVLSurrogate(model_id="OpenGVLab/InternVL2-1B", weight=weight, device=device))
        elif name == "dinov2":
            surrogates.append(DINOv2Surrogate(name="dinov2_vits14", weight=weight, device=device))
        else:
            raise ValueError(f"Unsupported surrogate: {name}")
    return surrogates


def tensor_to_image_array(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def compute_image_metrics(clean: torch.Tensor, adv: torch.Tensor) -> dict[str, float]:
    clean_np = tensor_to_image_array(clean)
    adv_np = tensor_to_image_array(adv)
    delta = adv_np - clean_np
    mse = float(np.mean(delta ** 2))
    psnr = float("inf") if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))
    ssim = float(structural_similarity(clean_np, adv_np, channel_axis=2, data_range=1.0))
    return {
        "linf": float(np.abs(delta).max()),
        "l2": float(np.sqrt(np.sum(delta ** 2))),
        "psnr": psnr,
        "ssim": ssim,
    }


def summarize_trace(trace: dict[str, object]) -> dict[str, float]:
    stages = trace.get("stages", [])
    if not stages:
        return {
            "final_mask_mean": 0.0,
            "final_mask_support_ratio": 0.0,
            "mean_conflict_rate": 0.0,
        }

    final_stage = stages[-1]
    conflict_rates: list[float] = []
    for stage in stages:
        for iteration in stage.get("iterations", []):
            conflict_rates.append(float(iteration.get("conflict_rate", 0.0)))

    return {
        "final_mask_mean": float(final_stage.get("mask_mean", 0.0)),
        "final_mask_support_ratio": float(final_stage.get("mask_support_ratio", 0.0)),
        "mean_conflict_rate": float(sum(conflict_rates) / max(len(conflict_rates), 1)),
    }


def compute_surrogate_losses(
    surrogates,
    clean: torch.Tensor,
    adv: torch.Tensor,
    prompt: str,
) -> dict[str, float]:
    loss_metrics: dict[str, float] = {}
    for surrogate in surrogates:
        clean_loss = float(surrogate.compute_loss(clean.clone(), prompt).loss.detach().item())
        adv_loss = float(surrogate.compute_loss(adv.clone(), prompt).loss.detach().item())
        loss_metrics[f"clean_loss_{surrogate.name}"] = clean_loss
        loss_metrics[f"adv_loss_{surrogate.name}"] = adv_loss
        loss_metrics[f"delta_loss_{surrogate.name}"] = adv_loss - clean_loss
    return loss_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 3SA attack experiments on a directory of images.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="Generate unsafe guidance.")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images")
    parser.add_argument("--use-mock", action="store_true", help="Use deterministic mock surrogates for smoke tests")
    parser.add_argument("--surrogates", nargs="+", default=list(SURROGATE_CHOICES), choices=SURROGATE_CHOICES)
    parser.add_argument("--surrogate-weights", nargs="+", default=None)
    parser.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    parser.add_argument("--step-size", type=float, default=2.0 / 255.0)
    parser.add_argument("--stages", type=int, default=10)
    parser.add_argument("--iterations-per-stage", type=int, default=5)
    parser.add_argument("--topk-ratio", type=float, default=0.10)
    parser.add_argument("--mask-strategy", choices=["weighted_topk", "intersection", "union", "global"], default="weighted_topk")
    parser.add_argument("--gradient-strategy", choices=["pcgrad", "vanilla_addition"], default="pcgrad")
    parser.add_argument("--use-soft-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--soft-mask-expansion", type=float, default=1.5)
    parser.add_argument("--soft-mask-sigma", type=float, default=3.0)
    parser.add_argument("--soft-mask-temperature", type=float, default=10.0)
    parser.add_argument("--normalize-gradients", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-pcgrad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attack-name", default="threesa")
    parser.add_argument("--save-traces", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir = output_dir / "images"
    trace_output_dir = output_dir / "traces"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_traces:
        trace_output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(image_size=args.image_size)
    config.attack.device = args.device
    config.attack.epsilon = args.epsilon
    config.attack.step_size = args.step_size
    config.attack.stages = args.stages
    config.attack.iterations_per_stage = args.iterations_per_stage
    config.attack.topk_ratio = args.topk_ratio
    config.attack.mask_strategy = args.mask_strategy
    config.attack.gradient_strategy = args.gradient_strategy
    config.attack.use_soft_mask = args.use_soft_mask
    config.attack.soft_mask_expansion = args.soft_mask_expansion
    config.attack.soft_mask_sigma = args.soft_mask_sigma
    config.attack.soft_mask_temperature = args.soft_mask_temperature
    config.attack.normalize_gradients = args.normalize_gradients
    config.attack.shuffle_pcgrad = args.shuffle_pcgrad

    dataset = ImageTextDataset(args.data_dir, image_size=config.image_size, default_prompt=args.prompt)
    dataloader = build_dataloader(dataset, batch_size=1, num_workers=0)

    surrogate_weights = parse_float_list(args.surrogate_weights, len(args.surrogates))
    surrogates = build_surrogates(
        args.surrogates,
        surrogate_weights,
        use_mock=args.use_mock,
        device=args.device,
    )
    attack = ThreeSAAttack(surrogates=surrogates, config=config.attack)

    rows: list[dict[str, object]] = []
    processed = 0

    for batch in dataloader:
        if args.max_images and processed >= args.max_images:
            break

        image = batch["image"].to(args.device)
        prompt = batch["prompt"][0] if batch["prompt"][0] else args.prompt
        image_id = batch["image_id"][0]
        relative_path = Path(batch["relative_path"][0])

        adv, trace = attack.run(image=image, target_text=prompt)
        metrics = compute_image_metrics(image[0], adv[0])
        metrics.update(summarize_trace(trace))
        metrics.update(compute_surrogate_losses(surrogates, image, adv, prompt))

        save_path = image_output_dir / relative_path.parent / f"{relative_path.stem}_adv.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        TF.to_pil_image(adv.squeeze(0).cpu()).save(save_path)

        row = {
            "image_id": image_id,
            "relative_path": batch["relative_path"][0],
            "prompt": prompt,
            "method": args.attack_name,
            "mask_strategy": args.mask_strategy,
            "gradient_strategy": args.gradient_strategy,
            "use_soft_mask": args.use_soft_mask,
            "surrogates": ",".join(surrogate.name for surrogate in surrogates),
            "adv_path": str(save_path),
            **metrics,
        }
        rows.append(row)

        if args.save_traces:
            trace_path = trace_output_dir / f"{sanitize_image_id(image_id)}.json"
            trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")

        processed += 1
        print(
            f"[{processed}] saved={save_path} "
            f"ssim={metrics['ssim']:.4f} psnr={metrics['psnr']:.2f} "
            f"linf={metrics['linf']:.6f} conflict={metrics['mean_conflict_rate']:.4f}"
        )

    if not rows:
        raise RuntimeError("No images were processed")

    metrics_path = output_dir / "stage2_metrics.json"
    metrics_csv_path = output_dir / "stage2_metrics.csv"
    summary_path = output_dir / "stage2_summary.json"
    metrics_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    summary = {
        "num_images": len(rows),
        "attack_name": args.attack_name,
        "mask_strategy": args.mask_strategy,
        "gradient_strategy": args.gradient_strategy,
        "use_soft_mask": args.use_soft_mask,
        "surrogates": [surrogate.name for surrogate in surrogates],
        "means": {
            key: float(sum(float(row[key]) for row in rows) / len(rows))
            for key in numeric_keys
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"metrics={metrics_path}")
    print(f"metrics_csv={metrics_csv_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
