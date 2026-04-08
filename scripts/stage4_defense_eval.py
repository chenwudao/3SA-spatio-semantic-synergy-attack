from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from threesa.config import ExperimentConfig
from threesa.data import ImageTextDataset, build_dataloader
from threesa.defenses import DefensePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply defense pipeline to adversarial images.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--image-size", type=int, default=336)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(image_size=args.image_size)
    dataset = ImageTextDataset(args.data_dir, image_size=config.image_size, default_prompt=args.prompt)
    dataloader = build_dataloader(dataset, batch_size=2, num_workers=0)
    pipeline = DefensePipeline(config.defense)

    for batch in dataloader:
        defended = pipeline(batch["image"])
        print(f"batch={list(batch['image_id'])} shape={tuple(defended.shape)}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
