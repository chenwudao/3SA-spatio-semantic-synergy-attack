from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torchvision.transforms import functional as TF

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from threesa.attacks import ThreeSAAttack
from threesa.config import ExperimentConfig
from threesa.data import ImageTextDataset, build_dataloader
from threesa.models import MockVisionLanguageSurrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 3SA attack on a directory of images.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="Generate unsafe guidance.")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(image_size=args.image_size)
    config.attack.device = args.device
    dataset = ImageTextDataset(args.data_dir, image_size=config.image_size, default_prompt=args.prompt)
    dataloader = build_dataloader(dataset, batch_size=1, num_workers=0)

    surrogates = [
        MockVisionLanguageSurrogate("clip_like", weight=1.0),
        MockVisionLanguageSurrogate("internvl_like", weight=1.0),
        MockVisionLanguageSurrogate("dinov2_like", weight=0.7),
    ]
    attack = ThreeSAAttack(surrogates=surrogates, config=config.attack)

    for batch in dataloader:
        image = batch["image"].to(args.device)
        prompt = batch["prompt"][0] if batch["prompt"][0] else args.prompt
        adv, _ = attack.run(image=image, target_text=prompt)
        save_path = output_dir / f"{batch['image_id'][0]}_adv.png"
        TF.to_pil_image(adv.squeeze(0).cpu()).save(save_path)
        print(f"saved={save_path}")


if __name__ == "__main__":
    main()
