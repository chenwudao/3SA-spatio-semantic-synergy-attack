from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class Sample:
    image_id: str
    image_path: Path
    relative_path: Path
    prompt: str


class ImageTextDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = 336,
        default_prompt: str = "",
        recursive: bool = True,
        skip_mirrored_root_subdir: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.default_prompt = default_prompt
        self.recursive = recursive
        self.skip_mirrored_root_subdir = skip_mirrored_root_subdir
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.samples = self._discover_samples()

    def _should_skip_path(self, image_path: Path) -> bool:
        if not self.skip_mirrored_root_subdir:
            return False
        try:
            relative_path = image_path.relative_to(self.data_dir)
        except ValueError:
            return False
        return bool(relative_path.parts) and relative_path.parts[0] == self.data_dir.name

    def _discover_samples(self) -> list[Sample]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

        prompt_map = self._load_prompt_map()
        samples: list[Sample] = []
        iterator = self.data_dir.rglob("*") if self.recursive else self.data_dir.iterdir()
        for image_path in sorted(path for path in iterator if path.is_file()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if self._should_skip_path(image_path):
                continue
            relative_path = image_path.relative_to(self.data_dir)
            relative_id = relative_path.with_suffix("").as_posix()
            prompt = prompt_map.get(relative_id, prompt_map.get(image_path.stem, self.default_prompt))
            samples.append(
                Sample(
                    image_id=relative_id,
                    image_path=image_path,
                    relative_path=relative_path,
                    prompt=prompt,
                )
            )
        if not samples:
            raise RuntimeError(f"No images found in {self.data_dir}")
        return samples

    def _load_prompt_map(self) -> dict[str, str]:
        prompt_file = self.data_dir / "prompts.csv"
        if not prompt_file.exists():
            return {}

        prompt_map: dict[str, str] = {}
        with prompt_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and {"image_id", "prompt"}.issubset(reader.fieldnames):
                for row in reader:
                    image_id = (row.get("image_id") or "").strip()
                    prompt = (row.get("prompt") or "").strip()
                    if image_id:
                        prompt_map[image_id] = prompt
                return prompt_map

        for line in prompt_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_id, prompt = line.split(",", 1)
            prompt_map[image_id.strip()] = prompt.strip()
        return prompt_map

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return {
            "image_id": sample.image_id,
            "image_path": str(sample.image_path),
            "relative_path": str(sample.relative_path.as_posix()),
            "image": tensor,
            "prompt": sample.prompt,
        }


def build_dataloader(dataset: Dataset, batch_size: int = 1, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
