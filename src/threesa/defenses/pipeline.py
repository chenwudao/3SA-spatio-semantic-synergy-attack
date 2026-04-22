from __future__ import annotations

import io
import random

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from threesa.config import DefenseConfig


class DefensePipeline:
    def __init__(self, config: DefenseConfig) -> None:
        self.config = config

    def _apply_rrc(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        scale = random.uniform(self.config.rrc_scale_min, self.config.rrc_scale_max)
        crop_height = max(1, int(height * scale))
        crop_width = max(1, int(width * scale))
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        cropped = TF.resized_crop(
            image,
            top=top,
            left=left,
            height=crop_height,
            width=crop_width,
            size=[height, width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return cropped

    def _apply_blur(self, image: torch.Tensor) -> torch.Tensor:
        kernel_size = random.choice(self.config.blur_kernel_sizes)
        sigma = random.uniform(self.config.blur_sigma_min, self.config.blur_sigma_max)
        return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    def _apply_jpeg(self, image: torch.Tensor) -> torch.Tensor:
        pil_image = TF.to_pil_image(image.clamp(0.0, 1.0))
        buffer = io.BytesIO()
        quality = random.randint(self.config.jpeg_quality_min, self.config.jpeg_quality_max)
        pil_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        return TF.to_tensor(compressed)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        defended = []
        for image in batch:
            current = image
            if self.config.enable_rrc:
                current = self._apply_rrc(current)
            if self.config.enable_gaussian_blur:
                current = self._apply_blur(current)
            if self.config.enable_jpeg:
                current = self._apply_jpeg(current)
            defended.append(current)
        return torch.stack(defended, dim=0)
