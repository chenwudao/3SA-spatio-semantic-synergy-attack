from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

from .base import AttentionOutput, LossOutput, VisionLanguageSurrogate


class MockVisionLanguageSurrogate(VisionLanguageSurrogate):
    """Deterministic surrogate used to validate the full pipeline before real checkpoints."""

    def __init__(self, name: str, weight: float = 1.0, temperature: float = 1.0) -> None:
        super().__init__(name=name, weight=weight)
        self.temperature = temperature

    def _seed_from_prompt(self, prompt: str) -> int:
        digest = hashlib.sha256(f"{self.name}:{prompt}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def extract_attention(self, image: torch.Tensor, text_prompt: str) -> AttentionOutput:
        if image.dim() != 4:
            raise ValueError("Expected image batch with shape [B, C, H, W]")
        batch, _, height, width = image.shape
        pooled = image.mean(dim=1, keepdim=True)
        pooled = F.avg_pool2d(pooled, kernel_size=7, stride=1, padding=3)
        generator = torch.Generator(device=image.device)
        generator.manual_seed(self._seed_from_prompt(text_prompt))
        noise = torch.rand((batch, 1, height, width), generator=generator, device=image.device)
        attention = (pooled + 0.15 * noise) / self.temperature
        attention = attention.flatten(2)
        attention = F.softmax(attention, dim=-1).view(batch, 1, height, width)
        return AttentionOutput(attention_map=attention, metadata={"surrogate": self.name})

    def compute_loss(self, image: torch.Tensor, text_prompt: str) -> LossOutput:
        attention = self.extract_attention(image, text_prompt).attention_map
        token_bias = (self._seed_from_prompt(text_prompt) % 997) / 997.0
        target = torch.full_like(attention, fill_value=token_bias)
        loss = F.mse_loss(attention, target) * self.weight
        return LossOutput(loss=loss, metadata={"surrogate": self.name})
