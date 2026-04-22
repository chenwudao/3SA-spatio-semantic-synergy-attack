from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class AttentionOutput:
    attention_map: torch.Tensor
    metadata: dict[str, object]


@dataclass(slots=True)
class LossOutput:
    loss: torch.Tensor
    metadata: dict[str, object]


class VisionLanguageSurrogate(ABC):
    def __init__(self, name: str, weight: float = 1.0) -> None:
        self.name = name
        self.weight = weight

    @abstractmethod
    def extract_attention(self, image: torch.Tensor, text_prompt: str) -> AttentionOutput:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, image: torch.Tensor, text_prompt: str) -> LossOutput:
        raise NotImplementedError
