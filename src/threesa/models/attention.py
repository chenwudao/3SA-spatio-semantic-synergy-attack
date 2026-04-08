from __future__ import annotations

import torch

from .base import VisionLanguageSurrogate


def topk_binary_mask(attention_map: torch.Tensor, ratio: float) -> torch.Tensor:
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be within (0, 1], got {ratio}")
    batch, _, height, width = attention_map.shape
    flat = attention_map.view(batch, -1)
    k = max(1, int(flat.shape[-1] * ratio))
    thresholds = flat.topk(k, dim=-1).values[..., -1]
    mask = flat >= thresholds.unsqueeze(-1)
    return mask.view(batch, 1, height, width).to(attention_map.dtype)


def compute_mask_iou(*masks: torch.Tensor) -> torch.Tensor:
    if len(masks) < 2:
        raise ValueError("At least two masks are required to compute IoU")
    stacked = torch.stack([mask.bool() for mask in masks], dim=0)
    intersection = stacked.all(dim=0).sum(dim=(1, 2, 3))
    union = stacked.any(dim=0).sum(dim=(1, 2, 3)).clamp_min(1)
    return intersection.float() / union.float()


def combine_attention_maps(
    surrogates: list[VisionLanguageSurrogate],
    image: torch.Tensor,
    text_prompt: str,
    strategy: str = "weighted_topk",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    attention_maps = {
        surrogate.name: surrogate.extract_attention(image, text_prompt).attention_map
        for surrogate in surrogates
    }
    stacked = torch.stack(list(attention_maps.values()), dim=0)

    if strategy == "intersection":
        combined = stacked.min(dim=0).values
    elif strategy == "union":
        combined = stacked.max(dim=0).values
    elif strategy == "weighted_topk":
        weights = torch.tensor(
            [surrogate.weight for surrogate in surrogates],
            device=image.device,
            dtype=stacked.dtype,
        ).view(-1, 1, 1, 1, 1)
        combined = (stacked * weights).sum(dim=0) / weights.sum()
    else:
        raise ValueError(f"Unsupported mask strategy: {strategy}")

    return combined, attention_maps
