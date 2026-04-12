from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import VisionLanguageSurrogate


def topk_binary_mask(attention_map: torch.Tensor, ratio: float) -> torch.Tensor:
    """Hard binary mask for IoU computation (Stage 1 analysis).
    
    Args:
        attention_map: (B, 1, H, W) or (B, H, W) continuous attention map
        ratio: Top-K ratio (e.g., 0.05 for 5%)
    
    Returns:
        mask: (B, 1, H, W) binary mask {0, 1}
    """
    if attention_map.dim() == 3:
        attention_map = attention_map.unsqueeze(1)
    
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be within (0, 1], got {ratio}")
    batch, _, height, width = attention_map.shape
    flat = attention_map.view(batch, -1)
    k = max(1, int(flat.shape[-1] * ratio))
    thresholds = flat.topk(k, dim=-1).values[..., -1]
    mask = flat >= thresholds.unsqueeze(-1)
    return mask.view(batch, 1, height, width).to(attention_map.dtype)


def topk_soft_mask(
    attention_map: torch.Tensor, 
    ratio: float,
    soft_ratio: float = 2.0,
    sigma: float = 3.0,
    temperature: float = 10.0,
) -> torch.Tensor:
    """Soft mask with Gaussian smoothing for gradient-based attacks (Stage 2).
    
    This addresses the "gradient suffocation" problem of hard masks:
    - Hard 5% mask → ~5600 pixels → insufficient attack budget
    - Soft 10-15% mask → ~11000-17000 pixels → adequate budget + smooth gradients
    
    Strategy (fully differentiable):
    1. Apply Gaussian smoothing to attention map
    2. Use sigmoid with temperature to create soft thresholding
    3. Renormalize to achieve effective coverage of soft_ratio * ratio
    
    Args:
        attention_map: (B, 1, H, W) or (B, H, W) continuous attention map
        ratio: Base Top-K ratio (e.g., 0.05 for 5% triple intersection)
        soft_ratio: Expansion factor (default 2.0 → 5% becomes 10%)
        sigma: Gaussian kernel std for smoothing (default 3.0)
        temperature: Sigmoid temperature for soft thresholding (default 10.0)
    
    Returns:
        soft_mask: (B, 1, H, W) soft mask with values in [0, 1]
    """
    if attention_map.dim() == 3:
        attention_map = attention_map.unsqueeze(1)
    
    batch, _, height, width = attention_map.shape
    device = attention_map.device
    
    # Step 1: Create Gaussian kernel
    kernel_size = int(2 * sigma * 2 + 1)  # Ensure odd size
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 1D Gaussian kernel
    coords = torch.arange(kernel_size, device=device, dtype=attention_map.dtype) - kernel_size // 2
    gaussian_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    
    # 2D Gaussian kernel via outer product
    gaussian_2d = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)  # (K, K)
    gaussian_kernel = gaussian_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
    
    # Step 2: Apply Gaussian smoothing to attention map
    padding = kernel_size // 2
    smoothed_attn = F.conv2d(
        attention_map, 
        gaussian_kernel, 
        padding=padding
    )
    
    # Step 3: Compute adaptive threshold using quantile
    # Use differentiable approximation via sorting
    flat_smooth = smoothed_attn.view(batch, -1)
    target_pixels = int(soft_ratio * ratio * height * width)
    
    if target_pixels > 0 and target_pixels < flat_smooth.shape[-1]:
        # Get threshold value (differentiable through attention_map)
        threshold = flat_smooth.topk(target_pixels, dim=-1).values[..., -1:]
        
        # Step 4: Soft thresholding using sigmoid (fully differentiable)
        # mask = sigmoid(temperature * (smoothed_attn - threshold))
        soft_mask = torch.sigmoid(temperature * (smoothed_attn - threshold.view(batch, 1, 1, 1)))
    else:
        # Fallback: return smoothed attention normalized
        soft_min = flat_smooth.min(dim=-1, keepdim=True).values.view(batch, 1, 1, 1)
        soft_max = flat_smooth.max(dim=-1, keepdim=True).values.view(batch, 1, 1, 1)
        soft_mask = (smoothed_attn - soft_min) / (soft_max - soft_min + 1e-8)
    
    return soft_mask


def compute_triple_intersection_soft(
    attention_maps: dict[str, torch.Tensor],
    ratio: float = 0.05,
    soft_ratio: float = 2.0,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Compute soft triple intersection mask for attack targeting.
    
    Args:
        attention_maps: dict mapping model_name → (B, H, W) attention maps
        ratio: Base ratio for triple intersection (default 0.05)
        soft_ratio: Expansion factor (default 2.0 → 5% becomes 10%)
        sigma: Gaussian smoothing sigma
    
    Returns:
        soft_triple_mask: (B, H, W) soft mask for attack
    """
    # Stack attention maps
    stacked = torch.stack(list(attention_maps.values()), dim=0)  # (N, B, H, W)
    
    # Compute triple intersection (element-wise minimum)
    triple_intersection = stacked.min(dim=0).values  # (B, H, W)
    triple_intersection = triple_intersection.unsqueeze(1)  # (B, 1, H, W)
    
    # Apply soft mask transformation
    soft_mask = topk_soft_mask(triple_intersection, ratio, soft_ratio, sigma)
    
    return soft_mask.squeeze(1)  # (B, H, W)


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
