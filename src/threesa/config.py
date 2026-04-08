from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AttackConfig:
    epsilon: float = 8.0 / 255.0
    step_size: float = 2.0 / 255.0
    stages: int = 10
    iterations_per_stage: int = 5
    topk_ratio: float = 0.10
    mask_strategy: str = "weighted_topk"
    device: str = "cuda"


@dataclass(slots=True)
class DefenseConfig:
    enable_rrc: bool = True
    enable_gaussian_blur: bool = True
    enable_jpeg: bool = True
    rrc_scale_min: float = 0.90
    rrc_scale_max: float = 1.00
    blur_kernel_sizes: tuple[int, ...] = (3, 5)
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 1.4
    jpeg_quality_min: int = 55
    jpeg_quality_max: int = 95


@dataclass(slots=True)
class ExperimentConfig:
    image_size: int = 336
    batch_size: int = 1
    num_workers: int = 0
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
