from .base import AttentionOutput, LossOutput, VisionLanguageSurrogate
from .mock import MockVisionLanguageSurrogate

__all__ = [
    "AttentionOutput",
    "LossOutput",
    "VisionLanguageSurrogate",
    "MockVisionLanguageSurrogate",
    "DINOv2Surrogate",
    "InternVLSurrogate",
    "LLaVASurrogate",
]


def __getattr__(name: str):
    if name in {"DINOv2Surrogate", "InternVLSurrogate", "LLaVASurrogate"}:
        from .real_surrogates import DINOv2Surrogate, InternVLSurrogate, LLaVASurrogate

        mapping = {
            "DINOv2Surrogate": DINOv2Surrogate,
            "InternVLSurrogate": InternVLSurrogate,
            "LLaVASurrogate": LLaVASurrogate,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
