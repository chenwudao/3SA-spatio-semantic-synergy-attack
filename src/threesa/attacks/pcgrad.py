from __future__ import annotations

import torch


def _flatten_gradient(gradient: torch.Tensor) -> torch.Tensor:
    return gradient.reshape(gradient.shape[0], -1)


def project_conflicting_gradients(gradients: list[torch.Tensor]) -> torch.Tensor:
    if not gradients:
        raise ValueError("gradients must not be empty")

    projected = [gradient.clone() for gradient in gradients]
    for i in range(len(projected)):
        gi = _flatten_gradient(projected[i])
        for j in range(len(projected)):
            if i == j:
                continue
            gj = _flatten_gradient(projected[j])
            dot = (gi * gj).sum(dim=1, keepdim=True)
            gj_norm_sq = gj.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12)
            conflict = dot < 0
            correction = (dot / gj_norm_sq) * gj
            gi = torch.where(conflict, gi - correction, gi)
        projected[i] = gi.view_as(projected[i])
    return torch.stack(projected, dim=0).mean(dim=0)
