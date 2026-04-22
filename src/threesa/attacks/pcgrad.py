from __future__ import annotations

import torch


def _flatten_gradient(gradient: torch.Tensor) -> torch.Tensor:
    return gradient.reshape(gradient.shape[0], -1)


def _normalize_gradient(gradient: torch.Tensor) -> torch.Tensor:
    flat = _flatten_gradient(gradient)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / norm).view_as(gradient)


def project_conflicting_gradients(
    gradients: list[torch.Tensor],
    *,
    normalize: bool = False,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    if not gradients:
        raise ValueError("gradients must not be empty")

    working = [_normalize_gradient(gradient) if normalize else gradient.clone() for gradient in gradients]
    pairwise_cosines: dict[str, float] = {}
    conflict_total = 0.0
    conflict_pairs = 0

    for i in range(len(working)):
        gi = _flatten_gradient(working[i])
        gi_norm = gi.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        for j in range(i + 1, len(working)):
            gj = _flatten_gradient(working[j])
            gj_norm = gj.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            cosine = (gi * gj).sum(dim=1, keepdim=True) / (gi_norm * gj_norm)
            pairwise_cosines[f"{i}__{j}"] = float(cosine.mean().item())
            conflict_total += float((cosine < 0).float().mean().item())
            conflict_pairs += 1

    projected = [gradient.clone() for gradient in working]
    order = list(range(len(projected)))
    if shuffle and len(order) > 1:
        permutation = torch.randperm(len(order), generator=generator).tolist()
        order = [order[index] for index in permutation]

    for i in order:
        gi = _flatten_gradient(projected[i])
        for j in order:
            if i == j:
                continue
            gj = _flatten_gradient(projected[j])
            dot = (gi * gj).sum(dim=1, keepdim=True)
            gj_norm_sq = gj.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12)
            conflict = dot < 0
            correction = (dot / gj_norm_sq) * gj
            gi = torch.where(conflict, gi - correction, gi)
        projected[i] = gi.view_as(projected[i])

    aggregated = torch.stack(projected, dim=0).mean(dim=0)
    if not return_info:
        return aggregated

    info = {
        "order": order,
        "pairwise_cosine_similarity": pairwise_cosines,
        "conflict_rate": conflict_total / max(conflict_pairs, 1),
    }
    return aggregated, info
