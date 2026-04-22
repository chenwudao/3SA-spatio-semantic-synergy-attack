from __future__ import annotations

import torch

from threesa.config import AttackConfig
from threesa.models import VisionLanguageSurrogate
from threesa.models.attention import combine_attention_maps, topk_binary_mask, topk_soft_mask

from .pcgrad import project_conflicting_gradients


class ThreeSAAttack:
    def __init__(self, surrogates: list[VisionLanguageSurrogate], config: AttackConfig) -> None:
        if not surrogates:
            raise ValueError("ThreeSAAttack requires at least one surrogate model")
        self.surrogates = surrogates
        self.config = config

    def _normalize_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        flat = gradient.reshape(gradient.shape[0], -1)
        norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        return (flat / norm).view_as(gradient)

    def _resolve_attention_strategy(self) -> str:
        if self.config.mask_strategy == "global":
            return "weighted_topk"
        return self.config.mask_strategy

    def _build_mask(
        self,
        combined_attention: torch.Tensor,
        attention_maps: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.config.mask_strategy == "global":
            return torch.ones_like(combined_attention, device=combined_attention.device)

        if self.config.use_soft_mask:
            return topk_soft_mask(
                combined_attention,
                ratio=self.config.topk_ratio,
                soft_ratio=self.config.soft_mask_expansion,
                sigma=self.config.soft_mask_sigma,
                temperature=self.config.soft_mask_temperature,
            )
        return topk_binary_mask(combined_attention, self.config.topk_ratio)

    def _collect_gradients(
        self,
        adv: torch.Tensor,
        target_text: str,
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        gradients: list[torch.Tensor] = []
        losses: dict[str, float] = {}

        for surrogate in self.surrogates:
            adv_var = adv.detach().clone().requires_grad_(True)
            loss_output = surrogate.compute_loss(adv_var, target_text)
            gradient = torch.autograd.grad(loss_output.loss, adv_var)[0].detach()
            gradients.append(gradient)
            losses[surrogate.name] = float(loss_output.loss.detach().item())

        return gradients, losses

    def _aggregate_gradients(self, gradients: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, object]]:
        normalized = [self._normalize_gradient(gradient) for gradient in gradients] if self.config.normalize_gradients else gradients
        if self.config.gradient_strategy == "pcgrad":
            aggregated, info = project_conflicting_gradients(
                normalized,
                shuffle=self.config.shuffle_pcgrad,
                return_info=True,
            )
            return aggregated, info
        if self.config.gradient_strategy == "vanilla_addition":
            aggregated = torch.stack(normalized, dim=0).mean(dim=0)
            return aggregated, {"pairwise_cosine_similarity": {}, "conflict_rate": 0.0, "order": list(range(len(normalized)))}
        raise ValueError(f"Unsupported gradient strategy: {self.config.gradient_strategy}")

    def run(self, image: torch.Tensor, target_text: str) -> tuple[torch.Tensor, dict[str, object]]:
        clean = image.detach()
        adv = clean.clone().detach()
        trace: dict[str, object] = {
            "surrogates": [surrogate.name for surrogate in self.surrogates],
            "mask_strategy": self.config.mask_strategy,
            "gradient_strategy": self.config.gradient_strategy,
            "use_soft_mask": self.config.use_soft_mask,
            "stages": [],
        }

        for stage in range(self.config.stages):
            combined_attention, attention_maps = combine_attention_maps(
                self.surrogates,
                adv.detach(),
                target_text,
                strategy=self._resolve_attention_strategy(),
            )
            mask = self._build_mask(combined_attention, attention_maps).to(adv.device, adv.dtype)
            stage_trace: dict[str, object] = {
                "stage": stage,
                "mask_mean": float(mask.mean().item()),
                "mask_support_ratio": float((mask > 0.5).float().mean().item()),
                "iterations": [],
            }

            for _ in range(self.config.iterations_per_stage):
                gradients, loss_map = self._collect_gradients(adv, target_text)
                aggregated, aggregation_info = self._aggregate_gradients(gradients)
                if self.config.mask_strategy == "global":
                    step_direction = aggregated.sign()
                elif self.config.use_soft_mask:
                    step_direction = aggregated.sign() * mask
                else:
                    step_direction = (aggregated * mask).sign()

                step = self.config.step_size * step_direction
                adv = adv.detach() + step
                delta = torch.clamp(adv - clean, min=-self.config.epsilon, max=self.config.epsilon)
                adv = torch.clamp(clean + delta, min=0.0, max=1.0)

                stage_trace["iterations"].append(
                    {
                        "losses": loss_map,
                        "conflict_rate": float(aggregation_info.get("conflict_rate", 0.0)),
                        "pairwise_cosine_similarity": aggregation_info.get("pairwise_cosine_similarity", {}),
                    }
                )

            delta = adv - clean
            stage_trace["delta_linf"] = float(delta.abs().amax().item())
            stage_trace["delta_l2"] = float(delta.reshape(delta.shape[0], -1).norm(p=2, dim=1).mean().item())
            trace["stages"].append(stage_trace)

        return adv.detach(), trace
