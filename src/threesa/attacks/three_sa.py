from __future__ import annotations

import torch

from threesa.config import AttackConfig
from threesa.models import VisionLanguageSurrogate
from threesa.models.attention import combine_attention_maps, topk_binary_mask

from .pcgrad import project_conflicting_gradients


class ThreeSAAttack:
    def __init__(self, surrogates: list[VisionLanguageSurrogate], config: AttackConfig) -> None:
        if not surrogates:
            raise ValueError("ThreeSAAttack requires at least one surrogate model")
        self.surrogates = surrogates
        self.config = config

    def run(self, image: torch.Tensor, target_text: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean = image.detach()
        adv = clean.clone().detach()
        trace: dict[str, torch.Tensor] = {}

        for stage in range(self.config.stages):
            combined_attention, attention_maps = combine_attention_maps(
                self.surrogates,
                adv.detach(),
                target_text,
                strategy=self.config.mask_strategy,
            )
            mask = topk_binary_mask(combined_attention, self.config.topk_ratio)
            trace[f"stage_{stage}_mask"] = mask.detach().cpu()

            for _ in range(self.config.iterations_per_stage):
                adv = adv.detach().requires_grad_(True)
                gradients: list[torch.Tensor] = []
                losses: list[torch.Tensor] = []
                for surrogate in self.surrogates:
                    loss = surrogate.compute_loss(adv, target_text).loss
                    gradient = torch.autograd.grad(loss, adv, retain_graph=True)[0]
                    gradients.append(gradient)
                    losses.append(loss.detach())

                aggregated = project_conflicting_gradients(gradients)
                aggregated = aggregated * mask
                step = self.config.step_size * aggregated.sign()
                adv = adv.detach() + step
                delta = torch.clamp(adv - clean, min=-self.config.epsilon, max=self.config.epsilon)
                adv = torch.clamp(clean + delta, min=0.0, max=1.0)

            trace[f"stage_{stage}_attention"] = combined_attention.detach().cpu()
            trace[f"stage_{stage}_losses"] = torch.stack(losses).detach().cpu()
            for name, attn in attention_maps.items():
                trace[f"stage_{stage}_attention_{name}"] = attn.detach().cpu()

        return adv.detach(), trace
