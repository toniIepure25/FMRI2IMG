# src/fmri2image/losses/clip_losses.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["ClipStyleContrastiveLoss"]

class ClipStyleContrastiveLoss(nn.Module):
    """
    CLIP-style InfoNCE cu temperatură învățabilă (logit_scale).
    Folosește negative in-batch. Opțional simetric (z->t și t->z).
    Returnează și logits pentru a calcula metrice de retrieval.
    """
    def __init__(self, temperature_init: float = 0.07, symmetric: bool = True):
        super().__init__()
        # în CLIP se învață logit_scale ~ log(1/T)
        self.logit_scale = nn.Parameter(
            torch.tensor(float(np.log(1.0 / float(temperature_init))), dtype=torch.float32)
        )
        self.symmetric = bool(symmetric)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> dict:
        """
        Args:
            z: [B, D] – proiecția din fMRI
            t: [B, D] – embedding-uri CLIP text (sau imagine) corespunzătoare
        Returns:
            dict(loss=..., logits_zt=..., logits_tz|None=..., temp=...)
        """
        # normalizăm (siguranță, chiar dacă t e deja unit-norm)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)

        # temperatură: scale = exp(logit_scale)
        logit_scale = self.logit_scale.clamp(-5.0, 8.0)  # stabilitate numerică
        scale = torch.exp(logit_scale)

        # similarități (cosine; pentru că e unit norm) * scale
        # logits_zt: rând i = query z_i, coloana j = key t_j
        logits_zt = (z @ t.t()) * scale  # [B, B]
        target = torch.arange(z.size(0), device=z.device)
        loss_zt = nn.functional.cross_entropy(logits_zt, target)

        if self.symmetric:
            logits_tz = (t @ z.t()) * scale  # [B, B]
            loss_tz = nn.functional.cross_entropy(logits_tz, target)
            loss = 0.5 * (loss_zt + loss_tz)
        else:
            logits_tz = None
            loss = loss_zt

        # raportăm temperatura (nu scale) pentru logging
        temp = torch.exp(-logit_scale)
        return {"loss": loss, "logits_zt": logits_zt, "logits_tz": logits_tz, "temp": temp}
