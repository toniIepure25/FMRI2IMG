# src/fmri2image/models/heads/visual_head.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Sequence, Optional

class VisualHead(nn.Module):
    """
    Simple MLP mapping from fMRI features -> CLIP-Image embedding space (e.g., 512-dim).
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[Sequence[int]] = None, dropout: float = 0.0):
        super().__init__()
        hidden = list(hidden) if hidden else []
        dims = [in_dim] + hidden + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
