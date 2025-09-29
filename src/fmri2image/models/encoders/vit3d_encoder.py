import torch
import torch.nn as nn

class ViT3DEncoderLite(nn.Module):
    """
    A very small '3D ViT' surrogate for ablations, operating on
    fMRI ROI vectors by reshaping the feature dim into T x token_dim.
    In real fMRI 4D you'd feed [B, T, V] or [B, T, H, W, D] tokens.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_steps: int = 8,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert in_dim % time_steps == 0, "in_dim must be divisible by time_steps for the lite reshaping"
        self.time_steps = int(time_steps)
        token_dim = in_dim // self.time_steps
        self.token_proj = nn.Linear(token_dim, token_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=int(token_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim] -> reshape to [B, T, token_dim], run transformer over T,
        pool (mean) and map to out_dim.
        """
        B, D = x.shape
        T = self.time_steps
        token_dim = D // T
        xt = x.view(B, T, token_dim)
        xt = self.token_proj(xt)
        xt = self.backbone(xt)   # [B, T, token_dim]
        xt = xt.mean(dim=1)      # temporal mean pool
        z = self.head(xt)        # [B, out_dim]
        return z
