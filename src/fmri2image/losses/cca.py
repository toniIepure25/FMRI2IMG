import torch
import torch.nn as nn

class DeepCCALoss(nn.Module):
    """
    Lightweight Deep CCA-style loss that learns linear projections Wz, Wt
    and maximizes correlation between projected views.
    Works on batch [B, D]. Returns negative total correlation (to minimize).
    """
    def __init__(self, in_dim: int, out_dim: int = 128, eps: float = 1e-6):
        super().__init__()
        self.proj_z = nn.Linear(in_dim, out_dim, bias=False)
        self.proj_t = nn.Linear(in_dim, out_dim, bias=False)
        self.eps = eps

    @staticmethod
    def _center(x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=0, keepdim=True)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> dict:
        """
        z, t: [B, D] (assumed roughly comparable dims; z already normalized upstream)
        Returns:
            {"loss": loss_scalar, "corr_sum": corr_sum}
        """
        z = self._center(self.proj_z(z))
        t = self._center(self.proj_t(t))

        # Normalize per-dim
        z = z / (z.std(dim=0, keepdim=True) + self.eps)
        t = t / (t.std(dim=0, keepdim=True) + self.eps)

        # Per-dimension correlation over the batch
        corr = (z * t).mean(dim=0)  # [out_dim]
        corr_sum = corr.abs().sum()  # encourage magnitude alignment

        # We MINIMIZE negative correlation sum
        loss = -corr_sum
        return {"loss": loss, "corr_sum": corr_sum}
