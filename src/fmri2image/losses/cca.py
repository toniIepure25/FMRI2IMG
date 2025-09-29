import torch
import torch.nn as nn

class DeepCCALoss(nn.Module):
    """
    Safe Deep CCA-style auxiliary loss:
    - proiectează z și t în același spațiu
    - centrează + standardizează cu unbiased=False
    - dacă batch < 2, sare peste (loss=0)
    - maximizează corelația medie (loss = -mean|corr|)
    """
    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        self.proj_z = nn.Linear(in_dim, out_dim)
        self.proj_t = nn.Linear(in_dim, out_dim)
        self.eps = float(eps)

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        # center + scale; unbiased=False ca să nu ceară d.o.f. >= 1
        x = x - x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False)
        std = torch.clamp(std, min=self.eps)
        return x / std

    def forward(self, z: torch.Tensor, t: torch.Tensor):
        B = z.size(0)
        device = z.device
        if B < 2:
            # prea puține eșantioane pentru o estimare rezonabilă
            zero = torch.zeros((), device=device, dtype=z.dtype)
            return {"loss": zero, "corr_sum": zero}

        z_p = self.proj_z(z)
        t_p = self.proj_t(t)

        z_p = self._standardize(z_p)
        t_p = self._standardize(t_p)

        # corelație pe feature-uri, apoi media pe feature-uri
        # echivalent cu cosine pe coloane (după standardizare)
        corr_per_dim = (z_p * t_p).mean(dim=0)           # [D]
        corr_mean = corr_per_dim.abs().mean()            # scalar

        loss = -corr_mean
        return {"loss": loss, "corr_sum": corr_mean}
