import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    Classic LoRA module: y = xW + alpha/ r * xBA
    W is frozen (assumed in parent), only A,B are trainable.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.A = nn.Linear(in_f, rank, bias=False)
        self.B = nn.Linear(rank, out_f, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + (self.alpha / self.rank) * self.B(self.A(x))
