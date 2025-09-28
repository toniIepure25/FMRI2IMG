import torch, torch.nn as nn

class FMRIEncoderMLP(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden: list[int]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, latent_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
