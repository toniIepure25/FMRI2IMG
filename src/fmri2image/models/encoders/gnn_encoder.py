import torch
import torch.nn as nn

class GraphMLPEncoderLite(nn.Module):
    """
    Minimal graph-like encoder using adjacency via (I + D^{-1/2} A D^{-1/2})
    applied as a fixed linear propagation step, followed by MLP.
    Avoids external deps (PyG). Works with ROI vectors shaped [B, D].
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: list[int], dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + list(hidden) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        # adjacency placeholder (registered later via set_adj)
        self.register_buffer("prop_matrix", None, persistent=False)

    def set_adj(self, adj: torch.Tensor):
        """
        adj: [D, D] unweighted/weighted adjacency (0/1 or weights).
        Builds normalized propagation matrix P = I + D^{-1/2} A D^{-1/2}.
        """
        A = adj
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        P = I + D_inv_sqrt @ A @ D_inv_sqrt
        self.prop_matrix = P

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]; apply one graph propagation x' = x P, then MLP.
        If no adj given, just MLP.
        """
        if self.prop_matrix is not None:
            x = x @ self.prop_matrix
        x = self.dropout(x)
        return self.mlp(x)
