import torch
from typing import Sequence

@torch.no_grad()
def topk_retrieval(sim: torch.Tensor, ks: Sequence[int] = (1, 5)) -> dict:
    """
    sim: [B, B] similarity matrix (rows = queries, cols = candidates).
    Diagonal entries are positives.
    Returns: {'top1': ..., 'top5': ...} accuracies in [0,1].
    """
    B = sim.size(0)
    target = torch.arange(B, device=sim.device)
    ranks = (-sim).argsort(dim=1)  # descending similarity
    metrics = {}
    for k in ks:
        pred_ok = (ranks[:, :k] == target[:, None]).any(dim=1).float().mean().item()
        metrics[f"top{k}"] = pred_ok
    return metrics
