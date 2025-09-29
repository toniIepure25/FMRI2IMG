from typing import Sequence, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class FMRITextDataset(Dataset):
    """
    Minimal dataset that returns:
      - X[idx]: fMRI feature vector as float32 tensor
      - (idx, text): sample index (for CLIP alignment) and raw caption
    """
    def __init__(self, X: np.ndarray, texts: Sequence[str]):
        assert len(X) == len(texts), "X and texts must have the same length"
        self.X = X
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, str]]:
        # Ensure float32 for model stability
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        i = torch.tensor(idx, dtype=torch.long)
        t = self.texts[idx]
        return x, (i, t)


def make_loaders(
    X: np.ndarray,
    texts: Sequence[str],
    batch_size: int = 2,
    num_workers: int = 0,
    *,
    drop_last: bool = True,         # <-- important for CCA stability
    pin_memory: bool = True,
):
    """
    Returns a single train DataLoader. We drop the last batch to avoid B=1,
    which can cause numerical issues in CCA/statistics.
    """
    ds = FMRITextDataset(X, texts)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,                 # <-- key change
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers) # keeps workers alive if >0
    )
    return loader
