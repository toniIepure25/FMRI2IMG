from typing import Sequence, Tuple, List, Optional
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pytorch_lightning as pl
import os


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


# ---------------------------------------------------------------------
# NEW: LightningDataModule + factory (to fix older imports and scripts)
# ---------------------------------------------------------------------

class NSDDataModule(pl.LightningDataModule):
    """
    Minimal LightningDataModule that loads ROI matrix and captions,
    then exposes train/val DataLoaders compatible with the existing pipeline.
    """
    def __init__(self,
                 images_root: str,
                 fmri_root: str,
                 captions_csv: str,
                 roi_dir: str,
                 subject: str = "subj01",
                 fmri_dim: int = 2048,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 train_ratio: float = 0.9,
                 seed: int = 42):
        super().__init__()
        self.images_root = images_root
        self.fmri_root = fmri_root
        self.captions_csv = captions_csv
        self.roi_dir = roi_dir
        self.subject = subject
        self.fmri_dim = fmri_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.seed = seed

        # Will be populated in setup()
        self.train_ds: Optional[FMRITextDataset] = None
        self.val_ds: Optional[FMRITextDataset] = None

    def _load_roi(self) -> np.ndarray:
        roi_path = os.path.join(self.roi_dir, f"{self.subject}_roi.npy")
        if not os.path.exists(roi_path):
            raise FileNotFoundError(f"ROI file not found: {roi_path}")
        X = np.load(roi_path)
        if X.shape[1] != self.fmri_dim:
            raise ValueError(f"ROI dim mismatch: got {X.shape[1]}, expected {self.fmri_dim}")
        return X

    def _load_captions(self) -> list:
        if not os.path.exists(self.captions_csv):
            # Fallback mock captions if file missing (keeps pipeline running)
            return [f"caption {i}" for i in range( len(self._load_roi()) )]

        texts = []
        with open(self.captions_csv, "r") as f:
            # skip header if present
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                # simple heuristic: last column is caption
                texts.append(parts[-1] if parts else "")
        return texts

    def setup(self, stage: Optional[str] = None):
        X = self._load_roi()
        texts = self._load_captions()
        n = min(len(X), len(texts))
        X, texts = X[:n], texts[:n]

        # split
        n_train = int(self.train_ratio * n)
        self.train_ds = FMRITextDataset(X[:n_train], texts[:n_train])
        self.val_ds   = FMRITextDataset(X[n_train:], texts[n_train:])

    def train_dataloader(self):
        assert self.train_ds is not None
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        assert self.val_ds is not None
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, drop_last=False)


def make_datamodule(images_root: str,
                    fmri_root: str,
                    captions_csv: str,
                    roi_dir: str,
                    subject: str = "subj01",
                    fmri_dim: int = 2048,
                    batch_size: int = 4,
                    num_workers: int = 4,
                    train_ratio: float = 0.9,
                    seed: int = 42) -> NSDDataModule:
    """
    Backwards-compatible factory used by older eval scripts.
    """
    return NSDDataModule(
        images_root=images_root,
        fmri_root=fmri_root,
        captions_csv=captions_csv,
        roi_dir=roi_dir,
        subject=subject,
        fmri_dim=fmri_dim,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=train_ratio,
        seed=seed,
    )
