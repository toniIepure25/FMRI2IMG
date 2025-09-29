from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

from omegaconf import OmegaConf
from fmri2image.data.nsd_reader import NSDReader

# Simple masked autoencoder over ROI vector (no time)
class MaskedVectorAE(nn.Module):
    def __init__(self, dim: int, hidden: list[int] = [2048, 1024]):
        super().__init__()
        enc_dims = [dim] + hidden
        dec_dims = hidden[::-1] + [dim]
        enc = []
        for i in range(len(enc_dims) - 1):
            enc += [nn.Linear(enc_dims[i], enc_dims[i+1]), nn.ReLU()]
        dec = []
        for i in range(len(dec_dims) - 1):
            dec += [nn.Linear(dec_dims[i], dec_dims[i+1])]
            if i < len(dec_dims) - 2:
                dec += [nn.ReLU()]
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class LitMaskedAE(pl.LightningModule):
    def __init__(self, dim: int, mask_ratio: float = 0.5, lr: float = 1e-3, hidden=[2048,1024]):
        super().__init__()
        self.save_hyperparameters()
        self.model = MaskedVectorAE(dim, hidden)
        self.mask_ratio = mask_ratio
        self.criterion = nn.MSELoss()
        self.lr = lr

    def training_step(self, batch, _):
        (x,) = batch
        B, D = x.shape
        # random mask per sample
        k = int(D * self.mask_ratio)
        idx = torch.rand(B, D, device=x.device).argsort(dim=1)
        mask = torch.ones_like(x)
        mask.scatter_(1, idx[:, :k], 0.0)  # masked positions -> 0
        x_masked = x * mask

        x_hat = self.model(x_masked)
        loss = self.criterion(x_hat * (1 - mask), x * (1 - mask))  # reconstruct only masked
        self.log("pretrain/mse_masked", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_root", type=str, default="data/raw/nsd/images")
    p.add_argument("--fmri_root", type=str, default="data/raw/nsd/fmri")
    p.add_argument("--captions", type=str, default="data/raw/nsd/captions.csv")
    p.add_argument("--roi_dir", type=str, default="data/processed/nsd/roi")
    p.add_argument("--subject", type=str, default="subj01")
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_ckpt", type=str, default="data/artifacts/encoder_pretrained.ckpt")
    args = p.parse_args()

    # Load ROI vectors (mock)
    reader = NSDReader(args.images_root, args.fmri_root, args.captions,
                       roi_dir=args.roi_dir, subject=args.subject)
    X, _ = reader.load(n=1024, fmri_dim=args.dim)  # more samples for pretrain
    X = torch.tensor(X, dtype=torch.float32)

    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    lit = LitMaskedAE(dim=args.dim, mask_ratio=args.mask_ratio, lr=args.lr)
    trainer = pl.Trainer(max_epochs=args.epochs, enable_checkpointing=False, logger=False)
    trainer.fit(lit, dl)

    # Save state dict so we can load encoder weights later
    ckpt = {"state_dict": lit.model.encoder.state_dict(), "dim": args.dim}
    out_path = args.out_ckpt
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"[ok] saved pretrained encoder -> {out_path}")

if __name__ == "__main__":
    main()
