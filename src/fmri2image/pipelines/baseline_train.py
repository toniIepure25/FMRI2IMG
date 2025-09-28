from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import numpy as np  # <— NEW
from ..models.encoders.mlp_encoder import FMRIEncoderMLP
from ..data.nsd_reader import NSDReader
from ..data.datamodule import make_loaders

class CosineAlignLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)
    def forward(self, latent, text_emb_batch):
        latent = latent / (latent.norm(dim=-1, keepdim=True) + 1e-8)
        return 1.0 - self.cos(latent, text_emb_batch).mean()

class LitModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, clip_text_feats: np.ndarray):  # <— expects feats
        super().__init__()
        m = cfg.train.model
        # project fmri -> CLIP text dim
        self.encoder = FMRIEncoderMLP(m.fmri_input_dim, clip_text_feats.shape[1], m.hidden)
        self.criterion = CosineAlignLoss()
        self.clip_text_feats = torch.tensor(clip_text_feats, dtype=torch.float32)
        self.save_hyperparameters()

    def training_step(self, batch, _):
        x, (idx, _texts) = batch            # <— get index from dataset
        z = self.encoder(x)
        text_emb = self.clip_text_feats[idx].to(z.device)  # align with correct caption
        loss = self.criterion(z, text_emb)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams["cfg"].train.optimizer.lr)

def run_baseline(cfg: DictConfig):
    reader = NSDReader(
        cfg.data.paths.images_root,
        cfg.data.paths.fmri_root,
        cfg.data.paths.captions,
        roi_dir=cfg.data.roi.out_dir,
        subject=cfg.data.subjects[0] if "subjects" in cfg.data and cfg.data.subjects else "subj01",
    )
    X, texts = reader.load(n=64, fmri_dim=cfg.train.model.fmri_input_dim)

    # Load CLIP text embeddings saved by DVC stage
    clip_feats = np.load("data/processed/nsd/clip_text.npy")
    # keep arrays aligned in length
    if len(clip_feats) < len(X):
        X = X[: len(clip_feats)]
        texts = texts[: len(clip_feats)]

    dl = make_loaders(X, texts, cfg.train.batch_size, cfg.train.num_workers)
    model = LitModule(cfg, clip_feats)      # <— pass feats here

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        default_root_dir=cfg.run.output_dir,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, dl)
