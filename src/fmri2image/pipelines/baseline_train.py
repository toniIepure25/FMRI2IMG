from omegaconf import DictConfig
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from ..models.encoders.mlp_encoder import FMRIEncoderMLP
from ..data.nsd_reader import NSDReader
from ..data.datamodule import make_loaders

class DummyCLIPLoss(nn.Module):
    def forward(self, latent, texts):
        # Placeholder: L2 spre origine (înlocuit ulterior cu CLIP guidance)
        return (latent ** 2).mean()

class LitModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        m = cfg.train.model
        self.encoder = FMRIEncoderMLP(m.fmri_input_dim, m.latent_dim, m.hidden)
        self.criterion = DummyCLIPLoss()
        self.save_hyperparameters()

    def training_step(self, batch, _):
        x, texts = batch
        z = self.encoder(x)
        loss = self.criterion(z, texts)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams["cfg"].train.optimizer.lr)

class CosineAlignLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)
    def forward(self, latent, text_emb_batch):
        latent = latent / (latent.norm(dim=-1, keepdim=True) + 1e-8)
        loss = 1.0 - self.cos(latent, text_emb_batch).mean()
        return loss

class LitModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, clip_text_feats: np.ndarray):
        super().__init__()
        m = cfg.train.model
        # project to CLIP text dim
        self.encoder = FMRIEncoderMLP(m.fmri_input_dim, clip_text_feats.shape[1], m.hidden)
        self.criterion = CosineAlignLoss()
        self.clip_text_feats = torch.tensor(clip_text_feats, dtype=torch.float32)
        self.save_hyperparameters()

    def training_step(self, batch, _):
        x, (idx, _texts) = batch
        z = self.encoder(x)
        text_emb = self.clip_text_feats[idx].to(z.device)  # align with correct caption
        loss = self.criterion(z, text_emb)
        self.log("train/loss", loss)
        return loss

def run_baseline(cfg: DictConfig):
    reader = NSDReader(
        cfg.data.paths.images_root,
        cfg.data.paths.fmri_root,
        cfg.data.paths.captions,
        roi_dir=cfg.data.roi.out_dir,
        subject=cfg.data.subjects[0] if "subjects" in cfg.data and cfg.data.subjects else "subj01",
    )
    X, texts = reader.load(n=64, fmri_dim=cfg.train.model.fmri_input_dim)
    dl = make_loaders(X, texts, cfg.train.batch_size, cfg.train.num_workers)
    model = LitModule(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        default_root_dir=cfg.run.output_dir,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, dl)
