import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
import numpy as np

from fmri2image.models.encoders.mlp_encoder import FMRIEncoderMLP
from fmri2image.losses.clip_losses import ClipStyleContrastiveLoss
from fmri2image.losses.cca import DeepCCALoss


class BaselineModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, clip_text_feats: np.ndarray = None):
        super().__init__()
        m = cfg.train.model

        # Encoder simplu (MLP, dar poate fi extins cu vit3d/gnn)
        self.encoder = FMRIEncoderMLP(m.fmri_input_dim, m.latent_dim, m.hidden)

        # Loss principal (contrastive + opțional Deep CCA)
        self.criterion = ClipStyleContrastiveLoss(
            temperature_init=cfg.train.loss.temperature_init,
            symmetric=cfg.train.loss.symmetric,
        )

        self.cca = None
        if cfg.train.cca.enabled:
            self.cca = DeepCCALoss(proj_dim=cfg.train.cca.proj_dim)

        # Dacă avem embeddings CLIP salvate, le punem ca buffer
        if clip_text_feats is not None:
            self.register_buffer(
                "clip_text_feats",
                torch.tensor(clip_text_feats, dtype=torch.float32),
                persistent=False,
            )

        self.save_hyperparameters()

    def training_step(self, batch, _):
        x, (idx, _texts) = batch
        z = self.encoder(x)
        text_emb = self.clip_text_feats.index_select(0, idx)
        out = self.criterion(z, text_emb)
        if self.cca is not None:
            out["loss"] += self.cca(z, text_emb) * self.hparams["cfg"].train.loss.weights.cca

        bs = x.size(0)
        self.log("train/loss", out["loss"], prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train/temp", out["temp"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)
        return out["loss"]

    def validation_step(self, batch, _):
        x, (idx, _texts) = batch
        z = self.encoder(x)
        text_emb = self.clip_text_feats.index_select(0, idx)
        out = self.criterion(z, text_emb)

        bs = x.size(0)
        self.log("val/loss", out["loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val/temp", out["temp"], prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)
        return out["loss"]

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams["cfg"].train.optimizer.lr)
