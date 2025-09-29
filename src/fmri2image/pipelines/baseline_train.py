from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from ..models.encoders.mlp_encoder import FMRIEncoderMLP
from ..models.encoders.vit3d_encoder import ViT3DEncoderLite
from ..models.encoders.gnn_encoder import GraphMLPEncoderLite

from ..data.nsd_reader import NSDReader
from ..data.datamodule import make_loaders
from .metrics import topk_retrieval


def build_encoder(cfg: DictConfig, out_dim: int) -> nn.Module:
    """
    Select and construct the encoder according to cfg.train.encoder.
    Supported: 'mlp' (default), 'vit3d', 'gnn'.
    """
    enc_type = str(getattr(cfg.train, "encoder", "mlp")).lower()
    m = cfg.train.model

    if enc_type == "mlp":
        return FMRIEncoderMLP(m.fmri_input_dim, out_dim, m.hidden)

    elif enc_type == "vit3d":
        v = cfg.train.vit3d
        # NOTE: ViT3D-lite expects fmri_input_dim % time_steps == 0
        return ViT3DEncoderLite(
            in_dim=m.fmri_input_dim,
            out_dim=out_dim,
            time_steps=int(getattr(v, "time_steps", 8)),
            depth=int(getattr(v, "depth", 2)),
            heads=int(getattr(v, "heads", 4)),
            mlp_ratio=float(getattr(v, "mlp_ratio", 2.0)),
            dropout=float(getattr(v, "dropout", 0.1)),
        )

    elif enc_type == "gnn":
        g = cfg.train.gnn
        enc = GraphMLPEncoderLite(
            in_dim=m.fmri_input_dim,
            out_dim=out_dim,
            hidden=m.hidden,
            dropout=float(getattr(g, "dropout", 0.1)),
        )
        # Mock adjacency: identity (no edges) unless you switch to another scheme
        use_identity = bool(getattr(g, "use_identity_adj", True))
        if use_identity:
            A = torch.zeros(m.fmri_input_dim, m.fmri_input_dim)
        else:
            # simple example: self-loops (you can customize later)
            A = torch.eye(m.fmri_input_dim, m.fmri_input_dim)
        enc.set_adj(A)
        return enc

    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")


class ClipStyleContrastiveLoss(nn.Module):
    """
    CLIP-style InfoNCE with learnable temperature (logit scale).
    Uses in-batch negatives. Optionally symmetric (z->t and t->z).
    """
    def __init__(self, temperature_init: float = 0.07, symmetric: bool = True):
        super().__init__()
        # CLIP uses a learnable logit scale initialized to log(1/temperature)
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1.0 / float(temperature_init)), dtype=torch.float32)
        )
        self.symmetric = bool(symmetric)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> dict:
        """
        Args:
            z: [B, D] fMRI->latent predictions
            t: [B, D] normalized CLIP text embeddings
        Returns:
            dict(loss=..., logits_zt=..., logits_tz|None=..., temp=...)
        """
        # Normalize both (t should already be normalized, but we re-normalize safely)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)

        # temperature = 1 / softmax temperature; we multiply similarities by exp(logit_scale)
        logit_scale = self.logit_scale.clamp(-5.0, 8.0)  # clamp for numerical stability
        scale = torch.exp(logit_scale)

        # Similarities (cosine since vectors are unit norm) scaled by temperature
        logits_zt = (z @ t.t()) * scale  # [B, B]
        target = torch.arange(z.size(0), device=z.device)
        loss_zt = nn.functional.cross_entropy(logits_zt, target)

        if self.symmetric:
            logits_tz = (t @ z.t()) * scale
            loss_tz = nn.functional.cross_entropy(logits_tz, target)
            loss = 0.5 * (loss_zt + loss_tz)
        else:
            logits_tz = None
            loss = loss_zt

        # Report the *temperature* for logging (inverse of scale)
        temp = torch.exp(-logit_scale)
        return {"loss": loss, "logits_zt": logits_zt, "logits_tz": logits_tz, "temp": temp}


class LitModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, clip_text_feats: np.ndarray):
        super().__init__()
        self.cfg = cfg
        m = cfg.train.model

        # Project fMRI -> CLIP text embedding dimension
        out_dim = int(clip_text_feats.shape[1])
        self.encoder = build_encoder(cfg, out_dim)

        self.criterion = ClipStyleContrastiveLoss(
            temperature_init=float(cfg.train.loss.temperature_init),
            symmetric=bool(cfg.train.loss.get("symmetric", True)),
        )

        # Register CLIP text features as a non-persistent buffer (moved with device)
        self.register_buffer(
            "clip_text_feats",
            torch.tensor(clip_text_feats, dtype=torch.float32),
            persistent=False,
        )

        self.topk = tuple(getattr(getattr(cfg.train, "eval", {}), "topk", [1, 5]))
        # Save hyperparameters except the big array (stored as buffer)
        try:
            self.save_hyperparameters({"cfg": cfg})
        except Exception:
            pass

    def training_step(self, batch, _):
        x, (idx, _texts) = batch                 # x: [B, in_dim], idx: [B]
        z = self.encoder(x)                      # [B, D]
        t = self.clip_text_feats.index_select(0, idx.long())  # [B, D] on correct device

        out = self.criterion(z, t)

        # --- Logging with explicit batch_size (fixes PL warning) ---
        bs = x.size(0)
        self.log("train/loss", out["loss"], prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train/temp", out["temp"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)

        # --- Retrieval metrics within-batch (ranking unaffected by temperature) ---
        with torch.no_grad():
            # remove temperature for ranking
            sim_zt = out["logits_zt"] / torch.exp(self.criterion.logit_scale)
            m_zt = topk_retrieval(sim_zt, self.topk)
            for k, v in m_zt.items():
                self.log(f"train/retrieval_zt_{k}", v, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)

            if out["logits_tz"] is not None:
                sim_tz = out["logits_tz"] / torch.exp(self.criterion.logit_scale)
                m_tz = topk_retrieval(sim_tz, self.topk)
                for k, v in m_tz.items():
                    self.log(f"train/retrieval_tz_{k}", v, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)

        return out["loss"]

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=float(self.cfg.train.optimizer.lr))


def run_baseline(cfg: DictConfig):
    # ---- Data ----
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
    # Keep arrays aligned
    n = min(len(X), len(clip_feats))
    X, texts, clip_feats = X[:n], texts[:n], clip_feats[:n]

    dl = make_loaders(X, texts, cfg.train.batch_size, cfg.train.num_workers)
    model = LitModule(cfg, clip_feats)

    # Optional: logger (W&B disabled by default in your config)
    logger = False
    try:
        if getattr(cfg.wandb, "enabled", False):
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(
                project=getattr(cfg.wandb, "project", "fmri2image"),
                entity=getattr(cfg.wandb, "entity", None),
                log_model=False,
            )
    except Exception:
        logger = False

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        default_root_dir=cfg.run.output_dir,
        enable_checkpointing=False,
        logger=logger,
    )
    trainer.fit(model, dl)
