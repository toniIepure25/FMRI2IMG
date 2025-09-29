#!/usr/bin/env python3
"""
Run ablations for encoders: mlp, vit3d, gnn
- compune config-ul Hydra (fără a schimba cwd)
- rulează training cu CSVLogger
- citește metri ce din csv și construiește summary.csv

Rezultate:
- outputs/ablations/<encoder>/metrics.csv         (Lightning CSV)
- outputs/ablations/summary.csv                   (rezumat)
"""
from __future__ import annotations

import os
import csv
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import pytorch_lightning as pl

# Hydra + OmegaConf
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf, DictConfig

# Importăm direct componentele din pipeline-ul tău
from fmri2image.pipelines.baseline_train import (
    LitModule,
    build_encoder,   # not used directly, kept for clarity
)
from fmri2image.data.nsd_reader import NSDReader
from fmri2image.data.datamodule import make_loaders


def run_once(cfg: DictConfig, encoder_name: str, out_root: Path) -> Dict[str, float]:
    """
    Rulează un training pentru encoderul 'encoder_name' și returnează metricele finale.
    """
    # Override encoder în config (fără a pierde restul)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"train": {"encoder": encoder_name}}))

    # ==== Data ====
    reader = NSDReader(
        cfg.data.paths.images_root,
        cfg.data.paths.fmri_root,
        cfg.data.paths.captions,
        roi_dir=cfg.data.roi.out_dir,
        subject=cfg.data.subjects[0] if "subjects" in cfg.data and cfg.data.subjects else "subj01",
    )
    X, texts = reader.load(n=64, fmri_dim=cfg.train.model.fmri_input_dim)

    clip_feats_path = Path("data/processed/nsd/clip_text.npy")
    if not clip_feats_path.exists():
        raise FileNotFoundError(f"Missing CLIP text features at: {clip_feats_path}")
    clip_feats = np.load(clip_feats_path)

    n = min(len(X), len(clip_feats))
    X, texts, clip_feats = X[:n], texts[:n], clip_feats[:n]

    dl = make_loaders(X, texts, cfg.train.batch_size, cfg.train.num_workers)

    # ==== Model ====
    model = LitModule(cfg, clip_feats)

    # ==== Logger CSV (un folder per encoder) ====
    # outputs/ablations/<encoder>/
    save_dir = out_root / encoder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    from pytorch_lightning.loggers import CSVLogger
    csv_logger = CSVLogger(save_dir=str(save_dir), name="", version=".")

    # ==== Trainer ====
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        default_root_dir=str(save_dir),
        enable_checkpointing=False,
        logger=csv_logger,
    )
    trainer.fit(model, dl)

    # ==== Citește ultimele valori din CSV ====
    metrics_file = save_dir / "metrics.csv"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Expected metrics.csv at {metrics_file}")

    last_row: Dict[str, Any] = {}
    with metrics_file.open("r", newline="") as f:
        reader_csv = csv.DictReader(f)
        rows = list(reader_csv)
        if not rows:
            raise RuntimeError(f"No rows in {metrics_file}")
        last_row = rows[-1]

    # Extrage chei utile; fallback la None dacă lipsesc
    def get_float(key: str) -> float | None:
        val = last_row.get(key, None)
        try:
            return float(val) if val is not None and val != "" else None
        except Exception:
            return None

    summary = {
        "encoder": encoder_name,
        "train/loss_epoch": get_float("train/loss_epoch"),
        "train/retrieval_zt_top1": get_float("train/retrieval_zt_top1_epoch"),
        "train/retrieval_zt_top5": get_float("train/retrieval_zt_top5_epoch"),
        # temperatură medie pe epocă (opțional)
        "train/temp_epoch": get_float("train/temp_epoch"),
    }
    return summary


def main():
    # Respectă WANDB_DISABLED by default
    os.environ.setdefault("WANDB_DISABLED", "true")

    # Compunem config-ul principal folosind Hydra fără a schimba CWD
    # `config_name="config"` -> configs/config.yaml
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]  # .../fmri2image
    configs_dir = repo_root / "configs"

    if not configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found at {configs_dir}")

    # Hidra: nu schimba directorul de lucru
    with initialize_config_dir(config_dir=str(configs_dir), version_base=None):
        cfg = compose(config_name="config")
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = OmegaConf.create(cfg)

    out_root = repo_root / "outputs" / "ablations"
    out_root.mkdir(parents=True, exist_ok=True)

    encoders = ["mlp", "vit3d", "gnn"]
    results: List[Dict[str, Any]] = []

    for enc in encoders:
        print(f"\n[ABLATION] Running encoder={enc} ...")
        res = run_once(cfg, enc, out_root)
        print(f"[ABLATION] Done encoder={enc}: "
              f"loss={res.get('train/loss_epoch')}, "
              f"top1={res.get('train/retrieval_zt_top1')}, "
              f"top5={res.get('train/retrieval_zt_top5')}")
        results.append(res)

    # Scrie summary.csv
    summary_path = out_root / "summary.csv"
    fieldnames = ["encoder", "train/loss_epoch", "train/retrieval_zt_top1",
                  "train/retrieval_zt_top5", "train/temp_epoch"]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n[ABLATION] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
