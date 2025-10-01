import optuna
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from fmri2image.pipelines.baseline_train import LitModule
from fmri2image.data.nsd_reader import NSDReader
from fmri2image.data.datamodule import make_loaders

def objective(trial: optuna.Trial) -> float:
    # --- Search space ---
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    temp_init = trial.suggest_float("temperature_init", 0.01, 0.2)
    cca_weight = trial.suggest_float("cca_weight", 0.0, 0.2)
    latent_dim = trial.suggest_categorical("latent_dim", [512, 768, 1024])

    # --- Load and mutate config ---
    cfg = OmegaConf.load("configs/train/baseline.yaml")
    cfg.train.optimizer.lr = lr
    cfg.train.loss.temperature_init = temp_init
    cfg.train.loss.weights.cca = cca_weight
    cfg.train.model.latent_dim = latent_dim
    cfg.run.name = f"hpo_lr{lr:.2e}_t{temp_init:.2f}_cca{cca_weight:.2f}_d{latent_dim}"

    # --- Data ---
    reader = NSDReader(
        cfg.data.paths.images_root,
        cfg.data.paths.fmri_root,
        cfg.data.paths.captions,
        roi_dir=cfg.data.roi.out_dir,
        subject="subj01",
    )
    X, texts = reader.load(n=128, fmri_dim=cfg.train.model.fmri_input_dim)
    clip_feats = np.load("data/processed/nsd/clip_text.npy")
    n = min(len(X), len(clip_feats))
    X, texts, clip_feats = X[:n], texts[:n], clip_feats[:n]
    dl = make_loaders(X, texts, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)

    # --- Model & Trainer ---
    model = LitModule(cfg, clip_feats)
    trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    trainer.fit(model, dl)

    # maximize retrieval@5
    return float(trainer.callback_metrics.get("train/retrieval_zt_top5", 0.0))

def main():
    study = optuna.create_study(direction="maximize", study_name="fmri2image_hpo")
    study.optimize(objective, n_trials=10)
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)

if __name__ == "__main__":
    main()
