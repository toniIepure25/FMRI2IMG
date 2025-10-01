# tools/eval.py
import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf, DictConfig
from hydra import main as hydra_main

# Modelul Lightning folosit la train
from fmri2image.pipelines.baseline_train import LitModule
from fmri2image.data.nsd_reader import NSDReader

# ===== Dataset simplu de evaluare (identic cu train, dar fără drop_last/shuffle) =====
class FMRITextDataset(Dataset):
    def __init__(self, X, texts):
        self.X = X
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        return x, (torch.tensor(idx, dtype=torch.long), self.texts[idx])


def topk_retrieval(sim: torch.Tensor, topk=(1, 5)):
    """
    sim: [N, N] cosine similarity (z @ t^T) between predicted fmri latents (rows) and text latents (cols)
    target = diagonal index
    """
    N = sim.size(0)
    device = sim.device
    target = torch.arange(N, device=device)
    out = {}
    for k in topk:
        kk = min(k, N)
        _, idxs = sim.topk(kk, dim=1, largest=True)
        correct = (idxs == target.view(-1, 1)).any(dim=1).float().mean()
        out[f"top{kk}"] = correct.item()
    return out


def _load_ckpt(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    # Dacă cheile sunt prefixate cu "model."/"module." etc, încearcă să potrivim automat
    model_keys = set(model.state_dict().keys())
    filtered = {}
    for k, v in state.items():
        kk = k
        if kk.startswith("model."):
            kk = kk[len("model.") :]
        if kk.startswith("encoder.") or kk.startswith("criterion.") or kk.startswith("cca."):
            # lăsăm așa, modelul Lightning are aceleași submodule
            pass
        # adaugă dacă e cheie validă
        if kk in model_keys:
            filtered[kk] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[eval] loaded checkpoint: {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})")


@hydra_main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Asigură-te că avem secțiunile necesare (fallback-uri safe)
    cfg = OmegaConf.merge(
        OmegaConf.create(
            {
                "train": {"batch_size": 2, "num_workers": 0, "model": {"fmri_input_dim": 2048}},
                "eval": {"topk": [1, 5], "checkpoint": ""},
                "paths": {
                    "data_dir": "./data",
                    "raw_dir": "./data/raw",
                    "proc_dir": "./data/processed",
                    "artifacts_dir": "./data/artifacts",
                },
                "data": {
                    "paths": {
                        "images_root": "./data/raw/nsd/images",
                        "fmri_root": "./data/raw/nsd/fmri",
                        "captions": "./data/raw/nsd/captions.csv",
                    },
                    "roi": {"out_dir": "./data/processed/nsd/roi"},
                    "subjects": ["subj01"],
                },
            }
        ),
        cfg,
    )

    # ---- 1) Încarcă datele + CLIP text feats ----
    reader = NSDReader(
        cfg.data.paths.images_root,
        cfg.data.paths.fmri_root,
        cfg.data.paths.captions,
        roi_dir=cfg.data.roi.out_dir,
        subject=cfg.data.subjects[0] if cfg.data.get("subjects") else "subj01",
    )
    # Folosim același subset ca la train demo (poți crește n)
    X, texts = reader.load(n=64, fmri_dim=cfg.train.model.fmri_input_dim)

    clip_feats = np.load(os.path.join(cfg.paths.proc_dir, "nsd", "clip_text.npy"))
    n = min(len(X), len(clip_feats))
    X, texts, clip_feats = X[:n], texts[:n], clip_feats[:n]

    ds = FMRITextDataset(X, texts)
    # la evaluare: fără shuffle, fără drop_last
    dl = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=bool(cfg.train.num_workers > 0),
        drop_last=False,
    )

    # ---- 2) Construiește modelul și încarcă checkpoint-ul ----
    model = LitModule(cfg, clip_feats)
    ckpt_path = cfg.eval.checkpoint
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Nu găsesc checkpoint-ul: '{ckpt_path}'. Dă-mi calea corectă prin "
            f"`eval.checkpoint=...` sau rulează un train cu checkpointing activat."
        )
    _load_ckpt(model, ckpt_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- 3) Encodează toate eșantioanele, calculează retrieval top-k pe setul complet ----
    with torch.no_grad():
        all_z = []
        all_t = []
        for x, (idx, _) in dl:
            x = x.to(device)
            idx = idx.to(device).long()
            z = model.encoder(x)  # [B, D]
            t = model.clip_text_feats.index_select(0, idx)  # [B, D]
            # normalize (ca în train)
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
            t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
            all_z.append(z)
            all_t.append(t)

        Z = torch.cat(all_z, dim=0)  # [N, D]
        T = torch.cat(all_t, dim=0)  # [N, D]
        sim = Z @ T.t()              # cosine (unit-norm)
        metrics = topk_retrieval(sim, tuple(cfg.eval.topk))

    # ---- 4) Afișează rezultatele ----
    print("\n=== Retrieval metrics (full set) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # (opțional) salvează un fișier cu rezultate
    out_dir = os.path.join("reports", "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, "retrieval.txt")
    with open(out_txt, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"\n[eval] Am scris rezultatele în: {out_txt}")


if __name__ == "__main__":
    main()
