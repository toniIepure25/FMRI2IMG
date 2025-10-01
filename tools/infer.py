# tools/infer.py
import argparse
import os
from pathlib import Path
import csv
import numpy as np
import torch
from omegaconf import OmegaConf

# Reuse encoder builders and small utilities from the training pipeline
from fmri2image.pipelines.baseline_train import build_encoder
from fmri2image.data.nsd_reader import NSDReader


def build_min_cfg(encoder: str,
                  fmri_input_dim: int,
                  latent_dim: int,
                  hidden=(2048, 1024, 768),
                  vit3d_cfg=None,
                  gnn_cfg=None):
    """
    Build a minimal OmegaConf cfg object compatible with `build_encoder`.
    """
    vit3d_cfg = vit3d_cfg or {"time_steps": 8, "depth": 2, "heads": 4, "mlp_ratio": 2.0, "dropout": 0.1}
    gnn_cfg = gnn_cfg or {"dropout": 0.1, "use_identity_adj": True}
    cfg = OmegaConf.create({
        "train": {
            "encoder": encoder,
            "model": {
                "fmri_input_dim": int(fmri_input_dim),
                "latent_dim": int(latent_dim),
                "hidden": list(hidden),
            },
            "vit3d": vit3d_cfg,
            "gnn": gnn_cfg,
        }
    })
    return cfg


def load_from_nsd(n: int,
                  fmri_dim: int | None = None,
                  subject: str = "subj01",
                  images_root: str = "data/raw/nsd/images",
                  fmri_root: str = "data/raw/nsd/fmri",
                  captions_csv: str = "data/raw/nsd/captions.csv",
                  roi_dir: str = "data/processed/nsd/roi",
                  clip_text_path: str = "data/processed/nsd/clip_text.npy"):
    """
    Load a quick subset from the NSD mock pipeline (same reader used for training demos).
    """
    reader = NSDReader(images_root, fmri_root, captions_csv, roi_dir=roi_dir, subject=subject)
    X, texts = reader.load(n=n, fmri_dim=fmri_dim)
    clip_feats = np.load(clip_text_path)
    # Align just in case
    m = min(len(X), len(clip_feats))
    return X[:m], texts[:m], clip_feats[:m]


def load_from_npy(x_path: str, clip_path: str, texts_csv: str | None):
    """
    Load fMRI features (X) and CLIP text features from .npy files.
    Optionally, load human-readable texts from a CSV with columns: idx,text
    """
    X = np.load(x_path)
    clip_feats = np.load(clip_path)
    texts = None
    if texts_csv and os.path.exists(texts_csv):
        idx_to_text = {}
        import pandas as pd
        df = pd.read_csv(texts_csv)
        # Expect columns: idx,text (idx must match the row order of clip embeddings)
        for _, row in df.iterrows():
            idx_to_text[int(row["idx"])] = str(row["text"])
        # Build texts list aligned to clip_feats indices
        texts = [idx_to_text.get(i, f"text_{i}") for i in range(len(clip_feats))]
    else:
        texts = [f"text_{i}" for i in range(len(clip_feats))]

    m = min(len(X), len(clip_feats))
    return X[:m], texts[:m], clip_feats[:m]


def topk_for_each_row(sim: torch.Tensor, k: int = 3):
    """
    sim: [N, M] similarity matrix (fMRI rows vs text columns)
    Returns list of tuples: [(indices, scores), ...] for each row.
    """
    k = min(k, sim.size(1))
    vals, idxs = torch.topk(sim, k=k, dim=1)
    return idxs.cpu().numpy(), vals.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Run inference: fMRI -> latent, compute text retrieval top-k.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to Lightning checkpoint (last.ckpt).")
    parser.add_argument("--encoder", type=str, default=None, choices=["mlp", "vit3d", "gnn"],
                        help="Encoder type. If omitted, will try to read from the checkpoint hyper_parameters.")
    parser.add_argument("--mode", type=str, default="nsd", choices=["nsd", "npy"],
                        help="Data source: 'nsd' quick subset or 'npy' arbitrary arrays.")
    parser.add_argument("--n", type=int, default=32, help="How many samples to load when mode=nsd.")
    parser.add_argument("--x", type=str, default=None, help="Path to X.npy (when mode=npy).")
    parser.add_argument("--clip", type=str, default=None, help="Path to clip_text.npy (when mode=npy).")
    parser.add_argument("--texts_csv", type=str, default=None, help="Optional CSV with columns: idx,text (when mode=npy).")
    parser.add_argument("--out_dir", type=str, default="reports/infer", help="Output directory for CSV artifacts.")
    parser.add_argument("--k", type=int, default=3, help="Top-k to report.")
    args = parser.parse_args()

    # ---- Load data ----
    if args.mode == "nsd":
        X, texts, clip_feats = load_from_nsd(n=args.n, fmri_dim=None)
    else:
        if not (args.x and args.clip):
            raise ValueError("When mode=npy, you must provide --x and --clip paths.")
        X, texts, clip_feats = load_from_npy(args.x, args.clip, args.texts_csv)

    fmri_dim = X.shape[1]
    latent_dim = clip_feats.shape[1]

    # ---- Determine encoder type from checkpoint hyperparameters if not provided ----
    enc_type = args.encoder
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    if enc_type is None:
        # best effort: try to dig out train.encoder from the saved cfg
        try:
            enc_type = str(hparams["cfg"]["train"]["encoder"])
        except Exception:
            enc_type = "mlp"  # fallback

    # ---- Build encoder with a minimal cfg and load weights from the checkpoint's state_dict ----
    cfg = build_min_cfg(encoder=enc_type, fmri_input_dim=fmri_dim, latent_dim=latent_dim)
    encoder = build_encoder(cfg, out_dim=latent_dim)

    # Extract only the encoder.* weights from Lightning state_dict and load into the bare encoder
    state = ckpt.get("state_dict", ckpt)
    enc_state = {k.replace("encoder.", "", 1): v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    print(f"[load] encoder weights loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    # ---- Forward & similarities ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device).eval()

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        Z = encoder(X_t)  # [N, D]
        # Normalize for cosine similarity
        Z = Z / (Z.norm(dim=-1, keepdim=True) + 1e-8)
        T = torch.tensor(clip_feats, dtype=torch.float32, device=device)
        T = T / (T.norm(dim=-1, keepdim=True) + 1e-8)
        sim = Z @ T.t()  # [N, M]
        idxs, vals = topk_for_each_row(sim, k=args.k)

    # ---- Save results ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save top-k as CSV
    csv_path = out_dir / f"top{args.k}_{enc_type}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["sample_idx"]
        for i in range(args.k):
            header += [f"top{i+1}_idx", f"top{i+1}_score", f"top{i+1}_text"]
        writer.writerow(header)

        for i in range(idxs.shape[0]):
            row = [i]
            for j in range(idxs.shape[1]):
                idx = int(idxs[i, j])
                score = float(vals[i, j])
                text_str = texts[idx] if 0 <= idx < len(texts) else ""
                row += [idx, f"{score:.4f}", text_str]
            writer.writerow(row)

    # Save latent predictions as .npy (optional, often useful downstream)
    np.save(out_dir / f"Z_{enc_type}.npy", Z.cpu().numpy())

    # Print a short preview for the first 5 samples
    print(f"[infer] wrote {csv_path}")
    for i in range(min(5, idxs.shape[0])):
        tops = []
        for j in range(idxs.shape[1]):
            tops.append(f"({idxs[i, j]}: {vals[i, j]:.3f})")
        print(f"sample {i}: top{args.k} -> " + ", ".join(tops))


if __name__ == "__main__":
    main()
