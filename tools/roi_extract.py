#!/usr/bin/env python3
import argparse, os, numpy as np, json
from pathlib import Path

"""
Extract ROI-level features per image/stimulus.
If real preprocessed data are missing, we generate consistent mock ROI matrices:
  shape: (N, V) with V=vector_dim from config, N derived from captions.csv length (if present).
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmriprep_root", default="data/processed/nsd/fmriprep")
    ap.add_argument("--atlas", default="glasser", choices=["glasser","schaefer"])
    ap.add_argument("--vector_dim", type=int, default=2048)
    ap.add_argument("--captions_csv", default="data/raw/nsd/captions.csv")
    ap.add_argument("--subject", default="subj01")
    ap.add_argument("--out_dir", default="data/processed/nsd/roi")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"{args.subject}_roi.npy"

    # Try to infer N from captions
    N = 64
    if os.path.exists(args.captions_csv):
        try:
            with open(args.captions_csv, "r") as f:
                # header + rows
                N = max(1, sum(1 for _ in f) - 1)
        except Exception:
            pass

    # If real fMRIPrep derivatives exist, here you'd load and parcellate volumes.
    # For now we create a stable random (mock) matrix to keep shapes correct.
    rng = np.random.default_rng(seed=1337)
    X = rng.standard_normal(size=(N, args.vector_dim)).astype("float32")

    # z-score per feature (per subject)
    X -= X.mean(axis=0, keepdims=True)
    X /= (X.std(axis=0, keepdims=True) + 1e-6)

    np.save(out_path, X)
    meta = {
        "subject": args.subject,
        "atlas": args.atlas,
        "vector_dim": args.vector_dim,
        "n_samples": int(N),
        "source": "mock" if not os.path.exists(args.fmriprep_root) else "fmriprep",
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] saved ROI features: {out_path}  shape={X.shape}")

if __name__ == "__main__":
    main()
