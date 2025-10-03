#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path
from fmri2image.data.nsd_reader import NSDReader

def load_texts(cfg_data_paths, roi_dir, subject, n, fmri_dim):
    reader = NSDReader(
        images_root=cfg_data_paths["images_root"],
        fmri_root=cfg_data_paths["fmri_root"],
        captions=cfg_data_paths["captions"],
        roi_dir=roi_dir,
        subject=subject,
    )
    # Load exactly n samples to keep indices consistent with infer.py default
    X, texts = reader.load(n=n, fmri_dim=fmri_dim)
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk_csv", required=True, help="Path to merged or single encoder top-k CSV (long format)")
    ap.add_argument("--encoder",  default=None,   help="Optional: filter by one encoder (mlp/vit3d/gnn)")
    ap.add_argument("--n",        type=int, default=32, help="How many samples were used in infer.py (default 32)")
    ap.add_argument("--fmri_dim", type=int, default=2048, help="ROI vector dim passed during training/infer")
    ap.add_argument("--subject",  default="subj01")
    # Hardcode your config paths (match your config)
    ap.add_argument("--images_root", default="data/raw/nsd/images")
    ap.add_argument("--fmri_root",   default="data/raw/nsd/fmri")
    ap.add_argument("--captions",    default="data/raw/nsd/captions.csv")
    ap.add_argument("--roi_dir",     default="data/processed/nsd/roi")
    ap.add_argument("--head",        type=int, default=5, help="How many samples to display")
    args = ap.parse_args()

    # Load texts aligned with infer data
    cfg_data_paths = {
        "images_root": args.images_root,
        "fmri_root": args.fmri_root,
        "captions": args.captions,
    }
    texts = load_texts(cfg_data_paths, args.roi_dir, args.subject, n=args.n, fmri_dim=args.fmri_dim)

    df = pd.read_csv(args.topk_csv)
    if args.encoder:
        df = df[df["encoder"] == args.encoder].copy()
    # Expect long format: sample, rank, text_idx, score, encoder
    if not {"sample", "rank", "text_idx", "score"}.issubset(df.columns):
        raise ValueError("Expected columns [sample, rank, text_idx, score]")

    # Print first N samples with captions for top-k
    for s in sorted(df["sample"].unique())[:args.head]:
        block = df[df["sample"] == s].sort_values(["rank"])
        print(f"\n=== Sample {s} ({args.encoder or 'all encoders'}) ===")
        for _, row in block.iterrows():
            t_idx = int(row["text_idx"])
            score = float(row["score"])
            enc   = row["encoder"] if "encoder" in row else "enc"
            cap = texts[t_idx] if 0 <= t_idx < len(texts) else "<out of range>"
            print(f"[{enc}] rank {int(row['rank'])} -> text_idx={t_idx:>3d}, score={score:+.3f} :: {cap}")

if __name__ == "__main__":
    main()
