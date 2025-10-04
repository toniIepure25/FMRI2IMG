import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from fmri2image.embeddings.clip_utils import load_openclip, encode_texts, save_embeddings_npy_pkl


def read_captions_csv(path: str):
    """
    Read captions from CSV. Tries common column names: 'caption', 'text', 'sentence'.
    Falls back to the first non-index column if needed.
    """
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    for cand in ["caption", "text", "sentence"]:
        if cand in cols:
            return df[df.columns[cols.index(cand)]].astype(str).tolist()
    # fallback: first non-index column
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            return df[c].astype(str).tolist()
    # final fallback: cast first column
    return df.iloc[:, 0].astype(str).tolist()


def main():
    ap = argparse.ArgumentParser(description="OpenCLIP Text Embedding Generator")
    ap.add_argument("--captions", type=str, required=True, help="Path to captions.csv")
    ap.add_argument("--out_npy", type=str, required=True, help="Output .npy for embeddings")
    ap.add_argument("--out_pkl", type=str, required=True, help="Output .pkl metadata")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    texts = read_captions_csv(args.captions)
    model, tokenizer, _pre, device = load_openclip(args.model, args.pretrained, args.device)
    emb = encode_texts(model, tokenizer, texts, device=device, batch_size=args.batch_size, normalize=True)

    # Save
    save_embeddings_npy_pkl(
        args.out_npy,
        args.out_pkl,
        emb,
        meta={
            "num_texts": len(texts),
            "model": args.model,
            "pretrained": args.pretrained,
            "device": device,
        },
    )
    print(f"[ok] wrote text embeddings -> {args.out_npy} ; meta -> {args.out_pkl} ; shape={emb.shape}")


if __name__ == "__main__":
    main()
