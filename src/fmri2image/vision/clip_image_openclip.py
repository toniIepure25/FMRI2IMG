# src/fmri2image/vision/clip_image_openclip.py
import argparse
import os
from pathlib import Path
import numpy as np
import pickle
from PIL import Image

import torch

def list_images(root: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root_p = Path(root)
    if not root_p.exists():
        return []
    files = []
    for p in root_p.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)

def save_meta(meta_path: str, items):
    meta = {"paths": [str(p) for p in items]}
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--out_npy", required=True)
    parser.add_argument("--out_pkl", required=True)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()

    paths = list_images(args.images_root)

    # Graceful no-op: if no images, write empty artifacts and exit(0)
    if len(paths) == 0:
        os.makedirs(Path(args.out_npy).parent, exist_ok=True)
        np.save(args.out_npy, np.zeros((0, 512), dtype=np.float32))  # assume 512-d OpenCLIP
        save_meta(args.out_pkl, [])
        print(f"[warn] no images under {args.images_root}; wrote empty embeddings -> {args.out_npy} and meta -> {args.out_pkl}")
        return

    # Lazy import OpenCLIP only if we actually need it
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    model.eval()

    feats = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            emb = model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.float().cpu().numpy())

    feats = np.concatenate(feats, axis=0)  # [N, D]
    os.makedirs(Path(args.out_npy).parent, exist_ok=True)
    np.save(args.out_npy, feats)
    save_meta(args.out_pkl, paths)
    print(f"[ok] wrote image embeddings -> {args.out_npy} ; meta -> {args.out_pkl} ; shape={feats.shape}")

if __name__ == "__main__":
    main()
