from pathlib import Path
import argparse
import torch
import numpy as np
import pandas as pd
import open_clip

@torch.no_grad()
def main(captions_csv: str, out_path: str,
         model_name: str = "ViT-B-32",
         pretrained: str = "laion2b_s34b_b79k",
         device: str = "cuda"):
    df = pd.read_csv(captions_csv)
    texts = df["caption"].astype(str).tolist()

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    toks = tokenizer(texts).to(device)
    feats = model.encode_text(toks)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, feats.float().cpu().numpy())
    print(f"[ok] saved CLIP text features -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_name", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(args.captions, args.out, args.model_name, args.pretrained, args.device)
