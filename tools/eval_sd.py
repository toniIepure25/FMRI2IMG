#!/usr/bin/env python
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
import pandas as pd
import open_clip


def load_clip(device: str = "cuda", model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def read_prompts(prompts_csv: Path) -> dict[int, str]:
    """
    Expect a CSV with at least two columns:
      sample,prompt
    """
    df = pd.read_csv(prompts_csv)
    if not {"sample", "prompt"}.issubset(df.columns):
        raise ValueError(f"{prompts_csv} must have columns: sample,prompt")
    mapping = {int(row["sample"]): str(row["prompt"]) for _, row in df.iterrows()}
    return mapping


@torch.inference_mode()
def clip_score_dir(
    images_dir: Path,
    prompts_csv: Path,
    device: str = "cuda",
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    out_csv: Path = None,
    out_txt: Path = None,
):
    model, preprocess, tokenizer = load_clip(device=device, model_name=model_name, pretrained=pretrained)

    # read prompts
    sample2prompt = read_prompts(prompts_csv)

    # collect images (skip grid.png)
    imgs = sorted([p for p in images_dir.glob("*.png") if p.name != "grid.png"])
    if len(imgs) == 0:
        raise FileNotFoundError(f"No per-sample PNGs found in {images_dir}")

    rows = []
    sims = []
    for img_path in imgs:
        # filename assumed pattern sample_###.png
        stem = img_path.stem
        # try to parse last number
        try:
            sid = int(stem.split("_")[-1])
        except Exception:
            # fallback: try everything after 'sample'
            s = stem.replace("sample", "").replace("_", "")
            sid = int(s)

        prompt = sample2prompt.get(sid, None)
        if prompt is None:
            # fallback to empty prompt to avoid crash
            prompt = ""

        # encode image
        img = Image.open(img_path).convert("RGB")
        img_t = preprocess(img).unsqueeze(0).to(device)

        # encode text
        tokens = tokenizer([prompt]).to(device)
        img_feat = model.encode_image(img_t)
        txt_feat = model.encode_text(tokens)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ txt_feat.T).squeeze().item()

        sims.append(sim)
        rows.append({"image": img_path.name, "sample": sid, "prompt": prompt, "clip_score": sim})

    df = pd.DataFrame(rows).sort_values("sample")
    mean_score = float(df["clip_score"].mean())
    std_score = float(df["clip_score"].std(ddof=0))

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    if out_txt is not None:
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"n={len(df)}\n")
            f.write(f"CLIPScore mean={mean_score:.4f}, std={std_score:.4f}\n")

    print(f"[ok] {images_dir.name}: CLIPScore mean={mean_score:.4f} (n={len(df)})")
    return df, mean_score, std_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Folder with generated PNGs (contains grid.png)")
    ap.add_argument("--prompts_csv", required=True, help="CSV with columns: sample,prompt")
    ap.add_argument("--encoder", required=True, help="Tag for output naming (mlp|vit3d|gnn)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_name", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--out_dir", default="reports/sd_eval")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    prompts_csv = Path(args.prompts_csv)
    out_dir = Path(args.out_dir)

    out_csv = out_dir / f"{args.encoder}_clipscores.csv"
    out_txt  = out_dir / f"{args.encoder}_summary.txt"

    clip_score_dir(
        images_dir=images_dir,
        prompts_csv=prompts_csv,
        device=args.device,
        model_name=args.model_name,
        pretrained=args.pretrained,
        out_csv=out_csv,
        out_txt=out_txt,
    )


if __name__ == "__main__":
    main()
