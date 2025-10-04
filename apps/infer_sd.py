# apps/infer_sd.py
"""
E2E baseline: fMRI -> CLIP-text latent -> prompt -> Stable Diffusion (text-to-image).
- Loads encoder weights from a baseline checkpoint (MLP/GNN/ViT3D).
- Uses top-k nearest captions from clip_text.npy (+ meta .pkl) to build prompts.
- Generates images with Diffusers (SD 1.5 or SDXL), saves per-sample and a grid.
"""
from __future__ import annotations
import argparse, os, pickle
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image

# --- imports from your codebase ---
from fmri2image.pipelines.baseline_train import build_encoder  # reuse dimension logic
from fmri2image.data.nsd_reader import NSDReader
from fmri2image.generation.sd_wrapper import SDConfig, StableDiffusionGenerator

def cosine_topk(x: np.ndarray, Y: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices and scores of top-k by cosine between single x and rows of Y."""
    x = x / (np.linalg.norm(x) + 1e-8)
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    sims = Y @ x  # [N]
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def make_prompt(captions: List[str]) -> str:
    """Very simple prompt template; refine later if you want."""
    best = captions[0] if captions else "a photo"
    alt = [c for c in captions[1:3] if c]
    extras = (". Alternatives: " + "; ".join(alt)) if alt else ""
    return f"A high-quality photo: {best}{extras}"

def save_grid(pil_images: List[Image.Image], out_path: str, nrow: int = 4):
    tensors = []
    for im in pil_images:
        arr = np.array(im)
        if arr.ndim == 2:  # grayscale -> RGB
            arr = np.stack([arr]*3, axis=-1)
        t = torch.from_numpy(arr).permute(2, 0, 1)  # HWC->CHW
        tensors.append(t)
    grid = vutils.make_grid(torch.stack(tensors), nrow=nrow)
    grid = grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(grid).save(out_path)

def load_encoder_from_ckpt(ckpt_path: str, encoder_type: str, fmri_input_dim: int, clip_dim: int):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # minimal cfg shim
    class CfgObj: ...
    cfg = CfgObj()
    cfg.train = CfgObj()
    cfg.train.encoder = encoder_type
    cfg.train.model = CfgObj()
    cfg.train.model.fmri_input_dim = fmri_input_dim

    encoder = build_encoder(cfg, out_dim=clip_dim)
    state = ckpt.get("state_dict", ckpt)
    enc_state = {k.split("encoder.", 1)[1]: v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    print(f"[load] encoder weights loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    return encoder

def load_caption_bank(meta_pkl_path: str) -> Optional[List[str]]:
    """
    Load captions bank aligned to clip_text.npy indices.
    Accepts either:
      - dict with key 'captions' (preferred), or
      - a plain list of strings.
    """
    if not os.path.exists(meta_pkl_path):
        print(f"[warn] caption meta not found: {meta_pkl_path}")
        return None
    try:
        with open(meta_pkl_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "captions" in obj:
            caps = obj["captions"]
            if isinstance(caps, list):
                return [str(c) for c in caps]
        if isinstance(obj, list):
            return [str(c) for c in obj]
        print(f"[warn] unrecognized meta schema in {meta_pkl_path}; falling back")
        return None
    except Exception as e:
        print(f"[warn] failed to load caption meta ({meta_pkl_path}): {e}")
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--encoder", choices=["mlp", "vit3d", "gnn"], default="mlp")
    p.add_argument("--subject", default="subj01")
    p.add_argument("--n", type=int, default=8, help="number of fMRI samples to infer")
    p.add_argument("--k", type=int, default=3, help="top-k captions for prompt building")
    p.add_argument("--out_dir", required=True)
    # SD params
    p.add_argument("--sd_model", default="sd15", choices=["sd15", "sdxl"])
    p.add_argument("--sd_steps", type=int, default=25)
    p.add_argument("--sd_guidance", type=float, default=7.5)
    p.add_argument("--sd_height", type=int, default=512)
    p.add_argument("--sd_width", type=int, default=512)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- data & clip text ---
    reader = NSDReader(
        images_root="data/raw/nsd/images",
        fmri_root="data/raw/nsd/fmri",
        captions="data/raw/nsd/captions.csv",  # <-- FIXED: 'captions' not 'captions_path'
        roi_dir="data/processed/nsd/roi",
        subject=args.subject,
    )
    fmri_dim = 2048  # must match your ROI vector dim
    X, texts = reader.load(n=max(args.n, 8), fmri_dim=fmri_dim)
    X = X[:args.n]; texts = texts[:args.n]

    clip_text = np.load("data/processed/nsd/clip_text.npy")  # [M, D]
    clip_dim = clip_text.shape[1]
    # try to load caption bank aligned with clip_text rows
    captions_bank = load_caption_bank("data/processed/nsd/clip_text_meta.pkl")

    # --- encoder ---
    encoder = load_encoder_from_ckpt(args.checkpoint, args.encoder, fmri_input_dim=fmri_dim, clip_dim=clip_dim)
    encoder.eval().to(device)

    # --- sd pipeline ---
    sd_cfg = SDConfig(
        model=args.sd_model,
        device=device,
        dtype="fp16" if device == "cuda" else "fp32",
        height=args.sd_height,
        width=args.sd_width,
        guidance_scale=args.sd_guidance,
        num_inference_steps=args.sd_steps,
        negative_prompt="blurry, low quality, low resolution, artifacts, distorted, deformed, text, watermark"
    )
    sd = StableDiffusionGenerator(sd_cfg)

    # --- infer loop ---
    prompts: List[str] = []
    rng = np.random.default_rng(args.seed)

    with torch.inference_mode():
        for i in range(len(X)):
            x = torch.tensor(X[i], dtype=torch.float32, device=device).unsqueeze(0)
            z = encoder(x).float().cpu().numpy()[0]  # [D]
            idx, _scores = cosine_topk(z, clip_text, k=args.k)

            # choose captions from bank if available, else fallback to local `texts`
            chosen = []
            for j, ix in enumerate(idx):
                if captions_bank and 0 <= int(ix) < len(captions_bank):
                    chosen.append(str(captions_bank[int(ix)]))
                else:
                    # local small fallback (may not be globally aligned, but avoids crash)
                    chosen.append(texts[min(j, len(texts)-1)])
            prompt = make_prompt(chosen)
            prompts.append(prompt)

    # Generate images in a batch (Diffusers supports list of prompts)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in prompts]
    images = sd.generate(prompts=prompts, seeds=seeds)

    # Save per-sample and one grid
    for i, img in enumerate(images):
        (out_dir / f"sample_{i:03d}.png").parent.mkdir(parents=True, exist_ok=True)
        img.save(out_dir / f"sample_{i:03d}.png")
    try:
        save_grid(images, str(out_dir / "grid.png"), nrow=4)
    except Exception:
        pass

    # Save prompts CSV
    import csv
    with open(out_dir / "prompts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "prompt"])
        for i, prompt in enumerate(prompts):
            writer.writerow([i, prompt])

    print(f"[ok] wrote {len(images)} images to {out_dir} (grid.png + per-sample PNGs)")
    print(f"[ok] prompts CSV -> {out_dir/'prompts.csv'}")

if __name__ == "__main__":
    main()
