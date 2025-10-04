import argparse
from pathlib import Path
from PIL import Image
from fmri2image.embeddings.clip_utils import load_openclip, encode_images, save_embeddings_npy_pkl


def collect_images(root: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    root = Path(root)
    paths = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    return paths


def main():
    ap = argparse.ArgumentParser(description="OpenCLIP Image Embedding Generator")
    ap.add_argument("--images_root", type=str, required=True, help="Root folder with images")
    ap.add_argument("--out_npy", type=str, required=True, help="Output .npy for embeddings")
    ap.add_argument("--out_pkl", type=str, required=True, help="Output .pkl metadata")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    paths = collect_images(args.images_root)
    if not paths:
        raise SystemExit(f"No images found under: {args.images_root}")

    model, _tok, preprocess, device = load_openclip(args.model, args.pretrained, args.device)

    imgs = [Image.open(p).convert("RGB") for p in paths]
    emb = encode_images(model, preprocess, imgs, device=device, batch_size=args.batch_size, normalize=True)

    # Save
    save_embeddings_npy_pkl(
        args.out_npy,
        args.out_pkl,
        emb,
        meta={
            "num_images": len(paths),
            "model": args.model,
            "pretrained": args.pretrained,
            "device": device,
            "paths": [str(p) for p in paths],
        },
    )
    print(f"[ok] wrote image embeddings -> {args.out_npy} ; meta -> {args.out_pkl} ; shape={emb.shape}")


if __name__ == "__main__":
    main()
