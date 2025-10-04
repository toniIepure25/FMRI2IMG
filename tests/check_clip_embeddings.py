#!/usr/bin/env python
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_npy", type=str, required=True)
    ap.add_argument("--img_npy", type=str)
    args = ap.parse_args()

    t = np.load(args.text_npy)
    print(f"[text] shape={t.shape}, dtype={t.dtype}, norm(mean)={np.mean(np.linalg.norm(t, axis=-1)):.4f}")

    if args.img_npy:
        i = np.load(args.img_npy)
        print(f"[img ] shape={i.shape}, dtype={i.dtype}, norm(mean)={np.mean(np.linalg.norm(i, axis=-1)):.4f}")

        # cosine similarity sanity: random pair
        c = (t[:1] @ i[:1].T).squeeze()
        print(f"[cosine] first text vs first image = {float(c):.4f}")


if __name__ == "__main__":
    main()
