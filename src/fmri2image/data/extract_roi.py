from pathlib import Path
import argparse
import numpy as np

def main(fmriprep_dir: str, out_dir: str, mode: str, atlas: str, vec_dim: int, subject: str = "subj01"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # TODO (real): load preprocessed BOLD + apply ROI parcellation (nilearn + atlas)
    # MOCK: generate a deterministic tensor so runs are reproducible
    rng = np.random.default_rng(1337)
    roi = rng.standard_normal((120, vec_dim), dtype=np.float32)  # 120 "samples" x vec_dim
    np.save(out / f"{subject}_roi.npy", roi)
    print(f"[ok] ROI tensor mock -> {out / f'{subject}_roi.npy'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmriprep_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", default="atlas")
    ap.add_argument("--atlas", default="glasser")
    ap.add_argument("--vector_dim", type=int, default=2048)
    ap.add_argument("--subject", type=str, default="subj01")
    args = ap.parse_args()
    main(args.fmriprep_dir, args.out_dir, args.mode, args.atlas, args.vector_dim, args.subject)
