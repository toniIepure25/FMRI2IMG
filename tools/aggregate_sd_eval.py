#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="reports/sd_eval")
    ap.add_argument("--out_csv",  default="reports/sd_eval/summary_all.csv")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    parts = []
    for enc in ["mlp", "vit3d", "gnn"]:
        csvp = eval_dir / f"{enc}_clipscores.csv"
        if csvp.exists():
            df = pd.read_csv(csvp)
            df["encoder"] = enc
            parts.append(df)
    if not parts:
        raise FileNotFoundError("No *_clipscores.csv found in eval dir.")

    all_df = pd.concat(parts, ignore_index=True)
    all_df.to_csv(args.out_csv, index=False)

    # summary
    s = all_df.groupby("encoder")["clip_score"].agg(["mean", "std", "count"])
    print(s.to_string())

if __name__ == "__main__":
    main()
