#!/usr/bin/env python3
import argparse
import re
import pandas as pd
from pathlib import Path

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "").replace("-", "_") for c in df.columns]
    return df

def _find_pairs(cols):
    """
    Return dict: suffix -> {"idx": colname or None, "score": colname or None}
    Accepts forms like: idx0, score0, top1_idx, top1_score, rank2_idx, rank2_score, etc.
    """
    pairs = {}
    for c in cols:
        # try to detect IDX with a numeric suffix somewhere
        m_idx = re.search(r'(?:^|_)(?:text)?idx(\d+)$', c)
        if not m_idx:
            m_idx = re.search(r'(?:^|_)(?:top|rank)?(\d+)_?(?:text)?idx$', c)
        if m_idx:
            suf = m_idx.group(1)
            pairs.setdefault(suf, {})["idx"] = c
            continue

        # detect SCORE with numeric suffix
        m_sc = re.search(r'(?:^|_)score(\d+)$', c)
        if not m_sc:
            m_sc = re.search(r'(?:^|_)(?:top|rank)?(\d+)_?score$', c)
        if m_sc:
            suf = m_sc.group(1)
            pairs.setdefault(suf, {})["score"] = c
            continue
    return pairs

def read_infer_csv(path: Path, encoder_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _norm_cols(df)
    df["encoder"] = encoder_name

    # Already long?
    if {"sample", "rank", "text_idx", "score"}.issubset(df.columns):
        return df[["sample", "rank", "text_idx", "score", "encoder"]].copy()

    # Try to coerce wide -> long by pairing *_idx / *_score columns
    if "sample" not in df.columns:
        # If no explicit 'sample', fabricate one by row index
        df = df.reset_index(drop=False).rename(columns={"index": "sample"})

    pairs = _find_pairs(df.columns)
    if not pairs:
        raise ValueError(f"Unrecognized schema in {path} (no idx/score pairs found)")

    rows = []
    for suf, d in pairs.items():
        idx_col = d.get("idx", None)
        sc_col  = d.get("score", None)
        if idx_col is None or sc_col is None:
            # skip incomplete pair
            continue
        tmp = df[["sample", idx_col, sc_col, "encoder"]].copy()
        tmp = tmp.rename(columns={idx_col: "text_idx", sc_col: "score"})
        # rank is 1-based if suffix starts at '1', else make it +1 from int
        try:
            rank_val = int(suf)
            if rank_val == 0:
                rank_val = 1
        except Exception:
            rank_val = None
        tmp["rank"] = rank_val
        rows.append(tmp)

    if not rows:
        raise ValueError(f"Unrecognized schema in {path} (found pairs but none usable)")

    out = pd.concat(rows, ignore_index=True)
    # If some ranks are None, sort by score desc per sample and assign ranks
    if out["rank"].isna().any():
        out = out.sort_values(["sample", "score"], ascending=[True, False])
        out["rank"] = out.groupby("sample").cumcount() + 1

    # ensure int types where possible
    out["rank"] = out["rank"].astype(int)
    out["text_idx"] = out["text_idx"].astype(int, errors="ignore")
    return out[["sample", "rank", "text_idx", "score", "encoder"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp_csv",   type=str, required=True)
    ap.add_argument("--vit3d_csv", type=str, required=True)
    ap.add_argument("--gnn_csv",   type=str, required=True)
    ap.add_argument("--out_csv",   type=str, required=True)
    args = ap.parse_args()

    mlp   = read_infer_csv(Path(args.mlp_csv),   "mlp")
    vit3d = read_infer_csv(Path(args.vit3d_csv), "vit3d")
    gnn   = read_infer_csv(Path(args.gnn_csv),   "gnn")

    merged = pd.concat([mlp, vit3d, gnn], ignore_index=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
