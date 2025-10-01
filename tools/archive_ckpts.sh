#!/usr/bin/env bash
set -euo pipefail

DAY="$(date +%Y-%m-%d)"
SRC_ROOT="outputs/${DAY}"
DST_ROOT="data/artifacts/ckpts/${DAY}"

mkdir -p "${DST_ROOT}"

declare -A MAP=(
  [abl_mlp_ckpt]=mlp_last.ckpt
  [abl_vit3d_ckpt]=vit3d_last.ckpt
  [abl_gnn_ckpt]=gnn_last.ckpt
)

for RUN in "${!MAP[@]}"; do
  SRC="${SRC_ROOT}/${RUN}/checkpoints/last.ckpt"
  if [[ -f "${SRC}" ]]; then
    cp -f "${SRC}" "${DST_ROOT}/${MAP[$RUN]}"
    dvc add "${DST_ROOT}/${MAP[$RUN]}"
  else
    echo "[warn] missing ${SRC} – skipping"
  fi
done

echo "[ok] archived to ${DST_ROOT}"
