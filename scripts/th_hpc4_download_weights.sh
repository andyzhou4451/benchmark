#!/bin/sh
set -eu

# TH-HPC4 login nodes allow data transfer. Run this from the repository root.
# Do not submit this as a pure download job to GPU partitions unless the center
# administrator explicitly asks you to do so.

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

: "${NWP_WEIGHTS_ROOT:=$ROOT_DIR/assets/weights}"
: "${HF_ENDPOINT:=https://hf-mirror.com}"
export NWP_WEIGHTS_ROOT
export HF_ENDPOINT

mkdir -p "$NWP_WEIGHTS_ROOT"

python3 scripts/download_all_weights.py \
  --weights-root "$NWP_WEIGHTS_ROOT" \
  --continue-on-error \
  "$@"
