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
export PYTHONUNBUFFERED=1

mkdir -p "$NWP_WEIGHTS_ROOT"

echo "NWP weight download"
echo "  repo:        $ROOT_DIR"
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "  git commit: $(git rev-parse --short HEAD)"
fi
echo "  python:      $(command -v python3)"
echo "  python ver:  $(python3 -V 2>&1)"
echo "  weights:     $NWP_WEIGHTS_ROOT"
echo "  HF endpoint: $HF_ENDPOINT"
echo

python3 -u scripts/download_all_weights.py \
  --weights-root "$NWP_WEIGHTS_ROOT" \
  --continue-on-error \
  "$@"
