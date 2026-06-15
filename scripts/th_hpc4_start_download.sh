#!/bin/sh
set -eu

# Start the TH-HPC4 weight download in the background from a login node.
# This is a convenience wrapper around th_hpc4_download_weights.sh.

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

: "${NWP_WEIGHTS_ROOT:=$ROOT_DIR/assets/weights}"
: "${HF_ENDPOINT:=https://hf-mirror.com}"
export NWP_WEIGHTS_ROOT
export HF_ENDPOINT

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
mkdir -p "$LOG_DIR" "$NWP_WEIGHTS_ROOT"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/download_weights.$STAMP.log"
PID_FILE="$LOG_DIR/download_weights.$STAMP.pid"

nohup sh scripts/th_hpc4_download_weights.sh "$@" > "$LOG_FILE" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"

echo "Started NWP weight download."
echo "  pid:         $PID"
echo "  log:         $LOG_FILE"
echo "  pid file:    $PID_FILE"
echo "  weights:     $NWP_WEIGHTS_ROOT"
echo "  HF endpoint: $HF_ENDPOINT"
echo
echo "Monitor:"
echo "  tail -f '$LOG_FILE'"
echo
echo "Verify after it exits:"
echo "  python3 scripts/download_all_weights.py --weights-root '$NWP_WEIGHTS_ROOT' --verify-only"
