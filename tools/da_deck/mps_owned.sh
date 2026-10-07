#!/usr/bin/env bash
# Box only. One owned MPS controller, UUID clients, no change to GPU mode.
set -Eeuo pipefail
MODE=${1:?start or stop}
MPS_ROOT=${2:?owned absolute MPS root beneath the deck workspace}
case "$MPS_ROOT" in /workspace/da-rerun-286/mps-*|/workspace/da-recent-286/mps-*) ;; *) echo 'MPS root outside deck-owned prefix' >&2; exit 2 ;; esac
if [[ "$MODE" == start ]]; then
    test ! -e "$MPS_ROOT"
    mkdir -p "$MPS_ROOT/pipe" "$MPS_ROOT/log"
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log"
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)
    [[ "$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)" -ge 1 ]]
    nvidia-cuda-mps-control -d
    printf 'export CUDA_MPS_PIPE_DIRECTORY=%q\nexport CUDA_MPS_LOG_DIRECTORY=%q\n' "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" > "$MPS_ROOT/client-env.sh"
    printf 'source %q\n' "$MPS_ROOT/client-env.sh"
elif [[ "$MODE" == stop ]]; then
    test -f "$MPS_ROOT/client-env.sh"
    source "$MPS_ROOT/client-env.sh"
    printf 'quit\n' | nvidia-cuda-mps-control
else
    echo 'Use start or stop' >&2
    exit 2
fi
