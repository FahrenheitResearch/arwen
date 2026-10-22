#!/usr/bin/env bash
# Run after the kit installs its pinned Python/CuPy/NVRTC environment.
# Inputs are local packs and their SHA256 sidecars, copied with the private kit.
set -euo pipefail
if [ "$#" -eq 0 ]; then
    echo 'usage: seed-box.sh PACK.tar.gz [PACK.tar.gz ...]' >&2
    exit 2
fi
for pack in "$@"; do
    "${GPUWM_PYTHON:-python}" -m gpuwm kernels seed "$pack"
done
