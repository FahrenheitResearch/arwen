#!/usr/bin/env bash
# Verify the deliverable against the actual slim export on a CPU box.
set -euo pipefail
lane=$(realpath "$1")
target="$2"
T="$lane/src"
source /work/env.sh
mkdir -p "$lane/tmp"
export TMPDIR="$lane/tmp" PYTHONDONTWRITEBYTECODE=1
cd "$T"
predecessor=91de806e22c44d91418c82a128319a026e213f9a
git diff --binary "$predecessor" "$target" -- gpuwm/core/sfire_coupler.py \
    tools/sfire_wrf471_oracle tests/test_sfire_regime_receipt.py \
    tests/test_sfire_subnormal_runtime.py > "$lane/lane.patch"
git apply --check --index "$lane/lane.patch"
git apply --index "$lane/lane.patch"
mapfile -t changed < <(git diff --name-only "$predecessor" "$target")
git diff --quiet --cached "$target" -- "${changed[@]}"
echo 'Changed source blobs match the lane tip exactly after applying to the slim export.'
CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 python -m pytest -q \
    tests/test_sfire_regime_receipt.py -m 'not gpu'
find -P "$lane" -print > "$lane/MANIFEST.txt"
