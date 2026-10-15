#!/usr/bin/env bash
set -euo pipefail
T=$(cd "$(dirname "$0")/../.." && pwd)
L=$(dirname "$T")
source /work/env.sh
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
cd "$T"
python -m pytest -q tests/test_kernel_source_freeze_per_module.py \
    -k 'swint or rrtmg_legacy_solar or every_kernel_source_on_disk_is_pinned_by_something' \
    > "$L/solar-source-freeze.log" 2>&1
cat "$L/solar-source-freeze.log"
