#!/usr/bin/env bash
# Launch on an authorized node with the OWNER append protocol.
set -euo pipefail
engine=$(realpath "${1:?engine checkout}")
reference=$(realpath "${2:?compiled WRF reference run}")
output=$(realpath -m "${3:?new ArWen output directory}")
seconds=${4:-3600}
python=${5:?Python interpreter}
arithmetic=${6:-default}
owner=${SFIRE_GPU_OWNER_FILE:?authorized node OWNER file}
bound=${SFIRE_GPU_BOUND_MINUTES:-12}
wall=$((bound * 60 - 15))
mkdir -p "$output"
lane=sol-sfire/ideal-phase2
{
  flock -x 9
  printf '%s %s start pid %s bounded %s min (native em_fire %ss %s arithmetic; shared)\n' \
    "$lane" "$(date -u +%FT%TZ)" "$$" "$bound" "$seconds" "$arithmetic" >> "$owner"
} 9>"${owner}.lock"
finish() {
  local rc=$?
  {
    flock -x 9
    printf '%s %s release rc %s\n' "$lane" "$(date -u +%FT%TZ)" "$rc" >> "$owner"
  } 9>"${owner}.lock"
  printf '%s\n' "$rc" > "$output/launcher.done"
}
trap finish EXIT
export CUDA_VISIBLE_DEVICES=0 GPUWM_NO_LOCAL_GPU=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
export PYTHONPATH="$engine"
export CUPY_CACHE_DIR="$output/../cupy-cache-$arithmetic"
export CUDA_CACHE_PATH="$output/../cuda-cache-$arithmetic"
cd "$engine"
timeout "$wall" nice -n 15 "$python" tools/sfire_coupled_ideal/run_arwen.py \
  --reference "$reference" --out "$output" --run-seconds "$seconds" --arithmetic "$arithmetic"
