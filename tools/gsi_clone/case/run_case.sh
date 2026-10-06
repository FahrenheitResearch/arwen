#!/usr/bin/env bash
# One public analysis case end to end on a host with the "full" image:
# observation preprocessing, the cold-start guess, GSI with the ensemble off,
# then the engine's exchange round trip on GSI's output. Run from the host.
#
#   run_case.sh <image> <valid YYYYMMDDHH> <case root> <engine dir> [ranks]
#
# <case root> holds case/<valid>/ (fetch_case.sh), fix/conus and crtm/fix
# (prepare_crtm_fix.sh); results go to <case root>/run-<valid>. <engine dir>
# is an engine checkout whose Rust reader and writer are named by
# GPUWM_RW_NETCDF and GPUWM_NCWRITE_BRIDGE.
#
# GSI runs the tag's two steps the way its analysis script runs them when the
# global ensemble is present, without the ensemble: the variational step on
# the 4x coarser analysis grid with the cloud analysis skipped (type 5), then
# the cloud analysis alone on the full grid (type 6).
set -euo pipefail
image="$1"; valid="$2"; root="$3"; engine="$4"; ranks="${5:-16}"
here="$(cd "$(dirname "$0")" && pwd)"
run="$root/run-$valid"
mkdir -p "$run"
export CASE_ROOT="$root"

wait_for() {  # wait_for <marker>: block until the step writes its marker
  until [[ -f "$1" ]]; do sleep 20; done
  grep -q '^rc=0$' "$1" || { echo "step failed: $1"; cat "$1"; exit 1; }
}

bash "$here/in_container.sh" "$image" "$run/prep.log" "$run/prep.done" 60g -- \
  /scripts/run_prep_obs.sh "$valid" "/data/case/$valid" /data/fix/conus "/data/run-$valid/prep" 8
wait_for "$run/prep.done"

bash "$here/in_container.sh" "$image" "$run/makeguess.log" "$run/makeguess.done" 110g -- \
  /scripts/run_makeguess.sh "$valid" "/data/case/$valid/rap/rap.t${valid:8:2}z.awp130bgrbf00.grib2" \
  /data/fix/conus "/data/run-$valid/makeguess" 8
wait_for "$run/makeguess.done"

GRID_RATIO=4 CLOUD_ANALYSIS_TYPE=5 CLOUD_STEP=1 PASS_ENV="GRID_RATIO CLOUD_ANALYSIS_TYPE CLOUD_STEP" \
  bash "$here/in_container.sh" "$image" "$run/gsi.log" "$run/gsi.done" 100g -- \
  /scripts/run_analysis.sh "$valid" "/data/run-$valid/makeguess/wrfinput_d01" "/data/case/$valid/obs" \
  /data/fix/conus /data/crtm/fix "/data/run-$valid/gsi" "$ranks" 1
wait_for "$run/gsi.done"

mkdir -p "$run/exchange"
( cd "$engine" && CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 \
    python3 -m gpuwm.io.analysis_exchange inventory "$run/gsi/wrf_inout" \
    > "$run/exchange/inventory-analysis.json" ) || true
( cd "$engine" && CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 \
    python3 -m gpuwm.io.analysis_exchange round-trip "$run/gsi/wrf_inout" \
    "$run/exchange/wrf_inout.back" --receipt "$run/exchange/receipt-analysis.json" \
    > "$run/exchange/round-trip-analysis.out" 2> "$run/exchange/round-trip-analysis.err" )
echo "rc=$?" > "$run/exchange.done"
cat "$run/exchange/round-trip-analysis.out"
