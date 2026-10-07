#!/usr/bin/env bash
# Reference runs remain distinct; completed output is never overwritten.
set -euo pipefail
source_root=$(realpath "${1:?WRF source directory}")
build_root=$(realpath "${2:?WRF build directory}")
run_root=$(realpath -m "${3:?new run directory}")
wall_seconds=${4:-21600}
if [[ -e "$run_root/wrfinput_d01" || -e "$run_root/wrf.log" ]]; then
  echo 'The reference directory already contains a run; use a new directory.' >&2
  exit 2
fi
mkdir -p "$run_root"
cd "$run_root"
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for input in namelist.input namelist.fire input_sounding; do cp "$source_root/test/em_fire/$input" .; done
sha256sum "$build_root/main/ideal" "$build_root/main/wrf" namelist.input namelist.fire input_sounding > INPUTS.sha256
date -u +%FT%TZ > started.txt
set +e
nice -n 15 "$build_root/main/ideal" > ideal.log 2>&1
ideal_status=$?
printf '%s\n' "$ideal_status" > ideal.exit
if [[ "$ideal_status" == 0 ]]; then
  sha256sum wrfinput_d01 > INITIAL.sha256
  timeout "$wall_seconds" nice -n 15 "$build_root/main/wrf" > wrf.log 2>&1
  wrf_status=$?
  printf '%s\n' "$wrf_status" > wrf.exit
else
  wrf_status=$ideal_status
fi
set -e
find . -maxdepth 1 -name 'wrfout*' -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
date -u +%FT%TZ > finished.txt
printf '%s\n' "$wrf_status" > run.done
exit "$wrf_status"
