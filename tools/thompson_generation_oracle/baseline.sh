#!/usr/bin/env bash
set -eu
L=$1
T=$L/src
source /work/env.sh
export GPUWM_RW_NETCDF=/work/engine-base/tools/rustwx/target/release/rw_netcdf
export TMPDIR=$L/tmp
cd "$T"
for mode in default strict; do
  if [ "$mode" = strict ]; then export GPUWM_WRF_EXACT=1; else unset GPUWM_WRF_EXACT; fi
  for columns in "$L"/columns/*.npz; do
    python tools/thompson_generation_oracle/benchmark_columns.py compare "$columns" "$L/receipts/baseline-$mode" --lane "$L"
  done
done
