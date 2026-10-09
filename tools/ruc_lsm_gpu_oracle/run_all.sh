#!/usr/bin/env bash
# Every oracle configuration, strict build then default arithmetic:
#   run_all.sh RUNS_DIR
# Needs T, WRF_SRC and env.sh as run_case.sh does.  Writes RUNS_DIR/summary.txt.
set -uo pipefail
runs=$(realpath -m "$1")
tool="$T/tools/ruc_lsm_gpu_oracle"
mkdir -p "$runs"
configs=(
    "nzs9|--nzs 9"
    "nzs6|--nzs 6"
    "mosaic|--nzs 9 --mosaic 1"
    "noseaicefrac|--nzs 9 --fractional-seaice 0"
    "lake-rdlai2d|--nzs 9 --lakemodel 1 --rdlai2d 1"
    "dt60-k12|--nzs 9 --dt 60 --steps 12"
)
: > "$runs/summary.txt"
for mode in strict default; do
    for entry in "${configs[@]}"; do
        name=${entry%%|*}; opts=${entry#*|}
        bash "$tool/run_case.sh" "$runs/$name-$mode" "$mode" $opts \
            | sed "s|^|$name-$mode |" >> "$runs/summary.txt"
    done
done
