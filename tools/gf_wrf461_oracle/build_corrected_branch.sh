#!/usr/bin/env bash
# Reuse private pristine objects, replacing only the corrected shallow unit.
set -euo pipefail
branch=$(realpath "$1")
regime=$(realpath "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$branch/corrected"
cp "$branch"/*.csv "$branch/corrected/"
cp "$regime/corrected/corrected.F" "$branch/corrected/"
cd "$branch/corrected"
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none -fcheck=bounds \
    -I "$branch" -J . -o sh.o corrected.F
gfortran -o run_cu_gf "$branch/module_gfs_machine_O0.o" \
    "$branch/module_gfs_physcons_O0.o" "$branch/module_cu_gf_deep_O0.o" sh.o \
    "$branch/module_cu_gf_wrfdrv_O0.o" "$branch/gf_cases.o" "$branch/run_cu_gf.o"
./run_cu_gf gf-levels.csv gf-surface.csv gf-isolation.csv
python "$script_dir/reference_receipt.py" "$branch" "$branch/corrected"
