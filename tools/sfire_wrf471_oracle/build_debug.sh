#!/usr/bin/env bash
set -eu
tools=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output=${1:?Specify an owned oracle build directory}
python=${PYTHON:-python3}
mkdir -p "$output"
"$python" "$tools/extract_debug.py" --source "$tools/reference/phys/module_fr_fire_util.F" --output "$output/debug_reference.F90"
gfortran -O0 -ffp-contract=off -fno-fast-math -ffree-line-length-none -J "$output" -I "$output" \
 "$output/debug_reference.F90" "$tools/run_debug.F90" -o "$output/run_debug"
cd "$output"
./run_debug
gfortran -O0 -ffp-contract=off -fno-fast-math -ffree-line-length-none -J "$output" -I "$output" \
 "$output/debug_reference.F90" "$tools/run_debug_defects.F90" -o "$output/run_debug_defects"
./run_debug_defects step-overflow
# Bound this deliberate reproduction of the WRF implied-DO overflow.
ulimit -c 0
set +e
timeout 5s ./run_debug_defects bound-overflow > bound-overflow.log 2>&1
overflow_status=$?
set -e
printf '%s\n' "$overflow_status" > bound-overflow.status
if [ "$overflow_status" -eq 0 ] || [ "$overflow_status" -eq 124 ]; then
 echo "The WRF bound-overflow control did not reproduce its immediate failure" >&2
 exit 1
fi
gfortran --version | head -n 1 > compiler.txt
