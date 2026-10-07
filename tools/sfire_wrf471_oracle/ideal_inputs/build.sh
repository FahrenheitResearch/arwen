#!/usr/bin/env bash
set -eu
tools=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
out=${1:?Specify an owned build directory}
initializer=${2:?Specify the pinned native module_initialize_fire.F}
python=${PYTHON:-python3}
mkdir -p "$out"
"$python" "$tools/extract.py" --util "$tools/../reference/phys/module_fr_fire_util.F" --initializer "$initializer" --output "$out"
gfortran -O0 -ffp-contract=off -fno-fast-math -fcheck=bounds -ffree-line-length-none -J "$out" -I "$out" "$out/input_reference.F90" "$tools/run.F90" -o "$out/run"
gfortran --version | head -n 1 > "$out/compiler.txt"
