#!/usr/bin/env bash
set -euo pipefail
root=$(realpath "$1"); out=$(realpath -m "$2"); here=$(cd "$(dirname "$0")" && pwd)
printf '%s  %s\n' a4a41a1593ef1d9bb4bc9cf4863fefea6dc67c3815bafbb897cb2a948e76a39f "$root/phys/module_sf_oml.F" | sha256sum -c -
printf '%s  %s\n' 5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062 "$root/share/module_model_constants.F" | sha256sum -c -
mkdir -p "$out"; cd "$out"
gfortran -O0 -fno-tree-vectorize -ffp-contract=off -cpp -ffree-form -ffree-line-length-none -c "$root/share/module_model_constants.F"
gfortran -O0 -fno-tree-vectorize -ffp-contract=off -cpp -ffree-form -ffree-line-length-none -c "$root/phys/module_sf_oml.F"
gfortran -O0 -fno-tree-vectorize -ffp-contract=off -ffree-line-length-none -I . "$here/run_oml.F90" module_sf_oml.o module_model_constants.o -o run_oml
./run_oml "$out"
gfortran --version | head -1 > compiler.txt
sha256sum "$root/phys/module_sf_oml.F" run_oml > hashes.txt
