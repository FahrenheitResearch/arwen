#!/usr/bin/env bash
set -euo pipefail
test $# = 2
root=$(realpath "$1")
out=$(realpath -m "$2")
here=$(cd "$(dirname "$0")" && pwd)
source_file=$root/phys/module_ra_aerosol.F
pin=57c7e7e50a2be0a6ab169fa376da077b2b4251bf9c20fa6a06446934379c8870
test "$(sha256sum "$source_file" | cut -d' ' -f1)" = "$pin" || { echo 'WRF aerosol source byte pin differs'; exit 3; }
mkdir -p "$out"
cd "$out"
gfortran -c -O0 -fno-tree-vectorize -ffp-contract=off -ffree-line-length-none "$here/stub_wrf.F90"
gfortran -c -O0 -fno-tree-vectorize -ffp-contract=off -ffree-form -ffree-line-length-none "$source_file"
gfortran -O0 -fno-tree-vectorize -ffp-contract=off -ffree-line-length-none "$here/run_aer_opt2.F90" module_ra_aerosol.o stub_wrf.o -o run_aer_opt2
./run_aer_opt2 columns.bin
sha256sum "$source_file" "$here/run_aer_opt2.F90" columns.bin > SHA256SUMS
gfortran --version | head -1 > compiler.txt
nm -u module_ra_aerosol.o > symbols.txt
