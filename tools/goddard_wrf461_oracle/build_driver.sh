#!/usr/bin/env bash
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
GODDARD_DEFINED_TAUX=1 bash "$D/build.sh" "$1" "$2"
cd "$2"
gfortran -O2 -fno-tree-vectorize -ffp-contract=off -fconvert=big-endian -frecord-marker=4 -ffree-form -ffree-line-length-none -fcheck=bounds -c "$D/run_driver.F90"
gfortran -o run_driver stub_wrf.o module_ra_goddard.o run_driver.o $(nf-config --flibs)
./run_driver driver.bin
sha256sum run_driver driver.bin.driver driver.bin.driver-inputs > DRIVER-SHA256SUMS
