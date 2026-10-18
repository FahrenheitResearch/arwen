#!/usr/bin/env bash
# Dump WRF v4.6.1 MORR_TWO_MOMENT_INIT's constants (both morr_rimed_ice
# modes) and radar_init's words as hex, from the same -O0 build flags as
# the Morrison oracle (tools/morrison_wrf461_oracle/build.sh).
# usage: build.sh WRF_SOURCE_ROOT BUILD_DIR   (writes BUILD_DIR/constants.txt)
set -euo pipefail
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
oracle=$(CDPATH= cd -- "$here/../morrison_wrf461_oracle" && pwd)
src=$(realpath "$1"); build=$(realpath -m "$2")
echo "2dcd1c70127ea57c8a2437baf2b8a2388b1722e4d4ead9411834946439b8c273  $src/phys/module_mp_morr_two_moment.F" | sha256sum -c -
echo "aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688  $src/phys/module_mp_radar.F" | sha256sum -c -
mkdir -p "$build"; cd "$build"
python3 "$here/dump_constants.py" "$src/phys/module_mp_morr_two_moment.F" \
    module_mp_morr_two_moment_dump.F
f="-O0 -cpp -ffree-form -ffree-line-length-none"
gfortran -c $f -DRWORDSIZE=4 "$src/phys/ccpp_kind_types.F"
gfortran -c $f "$oracle/stub_wrf_error.F90"
gfortran -c $f "$src/share/module_model_constants.F"
gfortran -c $f -I "$build" "$src/phys/module_mp_radar.F"
gfortran -c $f -DWRF_CHEM=0 -fallow-argument-mismatch -I "$build" \
    module_mp_morr_two_moment_dump.F -o module_mp_morr_two_moment.o
gfortran -c $f -I "$build" "$here/run_dump.F90"
gfortran -o run_dump stub_wrf_error.o ccpp_kind_types.o \
    module_model_constants.o module_mp_radar.o module_mp_morr_two_moment.o \
    run_dump.o
./run_dump | grep -E '^ ?[RDI] ' > constants.txt
wc -l constants.txt
