#!/usr/bin/env bash
# Standalone WRF v4.6.1 RRTMG oracle build (foundation lane).
#
# Compiles phys/module_ra_rrtmg_lw.F and phys/module_ra_rrtmg_sw.F exactly
# as shipped against the stub, mirroring WRF's effective cpp state for this
# bundle's group build:
#   EM_CORE=1, WRF_CHEM=0, HWRF=0, RWORDSIZE=4  (kind_rb = kind(1.0) = FP32)
# BYTESWAPIO -> -fconvert=big-endian, so rrtmg_lwlookuptable /
# rrtmg_swlookuptable read the non-DBL RRTMG_*_DATA files with WRF's byte
# order, and every oracle dump stream is likewise big-endian.
#
# Foundation drivers built here:
#   coeffs_dump_lw / coeffs_dump_sw   post-init kg-module coefficient dumps
#   mcica_fixture_lw / mcica_fixture_sw   McICA subcolumn generator fixtures
# The LW/SW compute lanes add their own drivers beside these (lw_*.F90 /
# sw_*.F90) using the same stub and flags.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lw_source="${source_root}/phys/module_ra_rrtmg_lw.F"
sw_source="${source_root}/phys/module_ra_rrtmg_sw.F"

test -f "${lw_source}"
test -f "${sw_source}"
mkdir -p "${build_dir}"
cd "${build_dir}"

FFLAGS="-O0 -cpp -ffree-form -ffree-line-length-none -fconvert=big-endian \
  -fallow-argument-mismatch -DEM_CORE=1 -DWRF_CHEM=0 -DHWRF=0 \
  -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"

gfortran -c ${FFLAGS} "${script_dir}/stub_wrf.F90"
gfortran -c ${FFLAGS} -I "${build_dir}" "${lw_source}"
gfortran -c ${FFLAGS} -I "${build_dir}" "${sw_source}"
gfortran -c ${FFLAGS} -I "${build_dir}" "${script_dir}/dump_kit.F90"

for driver in coeffs_dump_lw coeffs_dump_sw mcica_fixture_lw \
              mcica_fixture_sw; do
    gfortran -c -O0 -ffree-form -ffree-line-length-none \
        -fconvert=big-endian -I "${build_dir}" "${script_dir}/${driver}.F90"
    gfortran -o "${driver}" stub_wrf.o module_ra_rrtmg_lw.o \
        module_ra_rrtmg_sw.o dump_kit.o "${driver}.o"
done

cp "${source_root}/run/RRTMG_LW_DATA" .
cp "${source_root}/run/RRTMG_SW_DATA" .

./coeffs_dump_lw rrtmg-coeffs-lw.dump
./coeffs_dump_sw rrtmg-coeffs-sw.dump

sha256sum "${lw_source}" "${sw_source}" \
    "${script_dir}/stub_wrf.F90" "${script_dir}/dump_kit.F90" \
    "${script_dir}/coeffs_dump_lw.F90" "${script_dir}/coeffs_dump_sw.F90" \
    "${script_dir}/mcica_fixture_lw.F90" "${script_dir}/mcica_fixture_sw.F90" \
    RRTMG_LW_DATA RRTMG_SW_DATA \
    rrtmg-coeffs-lw.dump rrtmg-coeffs-sw.dump \
    > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "rrtmg_wrf461_oracle: build + coefficient dumps complete"
