#!/usr/bin/env bash
# Build the SW oracle programs against the UNMODIFIED WRF v4.6.1 RRTMG
# modules.  Mirrors WRF's effective cpp state for this bundle (EM_CORE=1,
# WRF_CHEM=0, HWRF=0, RWORDSIZE=4 so kind_rb = kind(1.0) = FP32) and
# BYTESWAPIO -> -fconvert=big-endian so rrtmg_swlookuptable reads
# RRTMG_SW_DATA with WRF's byte order.  Pattern proven by the RUC oracle
# (tools/ruc_wrf461_oracle) and the RRTMG Phase-A scratch build.
#
# usage: sw_build.sh WRF_SOURCE_ROOT BUILD_DIR
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: sw_build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
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

gfortran -c $FFLAGS "${script_dir}/sw_stub_wrf.F90"
gfortran -c $FFLAGS -I "${build_dir}" "${lw_source}"
gfortran -c $FFLAGS -I "${build_dir}" "${sw_source}"

gfortran -c $FFLAGS -I "${build_dir}" "${script_dir}/sw_dump_tables.F90"
gfortran -o sw_dump_tables sw_stub_wrf.o module_ra_rrtmg_lw.o \
    module_ra_rrtmg_sw.o sw_dump_tables.o

# The fixture driver reproduces WRF's own sequence-association call forms
# (rank-2 actuals against explicit-shape rank-1 dummies in inirad/relcalc),
# so it needs -fallow-argument-mismatch exactly as WRF's build does.
gfortran -c $FFLAGS -I "${build_dir}" "${script_dir}/sw_fixture_driver.F90"
gfortran -o sw_fixture_driver sw_stub_wrf.o module_ra_rrtmg_lw.o \
    module_ra_rrtmg_sw.o sw_fixture_driver.o

cp -f "${source_root}/run/RRTMG_SW_DATA" .
cp -f "${source_root}/run/RRTMG_LW_DATA" .

sha256sum "${lw_source}" "${sw_source}" RRTMG_SW_DATA RRTMG_LW_DATA \
    "${script_dir}/sw_stub_wrf.F90" "${script_dir}/sw_dump_tables.F90" \
    "${script_dir}/sw_fixture_driver.F90" > sw-oracle-sha256sums.txt
gfortran --version | head -1 > sw-compiler.txt
echo "sw_build.sh: OK"
