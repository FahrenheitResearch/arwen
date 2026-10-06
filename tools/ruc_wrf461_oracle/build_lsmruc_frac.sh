#!/usr/bin/env bash
# Build and run the LSMRUC oracle under the operational HRRR surface switches
# (xice_threshold = 0.02 from fractional_seaice = 1, rdlai2d = .true.).
# Self-contained: compiles the unmodified module itself, so it needs no
# build.sh tree.  Writes lsmruc_hrrr_switches.csv and its stack-fill control.
#
# usage: build_lsmruc_frac.sh WRF_SOURCE_ROOT BUILD_DIR [FC]
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build_lsmruc_frac.sh WRF_SOURCE_ROOT BUILD_DIR [FC]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
fc=${3:-gfortran}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ruc_source="${source_root}/phys/module_sf_ruclsm.F"

test -f "${ruc_source}"
mkdir -p "${build_dir}"
cd "${build_dir}"

"${fc}" -c -O0 -cpp -ffree-form -ffree-line-length-none \
    "${script_dir}/stub_wrf.F90"
"${fc}" -c -O0 -cpp -DEM_CORE=0 -Dwrf_chem=0 \
    -fallow-argument-mismatch -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${ruc_source}"
"${fc}" -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_lsmruc_frac.F90"
"${fc}" -o run_lsmruc_frac stub_wrf.o module_sf_ruclsm.o run_lsmruc_frac.o
cp "${source_root}/run/VEGPARM.TBL" .
cp "${source_root}/run/SOILPARM.TBL" .
cp "${source_root}/run/GENPARM.TBL" .
./run_lsmruc_frac lsmruc_hrrr_switches.csv
./run_lsmruc_frac lsmruc_hrrr_switches_stackfill.csv 12345.0
"${fc}" --version | head -1
sha256sum "${ruc_source}" \
    "${script_dir}/run_lsmruc_frac.F90" \
    "${script_dir}/build_lsmruc_frac.sh" \
    lsmruc_hrrr_switches.csv lsmruc_hrrr_switches_stackfill.csv
