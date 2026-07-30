#!/usr/bin/env bash
# Bounds-checked debug build of the extractor (diagnosis only).
# usage: lw_build_debug.sh WRF_SOURCE_ROOT BUILD_DIR INPUT_TXT OUT_DIR
set -euo pipefail
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
input_txt=$(realpath "$3")
out_dir=$(realpath -m "$4")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lw_source="${source_root}/phys/module_ra_rrtmg_lw.F"
mkdir -p "${build_dir}" "${out_dir}"
cd "${build_dir}"
FFLAGS="-O0 -g -fcheck=bounds -cpp -ffree-form -ffree-line-length-none \
  -fconvert=big-endian -fallow-argument-mismatch \
  -DEM_CORE=1 -DWRF_CHEM=0 -DHWRF=0 \
  -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
python3 "${script_dir}/lw_gen_dumpers.py" "${lw_source}" lw_dump_state.F90
gfortran -c $FFLAGS "${script_dir}/lw_stub_wrf.F90"
gfortran -c $FFLAGS -I . "${lw_source}"
gfortran -c $FFLAGS -I . "${script_dir}/lw_binio.F90"
gfortran -c $FFLAGS -I . lw_dump_state.F90
gfortran -c $FFLAGS -I . "${script_dir}/lw_extract.F90"
gfortran -g -o lw_extract_dbg lw_stub_wrf.o module_ra_rrtmg_lw.o lw_binio.o \
    lw_dump_state.o lw_extract.o
cp "${source_root}/run/RRTMG_LW_DATA" .
./lw_extract_dbg "${input_txt}" "${out_dir}"
