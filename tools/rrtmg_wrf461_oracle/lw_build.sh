#!/usr/bin/env bash
# Build + run the RRTMG LW per-routine fixture extractor against the
# UNMODIFIED WRF v4.6.1 module.  Mirrors WRF's effective cpp state for the
# campaign bundle: EM_CORE=1, WRF_CHEM=0, HWRF=0, RWORDSIZE=4 (kind_rb=FP32),
# BYTESWAPIO -> -fconvert=big-endian so rrtmg_lwlookuptable reads
# RRTMG_LW_DATA with WRF's byte order (fixture dumps pin little_endian
# explicitly in lw_binio).
#
# usage: lw_build.sh WRF_SOURCE_ROOT BUILD_DIR [INPUT_TXT OUT_DIR]
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: lw_build.sh WRF_SOURCE_ROOT BUILD_DIR [INPUT_TXT OUT_DIR]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lw_source="${source_root}/phys/module_ra_rrtmg_lw.F"

test -f "${lw_source}"
test -f "${source_root}/run/RRTMG_LW_DATA"
mkdir -p "${build_dir}"
cd "${build_dir}"

FFLAGS="-O0 -cpp -ffree-form -ffree-line-length-none -fconvert=big-endian \
  -fallow-argument-mismatch -DEM_CORE=1 -DWRF_CHEM=0 -DHWRF=0 \
  -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"

python3 "${script_dir}/lw_gen_dumpers.py" "${lw_source}" lw_dump_state.F90

gfortran -c $FFLAGS "${script_dir}/lw_stub_wrf.F90"
gfortran -c $FFLAGS -I "${build_dir}" "${lw_source}"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    "${script_dir}/lw_binio.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none -I "${build_dir}" \
    lw_dump_state.F90
# NOTE: the MAIN program must carry -fconvert=big-endian too -- gfortran sets
# the process-wide default conversion from the main program's compile flags,
# not from the compilation unit containing the OPEN.  lw_binio overrides it
# per-unit with convert='little_endian' for the fixture files.
gfortran -c $FFLAGS -I "${build_dir}" "${script_dir}/lw_extract.F90"
gfortran -o lw_extract lw_stub_wrf.o module_ra_rrtmg_lw.o lw_binio.o \
    lw_dump_state.o lw_extract.o

# Cheap uninitialized-variable lead sweep on the authority source (charter:
# never be bit-exact to a bug).  Non-fatal; log kept beside the fixtures.
gfortran -c $FFLAGS -O2 -Wall -Wno-unused-variable -Wno-unused-dummy-argument \
    -Wno-unused-label -o /dev/null "${lw_source}" \
    > lw_warnings_O2.log 2>&1 || true

cp "${source_root}/run/RRTMG_LW_DATA" .

if [[ $# -ge 4 ]]; then
    input_txt=$(realpath "$3")
    out_dir=$(realpath -m "$4")
    mkdir -p "${out_dir}"
    ./lw_extract "${input_txt}" "${out_dir}"
    sha256sum "${lw_source}" "${source_root}/run/RRTMG_LW_DATA" \
        "${script_dir}/lw_stub_wrf.F90" "${script_dir}/lw_binio.F90" \
        "${script_dir}/lw_extract.F90" "${script_dir}/lw_gen_dumpers.py" \
        lw_dump_state.F90 "${input_txt}" \
        > "${out_dir}/lw_oracle_sha256sums.txt"
    ( cd "${out_dir}" && sha256sum lw_coeffs.bin lw_case_*.bin \
        >> lw_oracle_sha256sums.txt )
    gfortran --version | head -1 > "${out_dir}/lw_compiler.txt"
fi
echo "lw_build: OK"
