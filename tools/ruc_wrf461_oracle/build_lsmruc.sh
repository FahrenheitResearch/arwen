#!/usr/bin/env bash
# Build and run the LSMRUC driver oracle on top of an existing build.sh tree.
# Kept separate from build.sh, like the other late RUC lanes, so the hashes
# build.sh pinned stay byte-identical; this script appends its own with >>.
#
# LSMRUC is a subroutine whose results are all returned arguments, so unlike
# the sfctmp snow-preparation prologue this needs no gdb -- only gfortran.
#
# The driver is run TWICE.  The first run zeroes the stack region LSMRUC's
# callee frames occupy, which is what makes WRF's uninitialised `ilnb`
# (module_sf_ruclsm.F:1385) reproducible on the first column of each call;
# the second fills it with a nonzero pattern and is the negative control that
# proves the uninitialised read is real.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build_lsmruc.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    echo "  BUILD_DIR must already contain a completed build.sh tree" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ruc_source="${source_root}/phys/module_sf_ruclsm.F"

test -f "${ruc_source}"
test -f "${build_dir}/module_sf_ruclsm.o"
cd "${build_dir}"

gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_lsmruc.F90"
gfortran -o run_lsmruc stub_wrf.o module_sf_ruclsm.o run_lsmruc.o
./run_lsmruc ruc-lsmruc.csv
./run_lsmruc ruc-lsmruc-stackfill.csv 12345.0
export PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}"
python3 "${script_dir}/validate_lsmruc_oracle.py" ruc-lsmruc.csv \
    --stack-control ruc-lsmruc-stackfill.csv
sha256sum "${ruc_source}" \
    "${script_dir}/run_lsmruc.F90" \
    "${script_dir}/validate_lsmruc_oracle.py" \
    "${script_dir}/build_lsmruc.sh" \
    ruc-lsmruc.csv ruc-lsmruc-stackfill.csv \
    >> oracle-sha256sums.txt
