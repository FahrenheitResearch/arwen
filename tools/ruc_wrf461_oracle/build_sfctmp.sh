#!/usr/bin/env bash
# Build and run the end-to-end sfctmp oracle on top of an existing build.sh
# tree.  Kept separate from build.sh, like the other late RUC lanes, so the
# hashes build.sh pinned stay byte-identical; this script appends its own with
# >>.
#
# sfctmp's snow-preparation prologue needs gdb (see build.sh), but the DISPATCH
# does not: it is the tail of the subroutine, so the returned arguments are its
# outputs.  This build therefore needs nothing beyond gfortran.
#
# The driver is run TWICE.  The first run zeroes the stack region that WRF's
# uninitialised `ilnb` (module_sf_ruclsm.F:1385) occupies, which is what makes
# the fixture reproducible at all; the second fills it with a nonzero pattern
# and is the negative control that proves the uninitialised read is real.  The
# validator requires the pair to differ in exactly five tsnav cells and
# nowhere else.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build_sfctmp.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
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
    -I "${build_dir}" "${script_dir}/run_sfctmp.F90"
gfortran -o run_sfctmp stub_wrf.o module_sf_ruclsm.o run_sfctmp.o
./run_sfctmp ruc-sfctmp.csv
./run_sfctmp ruc-sfctmp-stackfill.csv 12345.0
export PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}"
python3 "${script_dir}/validate_sfctmp_oracle.py" ruc-sfctmp.csv \
    --stack-control ruc-sfctmp-stackfill.csv
sha256sum "${ruc_source}" \
    "${script_dir}/run_sfctmp.F90" \
    "${script_dir}/validate_sfctmp_oracle.py" \
    ruc-sfctmp.csv ruc-sfctmp-stackfill.csv \
    >> oracle-sha256sums.txt
