#!/usr/bin/env bash
# Build and run the snowsoil contract oracle on top of an existing build.sh
# tree.  Kept separate from build.sh so the four parallel RUC snow branches
# stay a pure append; the hashes are appended to the same sums file with >> so
# the ordering build.sh pinned stays byte-identical.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build_snowsoil_contract.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
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
    -I "${build_dir}" "${script_dir}/run_snowsoil_contract.F90"
gfortran -o run_snowsoil_contract \
    stub_wrf.o module_sf_ruclsm.o run_snowsoil_contract.o
./run_snowsoil_contract ruc-snowsoil-contract.csv
export PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}"
python3 "${script_dir}/validate_snowsoil_oracle.py" ruc-snowsoil.csv
python3 "${script_dir}/validate_snowsoil_oracle.py" \
    --contract ruc-snowsoil-contract.csv
sha256sum "${script_dir}/run_snowsoil_contract.F90" \
    "${script_dir}/validate_snowsoil_oracle.py" \
    ruc-snowsoil-contract.csv \
    >> oracle-sha256sums.txt
