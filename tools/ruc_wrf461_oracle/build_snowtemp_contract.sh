#!/usr/bin/env bash
# Builds and runs only the snowtemp contract oracle.  Separate from build.sh
# so it adds no lines to the shared script and so this fixture can be
# regenerated without touching any already-pinned CSV.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build_snowtemp_contract.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ruc_source="${source_root}/phys/module_sf_ruclsm.F"

test -f "${ruc_source}"
mkdir -p "${build_dir}"
cd "${build_dir}"

gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none \
    "${script_dir}/stub_wrf.F90"
gfortran -c -O0 -cpp -DEM_CORE=0 -Dwrf_chem=0 \
    -fallow-argument-mismatch -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${ruc_source}"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_snowtemp_contract.F90"
gfortran -o run_snowtemp_contract \
    stub_wrf.o module_sf_ruclsm.o run_snowtemp_contract.o

./run_snowtemp_contract ruc-snowtemp-contract.csv
export PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}"
python3 "${script_dir}/validate_snowtemp_contract_oracle.py" \
    ruc-snowtemp-contract.csv
sha256sum "${ruc_source}" "${script_dir}/stub_wrf.F90" \
    "${script_dir}/run_snowtemp_contract.F90" \
    "${script_dir}/validate_snowtemp_contract_oracle.py" \
    ruc-snowtemp-contract.csv \
    >> oracle-sha256sums.txt
gfortran --version | head -1 > compiler-snowtemp-contract.txt
