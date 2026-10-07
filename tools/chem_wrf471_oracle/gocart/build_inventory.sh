#!/usr/bin/env bash
# The GOCART anthropogenic emission add (emiss_opt=6), WRF v4.7.1
# chem/emissions_driver.F:1597-1619, lifted by exact line range into a
# harness subroutine (emissions_driver.F USEs every emission package and does
# not compile standalone) and driven on fixed columns.
#
#   bash tools/chem_wrf471_oracle/gocart/build_inventory.sh WRF_SRC BUILD_DIR FIXTURE_DIR
#
# The lines are checked against the file's pinned sha256 and their first and
# last statements before extraction.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: build_inventory.sh WRF_SRC BUILD_DIR FIXTURE_DIR" >&2
    exit 2
fi
src=$(realpath "$1")
build=$(realpath -m "$2")
fixtures=$(realpath -m "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
file="${src}/chem/emissions_driver.F"
pinned=$(awk '$2 == "chem/emissions_driver.F" {print $1}' "${here}/SOURCES-extract.sha256")
got=$(sha256sum "${file}" | cut -d' ' -f1)
if [[ "${got}" != "${pinned}" ]]; then
    echo "emissions_driver.F sha256 ${got}, pinned ${pinned}" >&2
    exit 3
fi
first=$(sed -n '1597p' "${file}")
last=$(sed -n '1619p' "${file}")
[[ "${first}" == *"if(config_flags%emiss_opt == 6"* ]] || { echo "line 1597 moved" >&2; exit 3; }
[[ "${last}" == *"endif"* ]] || { echo "line 1619 moved" >&2; exit 3; }
mkdir -p "${build}" "${fixtures}"
cd "${build}"
sed -n '1597,1619p' "${file}" > emiss_opt6.inc
defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
flags="-O0 -cpp ${defines} -ffree-form -ffree-line-length-none -I ${build}"
gfortran -c ${flags} "${here}/stub_gocart.F90"
gfortran -c ${flags} "${here}/oracle_io.F90"
gfortran -c ${flags} "${here}/inventory_harness.F90"
gfortran -c ${flags} "${here}/inventory_driver.F90"
gfortran -o inventory_driver inventory_driver.o inventory_harness.o oracle_io.o stub_gocart.o
nm -u inventory_harness.o | grep -q _ZGV && { echo "libmvec symbol in -O0 object" >&2; exit 4; }
( cd "${fixtures}" && "${build}/inventory_driver" "${fixtures}" > "${build}/inventory-stdout.txt" )
( cd "${fixtures}" && find . -name '*.bin' -o -name 'MANIFEST.txt' | sort | xargs sha256sum > oracle-sha256sums.txt )
gfortran --version | head -1 > "${fixtures}/compiler.txt"
echo "inventory fixture in ${fixtures}"
