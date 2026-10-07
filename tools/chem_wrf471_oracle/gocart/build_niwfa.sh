#!/usr/bin/env bash
# The aerosol-aware Thompson coupling oracle: NOAA GSL's get_niwfa, compiled
# from the pinned ccpp-physics file by exact line range, run on fixed columns.
#
#   bash tools/chem_wrf471_oracle/gocart/build_niwfa.sh GSL_SRC BUILD_DIR FIXTURE_DIR
#
# GSL_SRC holds physics/MP/Thompson/mp_thompson.F90 of ufs-community/ccpp-physics
# at 3e6660c6df54e95a0871e990c2294dd397ae3860 (Apache-2.0, the licence ArWen
# carries).  mp_thompson.F90 USEs the whole Thompson module tree, so the one
# routine is lifted out by line range (1025-1068, checked against the file
# sha256 before extraction) into a harness module that supplies kind_phys.
# kind_phys = 8: the CCPP `machine` module's double-precision physics build,
# which is the RRFS/HRRR configuration the coupling comes from.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: build_niwfa.sh GSL_SRC BUILD_DIR FIXTURE_DIR" >&2
    exit 2
fi
src=$(realpath "$1")
build=$(realpath -m "$2")
fixtures=$(realpath -m "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
file="${src}/physics/MP/Thompson/mp_thompson.F90"
pinned=35b531fac3fa84bd26e5f0d9c436278954123f79f5c30a22944737a09db9d06d
got=$(sha256sum "${file}" | cut -d' ' -f1)
if [[ "${got}" != "${pinned}" ]]; then
    echo "mp_thompson.F90 sha256 ${got}, pinned ${pinned}" >&2
    exit 3
fi
mkdir -p "${build}" "${fixtures}"
cd "${build}"
first=$(sed -n '1025p' "${file}")
last=$(sed -n '1068p' "${file}")
[[ "${first}" == *"subroutine get_niwfa("* ]] || { echo "line 1025 is not get_niwfa" >&2; exit 3; }
[[ "${last}" == *"end subroutine get_niwfa"* ]] || { echo "line 1068 is not its end" >&2; exit 3; }
sed -n '1025,1068p' "${file}" > get_niwfa.inc
cat > niwfa_harness.F90 <<'F90'
module niwfa_harness
  implicit none
  integer, parameter :: kind_phys = 8
contains
#include "get_niwfa.inc"
end module niwfa_harness
F90
flags="-O0 -cpp -ffree-form -ffree-line-length-none -I ${build}"
gfortran -c ${flags} "${here}/oracle_io.F90"
gfortran -c ${flags} niwfa_harness.F90
gfortran -c ${flags} "${here}/niwfa_driver.F90"
gfortran -o niwfa_driver niwfa_driver.o niwfa_harness.o oracle_io.o
nm -u niwfa_harness.o | grep -q _ZGV && { echo "libmvec symbol in -O0 object" >&2; exit 4; }
( cd "${fixtures}" && "${build}/niwfa_driver" "${fixtures}" > "${build}/niwfa-stdout.txt" )
( cd "${fixtures}" && find . -name '*.bin' -o -name 'MANIFEST.txt' | sort | xargs sha256sum > oracle-sha256sums.txt )
gfortran --version | head -1 > "${fixtures}/compiler.txt"
echo "niwfa fixture in ${fixtures}"
