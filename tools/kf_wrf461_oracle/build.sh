#!/usr/bin/env bash
# Build the Kain-Fritsch (cu_physics=1) column oracle against the
# byte-unmodified WRF v4.6.1 phys/module_cu_kfeta.F and run it on a column
# file written by kf_columns.py.
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR COLUMNS.bin [COLUMNS.bin ...]
#
# Three builds of the same unmodified source, one driver:
#
#   O0     the reference: gfortran -O0, no vectoriser, scalar libm.  This is
#          the 0 ULP oracle WOOF is graded against.
#   stock  WRF's own GNU flags for phys/ (arch/configure.defaults, the line
#          the W1 WRF build used): -O2 -ftree-vectorize -funroll-loops.  Run
#          to show whether the reference and WRF-as-built differ on these
#          columns at all.
#   snan   -O0 with every local REAL initialised to a signalling NaN and
#          every local INTEGER to -2147483647 at each entry.  A column whose
#          outputs differ between O0 and snan read memory the Fortran never
#          wrote: in a real run that is the previous column's stack.  The
#          oracle refuses to call such a column a match on either side.
#
# The pin is the file: the source must hash to the v4.6.1 value recorded in
# gpuwm/data/kf_lutab/PROVENANCE.md (LF line endings).
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR COLUMNS.bin [...]" >&2
    exit 2
fi
source_root=$(realpath "$1"); shift
build_dir=$(realpath -m "$1"); shift
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
kf_source="${source_root}/phys/module_cu_kfeta.F"
pinned_sha256="e6376c2d85c45470f49d545b25d513b5ec111bf36b87beebc740bf42825c6e5f"

actual=$(sha256sum "${kf_source}" | awk '{print $1}')
if [[ "${actual}" != "${pinned_sha256}" ]]; then
    echo "module_cu_kfeta.F is ${actual}, not the pinned ${pinned_sha256}" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"
# WRF's -fconvert=big-endian/-frecord-marker=4 only touch unformatted I/O
# and are left out: the driver's stream files are little endian on purpose.
common="-ffree-form -ffree-line-length-none"

build_one () {   # name, flags
    local name=$1; shift
    mkdir -p "${name}"
    ( cd "${name}"
      gfortran -c "$@" ${common} -o stub_wrf_error.o "${script_dir}/stub_wrf_error.F90"
      gfortran -c "$@" -cpp ${common} -o module_cu_kfeta.o "${kf_source}"
      gfortran -c -O0 ${common} -o run_kf_oracle.o "${script_dir}/run_kf_oracle.F90"
      gfortran -o run_kf_oracle run_kf_oracle.o module_cu_kfeta.o stub_wrf_error.o
      nm -u module_cu_kfeta.o | sed 's/^ *//' | sort > undefined.txt )
}

build_one O0 -O0
build_one stock -O2 -ftree-vectorize -funroll-loops
build_one snan -O0 -finit-real=snan -finit-integer=-2147483647 -finit-logical=false

if grep -q '_ZGV' O0/undefined.txt; then
    echo "the -O0 reference pulled in a libmvec symbol" >&2
    exit 4
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    for b in O0 stock snan; do
        echo "# ${b} libm/libmvec symbols:"
        grep -E 'expf|powf|logf|sqrtf|_ZGV' "${b}/undefined.txt" || true
    done
} > libm-report.txt
cat libm-report.txt

for cols in "$@"; do
    stem=$(basename "${cols}" .bin)
    for b in O0 stock snan; do
        ( cd "${b}" && rm -f fort.98 && ./run_kf_oracle "$(realpath "${cols}")" \
              "../${stem}.${b}.out" 2> "../${stem}.${b}.stderr" \
          && { [[ -f fort.98 ]] && mv fort.98 "../${stem}.${b}.fort98" || true; } )
    done
done

sha256sum "${kf_source}" "${script_dir}"/stub_wrf_error.F90 \
    "${script_dir}"/run_kf_oracle.F90 "${script_dir}"/kf_columns.py \
    "$@" ./*.out > oracle-sha256sums.txt
echo "oracle built and run in ${build_dir}"
