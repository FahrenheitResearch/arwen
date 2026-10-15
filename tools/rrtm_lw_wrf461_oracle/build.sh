#!/usr/bin/env bash
# Build and run the RRTM longwave (ra_lw_physics=1) column oracle against the
# byte-unmodified WRF v4.6.1 phys/module_ra_rrtm.F.
#
# usage: build.sh WRF_SOURCE_ROOT BUILD_DIR INPUT.bin
#
# Two builds of the same untouched module:
#   O0     the REFERENCE: -O0, so every EXP/LOG/** is one scalar libm call and
#          nothing is reassociated or vectorised.  This is what WOOF is graded
#          against, bit for bit.
#   O2vec  WRF's own phys/ flags from arch/configure.defaults (GNU):
#          -O2 -ftree-vectorize -funroll-loops.  Built and run only to measure
#          how far stock WRF on this toolchain sits from the reference (the
#          vectoriser may call libmvec's 4-ULP vector EXP); never graded.
# Both builds carry WRF's -fconvert=big-endian -frecord-marker=4, without which
# rrtm_lookuptable cannot read RRTM_DATA.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR INPUT.bin" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
input=$(realpath "$3")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rrtm_source="${source_root}/phys/module_ra_rrtm.F"
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"

test -f "${rrtm_source}"
test -f "${source_root}/run/RRTM_DATA"
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- phys/module_ra_rrtm.F run/RRTM_DATA; then
    echo "module_ra_rrtm.F or RRTM_DATA differs from the pinned commit" >&2
    exit 3
fi
# The packaged copy WOOF reads its tables from must be this exact file.
want_rrtm_sha=07ebd4e83b2a571837a27748a29ca678440c363f96a3694fc9549ae0911335da
have_rrtm_sha=$(sed 's/$/\r/' "${rrtm_source}" | sha256sum | cut -d' ' -f1)
if [[ "${have_rrtm_sha}" != "${want_rrtm_sha}" ]]; then
    echo "module_ra_rrtm.F (CRLF form) sha256 ${have_rrtm_sha} != ${want_rrtm_sha}" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"
cp "${source_root}/run/RRTM_DATA" .

CPPDEFS="-cpp -DEM_CORE=1 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
COMMON="-ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch"

build_variant() {
    local tag=$1; shift
    local opt="$*"
    mkdir -p "${tag}"
    ( cd "${tag}"
      gfortran -c ${opt} ${COMMON} ${CPPDEFS} "${script_dir}/stub_wrf.F90"
      gfortran -c ${opt} ${COMMON} ${CPPDEFS} -I . -o module_ra_rrtm.o "${rrtm_source}"
      gfortran -c ${opt} ${COMMON} -I . "${script_dir}/run_rrtm_lw.F90"
      gfortran -o run_rrtm_lw stub_wrf.o module_ra_rrtm.o run_rrtm_lw.o
      nm -u module_ra_rrtm.o | sed 's/^ *//' | sort > undefined.txt
      cp ../RRTM_DATA . )
}

build_variant O0 -O0
if grep -q '_ZGV' O0/undefined.txt; then
    echo "-O0 object pulled in a libmvec vector symbol" >&2
    exit 4
fi
build_variant O2vec -O2 -ftree-vectorize -funroll-loops

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 (THE REFERENCE) libm symbols:"
    grep -E 'expf|logf|powf|exp$|log$|pow$|_ZGV' O0/undefined.txt || true
    echo "# -O2 -ftree-vectorize -funroll-loops (WRF's GNU phys/ flags) libm symbols:"
    grep -E 'expf|logf|powf|exp$|log$|pow$|_ZGV' O2vec/undefined.txt || true
} > libmvec-report.txt
cat libmvec-report.txt

( cd O0 && ./run_rrtm_lw "${input}" ../wrf-out-O0.bin )
( cd O2vec && ./run_rrtm_lw "${input}" ../wrf-out-O2vec.bin )

sha256sum "${rrtm_source}" "${source_root}/run/RRTM_DATA" \
    "${script_dir}/run_rrtm_lw.F90" "${script_dir}/stub_wrf.F90" \
    "${input}" wrf-out-O0.bin wrf-out-O2vec.bin > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built and run in ${build_dir}"
