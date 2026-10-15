#!/usr/bin/env bash
# Build the Milbrandt-Yau (mp_physics=9) WRF v4.6.1 column oracle.
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILT_TREE]
#
# WRF_SOURCE_ROOT  pristine WRF checkout at the pinned commit (never edited).
# BUILD_DIR        where the oracle objects and executables go.
# WRF_BUILT_TREE   optional: a configured+compiled WRF tree (configure option
#                  32/34, GNU).  When given, its phys/module_mp_milbrandt2mom.o
#                  -- the very object inside that wrf.exe -- is linked as a
#                  second runner, and the object this script compiles must be
#                  byte-identical to it.
#
# The scheme is compiled with WRF's own GNU flags from arch/configure.defaults
# (configure.wrf FCOPTIM + FCBASEOPTS), after WRF's own CPP step.  An -O0
# build of the same preprocessed source is also made; run_oracle.sh requires
# the two to produce bit-identical output.  The build refuses if the scheme
# object references any libmvec (_ZGV*) vector entry point, because then the
# Fortran's own rounding would depend on loop position, not on the scheme.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILT_TREE]" >&2
    exit 2
fi
src_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
built_tree=${3:-}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
scheme="${src_root}/phys/module_mp_milbrandt2mom.F"
test -f "${scheme}"
head_commit=$(git -C "${src_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${src_root}" diff --quiet HEAD -- phys/module_mp_milbrandt2mom.F; then
    echo "phys/module_mp_milbrandt2mom.F differs from pinned HEAD" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

# WRF's GNU entry: CPP = /lib/cpp -P -nostdinc; FCOPTIM; FCBASEOPTS.
/lib/cpp -P -nostdinc -traditional-cpp "${scheme}" > module_mp_milbrandt2mom.f90 2>/dev/null \
  || /lib/cpp -P -nostdinc "${scheme}" > module_mp_milbrandt2mom.f90
wrf_flags=(-O2 -ftree-vectorize -funroll-loops -w -ffree-form -ffree-line-length-none
           -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch
           -fallow-invalid-boz)

mkdir -p o2 o0
gfortran -c "${wrf_flags[@]}" -J o2 -o o2/stub_wrf_error.o "${here}/stub_wrf_error.F90"
gfortran -c "${wrf_flags[@]}" -J o2 -I o2 -o o2/module_mp_milbrandt2mom.o module_mp_milbrandt2mom.f90
gfortran -c -O2 -J o2 -I o2 -o o2/run_my2.o "${here}/run_my2.F90"
gfortran -o run_my2_O2 o2/stub_wrf_error.o o2/module_mp_milbrandt2mom.o o2/run_my2.o

o0_flags=("${wrf_flags[@]}"); o0_flags[0]=-O0
gfortran -c "${o0_flags[@]}" -J o0 -o o0/stub_wrf_error.o "${here}/stub_wrf_error.F90"
gfortran -c "${o0_flags[@]}" -J o0 -I o0 -o o0/module_mp_milbrandt2mom.o module_mp_milbrandt2mom.f90
gfortran -c -O0 -J o0 -I o0 -o o0/run_my2.o "${here}/run_my2.F90"
gfortran -o run_my2_O0 o0/stub_wrf_error.o o0/module_mp_milbrandt2mom.o o0/run_my2.o

for obj in o2/module_mp_milbrandt2mom.o o0/module_mp_milbrandt2mom.o; do
    if nm -u "${obj}" | grep -q '_ZGV'; then
        echo "${obj} references a libmvec vector entry point:" >&2
        nm -u "${obj}" | grep '_ZGV' >&2
        exit 4
    fi
    # FMA would be a contraction the CUDA side must then reproduce; baseline
    # x86-64 has none, and the oracle states that rather than assumes it.
    if objdump -d "${obj}" | grep -Eq 'vfn?m(add|sub)'; then
        echo "${obj} contains FMA instructions" >&2
        exit 4
    fi
done

if [[ -n "${built_tree}" ]]; then
    wrf_obj="${built_tree}/phys/module_mp_milbrandt2mom.o"
    if ! cmp -s "${wrf_obj}" o2/module_mp_milbrandt2mom.o; then
        echo "compiled object differs from ${wrf_obj}" >&2
        exit 5
    fi
    mkdir -p wrfobj
    cp "${wrf_obj}" wrfobj/module_mp_milbrandt2mom.o
    gfortran -c -O2 -J wrfobj -I "${built_tree}/phys" -o wrfobj/run_my2.o "${here}/run_my2.F90"
    gfortran -o run_my2_wrfobj o2/stub_wrf_error.o wrfobj/module_mp_milbrandt2mom.o wrfobj/run_my2.o
    echo "object identical to the WRF build's: ${wrf_obj}" > wrf-object.txt
fi

{
    echo "wrf_commit ${pinned_commit}"
    echo "gfortran $(gfortran --version | head -1)"
    echo "glibc $(ldd --version | head -1)"
    echo "flags ${wrf_flags[*]}"
    echo "libm_imports $(nm -u o2/module_mp_milbrandt2mom.o | awk '{print $2}' | grep -Ev '^_gfortran|^(free|malloc|memcpy|memset)$' | tr '\n' ' ')"
    sha256sum "${scheme}" module_mp_milbrandt2mom.f90 "${here}/run_my2.F90" "${here}/stub_wrf_error.F90"
} > build-receipt.txt
cat build-receipt.txt
