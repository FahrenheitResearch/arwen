#!/usr/bin/env bash
# Build and run the revised MM5 surface-layer oracle (sf_sfclay_physics=1)
# against the byte-unmodified WRF v4.6.1 phys/module_sf_sfclayrev.F (the WRF
# wrapper SFCLAYREV) and phys/physics_mmm/sf_sfclayrev.F90 (the scheme).
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR [INPUTS] [--seed-exchange]
#
# INPUTS defaults to tests/data/oracles/sfclayrev/sfclayrev-inputs.hex next
# to this tool.  Writes BUILD_DIR/sfclayrev-outputs.hex plus receipts.
#
# Reference build: gfortran -O0 (no vectoriser, no FMA on baseline x86-64),
# glibc's scalar libm.  WRF's own -O2 -ftree-vectorize could swap a scalar
# libm call for a libmvec vector form; this script fails if any _ZGV* symbol
# appears in the -O0 objects, and a positive control proves the grep can see
# one at all on this toolchain.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [INPUTS] [--seed-exchange]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
inputs=$(realpath "${3:-${repo_root}/tests/data/oracles/sfclayrev/sfclayrev-inputs.hex}")
kind_source="${source_root}/phys/ccpp_kind_types.F"
scheme_source="${source_root}/phys/physics_mmm/sf_sfclayrev.F90"
wrapper_source="${source_root}/phys/module_sf_sfclayrev.F"
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"

test -f "${kind_source}"
test -f "${scheme_source}"
test -f "${wrapper_source}"
test -f "${inputs}"

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- \
        phys/ccpp_kind_types.F phys/physics_mmm/sf_sfclayrev.F90 \
        phys/module_sf_sfclayrev.F; then
    echo "a pinned source differs from ${pinned_commit}" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

# --- the reference build: -O0 -----------------------------------------------
# EM_CORE=1 is what WRF's ARW build defines; it selects the wrapper's call
# with isftcflx, iz0tlnd, scm_force_flux, ustm and ck/cka/cd/cda present.
gfortran -c -O0 -cpp -DRWORDSIZE=4 -ffree-form -ffree-line-length-none \
    "${kind_source}"
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none \
    -I "${build_dir}" -o sf_sfclayrev_O0.o "${scheme_source}"
gfortran -c -O0 -cpp -DEM_CORE=1 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" -o module_sf_sfclayrev_O0.o "${wrapper_source}"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_sfclayrev.F90"
gfortran -o run_sfclayrev ccpp_kind_types.o sf_sfclayrev_O0.o \
    module_sf_sfclayrev_O0.o run_sfclayrev.o

# --- receipts ---------------------------------------------------------------
nm -u sf_sfclayrev_O0.o module_sf_sfclayrev_O0.o | sed 's/^ *//' \
    | grep -v ':$' | sort -u > undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-O0.txt >&2
    exit 4
fi
gfortran -c -O2 -ftree-vectorize -cpp -ffree-form -ffree-line-length-none \
    -I "${build_dir}" -o sf_sfclayrev_O2vec.o "${scheme_source}"
nm -u sf_sfclayrev_O2vec.o | sed 's/^ *//' | sort > undefined-O2vec.txt
gfortran -c -Ofast -ftree-vectorize \
    -o libmvec_positive_control.o "${script_dir}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol; the guard is" \
         "vacuous on this toolchain" >&2
    exit 5
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 libm symbols (THE REFERENCE):"
    grep -E 'expf|powf|logf|atanf|sqrtf|_ZGV' undefined-O0.txt || true
    echo "# -O2 -ftree-vectorize (WRF's own phys/ setting):"
    grep -E 'expf|powf|logf|atanf|sqrtf|_ZGV' undefined-O2vec.txt || true
    echo "# positive control (plain expf loop at -Ofast):"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt
cat libmvec-report.txt

# --- run --------------------------------------------------------------------
./run_sfclayrev "${inputs}" sfclayrev-outputs.hex "${4:-}"

sha256sum "${kind_source}" "${scheme_source}" "${wrapper_source}" \
    "${script_dir}/run_sfclayrev.F90" "${inputs}" \
    sfclayrev-outputs.hex > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built in ${build_dir}"
