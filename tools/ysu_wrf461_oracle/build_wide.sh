#!/usr/bin/env bash
# Build and run the WIDE YSU oracle (run_bl_ysu_wide.F90) against the
# byte-unmodified WRF v4.6.1 phys/physics_mmm/bl_ysu.F90.
#
#   bash tools/ysu_wrf461_oracle/build_wide.sh WRF_SOURCE_ROOT BUILD_DIR [OUT_DIR]
#
# OUT_DIR (default: tests/data/oracles/ysu_wide in this tree) receives
#   ysu-wide-inputs.bin / .json   the columns (make_wide_columns.py)
#   ysu-wide-wrf.bin              every word bl_ysu_run wrote, built with
#                                 WRF's own flags (THE reference)
#   ysu-wide-wrf-O0.bin           the same, built at -O0
#   build-receipt.txt             toolchain, symbols, and where the two differ
#   oracle-sha256sums.txt         sources and artefacts
#
# Two builds of the same, byte-unmodified WRF source:
#   * WRF's own phys/ flags from arch/configure.defaults (GNU entry):
#     -O2 -ftree-vectorize -funroll-loops, as the stock wrf.exe compiles it.
#     This is the reference: it is the arithmetic a WRF run performs.
#   * -O0, which every earlier oracle in this tree used.
# They are NOT the same function.  At -O1 and above GCC rewrites
# pow(x, 2.0) as x*x (and pow(x, 0.0) as 1.0) whatever the math flags,
# while -O0 calls powf; powf is not correctly rounded at exact ties, so the
# five `**2.` / `**pfac` sites of bl_ysu.F90 (:930, :949 twice, :960-961)
# can differ by one rounding.  The receipt records how many words differ.
# The script fails if either build pulls in a libmvec vector symbol or the
# -O0 object carries an FMA instruction.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build_wide.sh WRF_SOURCE_ROOT BUILD_DIR [OUT_DIR]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
out_dir=$(realpath -m "${3:-${repo_root}/tests/data/oracles/ysu_wide}")
kind_source="${source_root}/phys/ccpp_kind_types.F"
ysu_source="${source_root}/phys/physics_mmm/bl_ysu.F90"
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"

test -f "${kind_source}"
test -f "${ysu_source}"
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- \
        phys/ccpp_kind_types.F phys/physics_mmm/bl_ysu.F90; then
    echo "bl_ysu.F90 or ccpp_kind_types.F differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}" "${out_dir}"
cd "${build_dir}"

python3 "${script_dir}/make_wide_columns.py" "${out_dir}/ysu-wide-inputs.bin"

build() {   # build <tag> <flags...>
    local tag=$1; shift
    mkdir -p "${tag}"
    (cd "${tag}" &&
     gfortran -c "$@" -cpp -DRWORDSIZE=4 -ffree-form \
         -ffree-line-length-none "${kind_source}" &&
     gfortran -c "$@" -cpp -ffree-form -ffree-line-length-none \
         -I . -o bl_ysu.o "${ysu_source}" &&
     gfortran -c -O0 -ffree-form -ffree-line-length-none -I . \
         "${script_dir}/run_bl_ysu_wide.F90" &&
     gfortran -o run_bl_ysu_wide ccpp_kind_types.o bl_ysu.o \
         run_bl_ysu_wide.o &&
     nm -u bl_ysu.o | sed 's/^ *//' | sort > undefined.txt)
}
build O0 -O0
build wrfflags -O2 -ftree-vectorize -funroll-loops

if grep -q '_ZGV' O0/undefined.txt; then
    echo "-O0 object pulled in a libmvec vector symbol" >&2
    exit 4
fi
if grep -Eq '(^| )(vfmadd|vfmsub|vfnmadd)' <(objdump -d O0/bl_ysu.o); then
    echo "-O0 object contains FMA instructions" >&2
    exit 4
fi

if grep -q '_ZGV' wrfflags/undefined.txt; then
    echo "the WRF-flags object pulled in a libmvec vector symbol" >&2
    exit 4
fi
if grep -Eq '(^| )(vfmadd|vfmsub|vfnmadd)' <(objdump -d wrfflags/bl_ysu.o); then
    echo "the WRF-flags object contains FMA instructions" >&2
    exit 4
fi

./wrfflags/run_bl_ysu_wide "${out_dir}/ysu-wide-inputs.bin" "${out_dir}/ysu-wide-wrf.bin"
./O0/run_bl_ysu_wide "${out_dir}/ysu-wide-inputs.bin" "${out_dir}/ysu-wide-wrf-O0.bin"
ndiff=$(cmp -l "${out_dir}/ysu-wide-wrf.bin" "${out_dir}/ysu-wide-wrf-O0.bin" | wc -l || true)

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# WRF:      ${pinned_commit} (clean: bl_ysu.F90, ccpp_kind_types.F)"
    echo "# -O0 undefined symbols:"
    sed 's/^/#   /' O0/undefined.txt
    echo "# WRF flags (-O2 -ftree-vectorize -funroll-loops, THE REFERENCE) undefined symbols:"
    sed 's/^/#   /' wrfflags/undefined.txt
    echo "# FMA instructions in either object: 0"
    echo "# pow(x,2.0) calls left by the WRF-flags build (GCC folds them to x*x):"
    echo "#   -O0: $(objdump -dr O0/bl_ysu.o | grep -c 'R_X86_64_PLT32.*powf') powf call sites; WRF flags: $(objdump -dr wrfflags/bl_ysu.o | grep -c 'R_X86_64_PLT32.*powf') (after unrolling)"
    echo "# bytes differing between the WRF-flags and -O0 outputs: ${ndiff}"
} > "${out_dir}/build-receipt.txt"
cat "${out_dir}/build-receipt.txt"

(cd "${out_dir}" && sha256sum ysu-wide-inputs.bin ysu-wide-inputs.json \
     ysu-wide-wrf.bin ysu-wide-wrf-O0.bin > oracle-sha256sums.txt)
sha256sum "${kind_source}" "${ysu_source}" \
    "${script_dir}/run_bl_ysu_wide.F90" "${script_dir}/make_wide_columns.py" \
    | sed -e "s#${source_root}#<wrf-4.6.1-checkout>#" \
          -e "s#${repo_root}#<oracle-workspace>#" \
    >> "${out_dir}/oracle-sha256sums.txt"
# The reference-platform tanhf sample the GPU test reads is written by
# tanhf_sweep.py --write-sample on a GPU box; its hash joins this receipt.
if [[ -f "${out_dir}/tanhf-ref.bin" ]]; then
    (cd "${out_dir}" && sha256sum tanhf-ref.bin >> oracle-sha256sums.txt)
fi
echo "wide oracle written to ${out_dir}"
