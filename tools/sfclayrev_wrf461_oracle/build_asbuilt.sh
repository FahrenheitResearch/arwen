#!/usr/bin/env bash
# Run the revised MM5 oracle driver against the objects a real WRF v4.6.1
# build compiled (WRF's own configure flags, -O2 -ftree-vectorize
# -funroll-loops), instead of the -O0 reference build of build.sh.
#
#   bash build_asbuilt.sh WRF_BUILD_ROOT BUILD_DIR [INPUTS]
#
# WRF_BUILD_ROOT is a compiled WRF tree (for example W1's
# /work/pverify/wrf-build/serial/WRF-4.6.1).  Writes
# BUILD_DIR/sfclayrev-outputs-asbuilt.hex and asbuilt-receipt.txt.
#
# This is a measurement of the build, not the reference.  At -O2 gfortran
# rewrites the scheme's REAL powers x**2. into x*x (the four powf(x, 2.0)
# calls of the -O0 tree dump are gone at -O2), and the object imports one
# libmvec vector powf (_ZGVbN4vv_powf) whose rounding depends on where a
# column sits in a vectorised loop.  build.sh's -O0 object is the scheme's
# own arithmetic and stays the 0 ULP reference.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build_asbuilt.sh WRF_BUILD_ROOT BUILD_DIR [INPUTS]" >&2
    exit 2
fi

wrf_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
inputs=$(realpath "${3:-${repo_root}/tests/data/oracles/sfclayrev/sfclayrev-inputs.hex}")
phys="${wrf_root}/phys"

for f in ccpp_kind_types.o ccpp_kind_types.mod module_sf_sfclayrev.o \
         module_sf_sfclayrev.mod sf_sfclayrev.mod physics_mmm/sf_sfclayrev.o; do
    test -f "${phys}/${f}"
done

mkdir -p "${build_dir}"
cd "${build_dir}"
cp "${phys}/ccpp_kind_types.o" "${phys}/ccpp_kind_types.mod" \
   "${phys}/module_sf_sfclayrev.o" "${phys}/module_sf_sfclayrev.mod" \
   "${phys}/sf_sfclayrev.mod" "${phys}/physics_mmm/sf_sfclayrev.o" .
gfortran -c -O0 -ffree-form -ffree-line-length-none -I "${build_dir}" \
    "${script_dir}/run_sfclayrev.F90"
gfortran -o run_sfclayrev_asbuilt ccpp_kind_types.o sf_sfclayrev.o \
    module_sf_sfclayrev.o run_sfclayrev.o -lmvec -lm
./run_sfclayrev_asbuilt "${inputs}" sfclayrev-outputs-asbuilt.hex

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# WRF build: ${wrf_root}"
    echo "# libm symbols the as-built scheme and wrapper import:"
    nm -u sf_sfclayrev.o module_sf_sfclayrev.o | sed 's/^ *//' \
        | grep -E 'expf|powf|logf|atanf|sqrtf|_ZGV' | sort -u
    sha256sum ccpp_kind_types.o sf_sfclayrev.o module_sf_sfclayrev.o \
        "${script_dir}/run_sfclayrev.F90" "${inputs}" \
        sfclayrev-outputs-asbuilt.hex
} > asbuilt-receipt.txt
cat asbuilt-receipt.txt
echo "as-built oracle run in ${build_dir}"
