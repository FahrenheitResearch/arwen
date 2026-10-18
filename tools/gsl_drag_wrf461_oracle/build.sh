#!/usr/bin/env bash
# Build and run the GSL drag (gwd_opt = 3) column oracle against WRF v4.6.1.
#
#   bash tools/gsl_drag_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh lays
# them out); each is checked before anything compiles.  Three executables,
# all on the same columns and the same byte-unmodified module_bl_gwdo_gsl.F:
#
#   fixtures/           THE ORACLE OF RECORD.  -O0, scalar libm, no FMA
#                       (baseline x86-64 has no FMA instruction for -O0 to
#                       emit).  Fortran's own semantics, one libm call per
#                       intrinsic.
#   fixtures-wrfflags/  The module compiled with WRF v4.6.1's own gfortran
#                       flags from arch/configure.defaults (FCOPTIM -O2
#                       -ftree-vectorize -funroll-loops, FCBASEOPTS -w
#                       -ffree-form -ffree-line-length-none
#                       -fconvert=big-endian -frecord-marker=4, FCCOMPAT
#                       -fallow-argument-mismatch -fallow-invalid-boz).
#                       Measured against, not matched to: gfortran
#                       vectorizes the form-drag loop to libmvec there.
#   fixtures-cr-atan2/  -O0 again with cr_atan2f.c linked in, which replaces
#                       the C library's atan2f by the correctly rounded one.
#                       Attribution only.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | sha256sum -c --quiet - ) || {
    echo "a WRF source differs from SOURCES.sha256; refusing to build" >&2
    exit 3
}
mkdir -p "${build_dir}"
cd "${build_dir}"
rm -rf fixtures fixtures-wrfflags fixtures-cr-atan2 o0 o2 *.o *.mod
ff=(-ffree-form -ffree-line-length-none)
gsl="${source_root}/phys/module_bl_gwdo_gsl.F"
io="${script_dir}/../urban_wrf471_oracle/oracle_io.F90"

build_variant() {   # dir, flags for the WRF module..., then extra objects
    local dir=$1; shift
    mkdir -p "${dir}"
    ( cd "${dir}"
      gfortran -c "$@" -cpp -o module_bl_gwdo_gsl.o "${gsl}"
      gfortran -c -O0 "${ff[@]}" -o oracle_io.o "${io}"
      gfortran -c -O0 "${ff[@]}" -o gsl_columns.o "${script_dir}/gsl_columns.F90"
      gfortran -c -O0 "${ff[@]}" -o run_gsl_oracle.o "${script_dir}/run_gsl_oracle.F90"
    )
}
build_variant o0 -O0 "${ff[@]}"
build_variant o2 -O2 -ftree-vectorize -funroll-loops -w "${ff[@]}" \
    -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch \
    -fallow-invalid-boz
gcc -c -O0 -o o0/cr_atan2f.o "${script_dir}/cr_atan2f.c"

# The reference object must be scalar libm: no libmvec symbol at -O0.
if nm -u o0/module_bl_gwdo_gsl.o | grep -q '_ZGV'; then
    echo "-O0 module_bl_gwdo_gsl.o pulled in a libmvec vector symbol" >&2
    exit 4
fi
{
    echo "# -O0 module_bl_gwdo_gsl.o libm symbols:"
    nm -u o0/module_bl_gwdo_gsl.o | grep -E 'expf|powf|logf|sinf|atan2f|_ZGV' || true
    echo "# WRF-flags (-O2 -ftree-vectorize -funroll-loops) libm symbols:"
    nm -u o2/module_bl_gwdo_gsl.o | grep -E 'expf|powf|logf|sinf|atan2f|_ZGV' || true
} > libm-symbols.txt

( cd o0 && gfortran -o run_gsl_oracle run_gsl_oracle.o gsl_columns.o \
      module_bl_gwdo_gsl.o oracle_io.o -lm )
( cd o0 && gfortran -o run_gsl_oracle_cr run_gsl_oracle.o gsl_columns.o \
      module_bl_gwdo_gsl.o oracle_io.o cr_atan2f.o -lm )
( cd o2 && gfortran -o run_gsl_oracle run_gsl_oracle.o gsl_columns.o \
      module_bl_gwdo_gsl.o oracle_io.o -lm )
if nm o0/run_gsl_oracle_cr | grep -q ' T atan2f$'; then :; else
    echo "the attribution build did not take cr_atan2f.c's atan2f" >&2
    exit 5
fi
./o0/run_gsl_oracle fixtures
./o2/run_gsl_oracle fixtures-wrfflags
./o0/run_gsl_oracle_cr fixtures-cr-atan2

{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "host: $(uname -srm)"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
} > fixtures/TOOLCHAIN.txt
cp libm-symbols.txt fixtures/
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" ) \
    > fixtures/SOURCES.sha256
( cd "${script_dir}" && sha256sum build.sh fetch_sources.sh gsl_columns.F90 \
    run_gsl_oracle.F90 cr_atan2f.c pack.py ../urban_wrf471_oracle/oracle_io.F90 ) \
    > fixtures/HARNESS.sha256
echo "fixtures in ${build_dir}/fixtures (also fixtures-wrfflags, fixtures-cr-atan2)"
