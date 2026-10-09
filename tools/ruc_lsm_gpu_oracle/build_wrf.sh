#!/usr/bin/env bash
# Compile the unmodified WRF v4.6.1 RUC LSM and its 2-m diagnostic with the
# generated column driver, three ways, and run each on a case directory that
# run_woof.py has filled (inputs.bin + run_columns.F90).  CPU only.
#
#   defined : WRF's own GNU flags (arch/configure.defaults, as built on this
#             box) plus -finit-local-zero.  LSMRUC reads locals before any
#             write (SFCTMP's ilnb at module_sf_ruclsm.F:4410 under a pack
#             thinner than snth; LSMRUC's snoh/snflx/s/sublim/evapl after
#             ktau 1).  Zero-initialising them gives those reads one defined
#             value, the one WOOF uses, so the comparison is meaningful.
#   stock   : WRF's own GNU flags, nothing added.  What wrf.exe does.  Any
#             word that differs from "defined" is the uninitialised read.
#   o0      : -O0 -finit-local-zero; the optimiser cross-check.
#
# usage: build_wrf.sh WRF_SOURCE_ROOT CASE_DIR [ARMS]
set -euo pipefail
src=$(realpath "$1")
case_dir=$(realpath "$2")
arms=${3:-defined stock o0}
tool=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stub="$tool/../ruc_hrrr_fork_oracle/column_stubs.F90"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
wrf_flags="-O2 -ftree-vectorize -funroll-loops -w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch -fallow-invalid-boz"
cd "$case_dir"
cp "$src/run/VEGPARM.TBL" "$src/run/SOILPARM.TBL" "$src/run/GENPARM.TBL" .
for arm in $arms; do
    case $arm in
        defined) flags="$wrf_flags -finit-local-zero" ;;
        stock)   flags="$wrf_flags" ;;
        o0)      flags="-O0 -w -ffree-form -ffree-line-length-none -fallow-argument-mismatch -fallow-invalid-boz -finit-local-zero" ;;
        *) echo "unknown arm $arm" >&2; exit 2 ;;
    esac
    b="build-$arm"
    mkdir -p "$b"
    (
        cd "$b"
        gfortran -c $flags -cpp -DNMM_CORE=0 "$src/share/module_model_constants.F"
        gfortran -c $flags "$stub"
        gfortran -c $flags -cpp -DEM_CORE=1 -Dwrf_chem=0 "$src/phys/module_sf_ruclsm.F"
        gfortran -c $flags -cpp "$src/phys/module_sf_sfcdiags_ruclsm.F"
        # The driver is not WRF code: no big-endian conversion on its stream I/O.
        gfortran -c -O0 -ffree-form -ffree-line-length-none "$case_dir/run_columns.F90"
        gfortran -o run_columns module_model_constants.o column_stubs.o \
            module_sf_ruclsm.o module_sf_sfcdiags_ruclsm.o run_columns.o
    )
    "./$b/run_columns" inputs.bin "outputs-$arm.bin" > "wrf-$arm.log" 2>&1
done
gfortran --version | sed -n 1p > compiler.txt
ldd --version | sed -n 1p >> compiler.txt
sha256sum "$src/phys/module_sf_ruclsm.F" "$src/phys/module_sf_sfcdiags_ruclsm.F" \
    "$src/share/module_model_constants.F" VEGPARM.TBL SOILPARM.TBL GENPARM.TBL \
    "$stub" run_columns.F90 inputs.bin outputs-*.bin > sha256.txt
