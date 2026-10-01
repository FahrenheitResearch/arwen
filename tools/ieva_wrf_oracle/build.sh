#!/bin/bash
# Build the IEVA oracle against a compiled WRF 4.7.1 tree.
#
#   WRF=/path/to/WRF-4.7.1 bash build.sh OUTDIR wrf     # WRF's own routines
#   WRF=/path/to/WRF-4.7.1 bash build.sh OUTDIR a179    # with A179's corrections
#
# ``wrf`` links ieva_oracle.F90 to main/libwrflib.a as WRF built it.
# ``a179`` first writes WRF's module_ieva_em.f90 (the source WRF's build
# preprocessed) with wrf_a179.py's two advect_w_implicit corrections,
# compiles it with WRF's own Fortran flags (configure.wrf FCOPTIM and
# FCBASEOPTS), and links it ahead of the library, so every IEVA routine
# comes from that object and calc_mu_uv_1 still from WRF's.  Either way the
# binary is OUTDIR/ieva_oracle; run it as ``ieva_oracle DIR [chain]``.
# The driver itself is compiled without WRF's -fconvert=big-endian: gfortran
# applies that flag to the program's unformatted I/O, and the captures are
# little-endian float32.
set -euo pipefail
OUT=$1
MODE=$2
WRF=${WRF:?set WRF to a compiled WRF 4.7.1 tree}
HERE=$(cd "$(dirname "$0")" && pwd)
FC=${FC:-mpif90}
FLAGS="-O2 -ftree-vectorize -funroll-loops -fno-lto -w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch -fallow-invalid-boz"
DRIVER="-O2 -fno-lto -w -ffree-form -ffree-line-length-none -fallow-argument-mismatch"
INC="-I$WRF/main -I$WRF/frame -I$WRF/share -I$WRF/phys -I$WRF/inc -I$WRF/external/esmf_time_f90"
LIBS="$WRF/main/libwrflib.a $WRF/external/fftpack/fftpack5/libfftpack.a $WRF/external/io_grib1/libio_grib1.a $WRF/external/io_grib_share/libio_grib_share.a $WRF/external/io_int/libwrfio_int.a -L$WRF/external/esmf_time_f90 -lesmf_time $WRF/external/RSL_LITE/librsl_lite.a $WRF/frame/module_internal_header_util.o $WRF/frame/pack_utils.o -L$WRF/external/io_netcdf -lwrfio_nf ${WRF_NETCDF_LIBS:--lnetcdff -lnetcdf} -lm"
mkdir -p "$OUT"
cd "$OUT"
case "$MODE" in
  wrf)
    $FC $DRIVER $INC -I"$WRF/dyn_em" -c "$HERE/ieva_oracle.F90" -o ieva_oracle.o
    $FC $DRIVER -o ieva_oracle ieva_oracle.o $LIBS ;;
  a179)
    python3 "$HERE/wrf_a179.py" "$WRF/dyn_em/module_ieva_em.f90" module_ieva_em.f90
    $FC $FLAGS -I. $INC -J. -c module_ieva_em.f90 -o module_ieva_em.o
    $FC $DRIVER -DA179 -I. $INC -I"$WRF/dyn_em" -c "$HERE/ieva_oracle.F90" -o ieva_oracle.o
    $FC $DRIVER -o ieva_oracle ieva_oracle.o module_ieva_em.o $LIBS ;;
  *) echo "mode must be wrf or a179" >&2; exit 2 ;;
esac
echo "built $OUT/ieva_oracle ($MODE)"
