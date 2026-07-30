#!/bin/sh
# Build the WRF v4.6.1 Noah-MP PHENOLOGY / PRECIP_HEAT bitwise oracle.
#
# Usage:  build_vegprecip.sh <wrf-tree> <workdir> [optlevel]
#
#   <wrf-tree>  pinned WRF checkout, e.g. /home/drew/wrf-stock-v461-gate-20260721
#               commit d66e442fccc04111067e29274c9f9eaccc3cef28
#   <workdir>   scratch directory (created if absent)
#   [optlevel]  "wrf" (default) reproduces WRF's own gfortran FCOPTIM
#               "noopt" builds at -O0
#               "nocontract" builds at WRF FCOPTIM with -ffp-contract=off
#               The latter two are negative controls: the fixture must not
#               depend on the optimiser or on compiler FP contraction.
#               "objproof" does not build the oracle at all: it compiles the
#               pristine and patched modules at -O0 and proves gfortran emits
#               the same code for both.
#
# Note for "objproof": the two copies MUST have the same file *basename*.
# gfortran embeds the source filename in .rodata for runtime diagnostics, so
# compiling ./pristine.F and ./patched.F makes .rodata differ by exactly that
# string and the proof reports a spurious failure.  They are therefore built
# in sibling directories under the module's own name.
#
# The only change made to phys/module_sf_noahmplsm.F is the accessibility lift
# performed by visibility_patch_leaves.py, which self-checks the pristine
# sha256 and proves the diff is nothing but private:: -> public::.
set -eu

WRF_TREE=$1
WORK=$2
OPTLEVEL=${3:-wrf}

HERE=$(cd "$(dirname "$0")" && pwd)

case "$OPTLEVEL" in
  wrf)        FCOPTIM="-O2 -ftree-vectorize -funroll-loops" ;;
  noopt)      FCOPTIM="-O0" ;;
  nocontract) FCOPTIM="-O2 -ftree-vectorize -funroll-loops -ffp-contract=off" ;;
  objproof)   FCOPTIM="-O0" ;;
  *)          echo "unknown optlevel: $OPTLEVEL" >&2; exit 2 ;;
esac

# WRF arch/configure.defaults, "Linux x86_64, gfortran" block:
#   FORMAT_FREE = -ffree-form -ffree-line-length-none
#   BYTESWAPIO  = -fconvert=big-endian -frecord-marker=4
#   FCBASEOPTS_NO_G = -w $(FORMAT_FREE) $(BYTESWAPIO) $(FCCOMPAT)
FCBASE="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4"

mkdir -p "$WORK"
cd "$WORK"

if [ "$OPTLEVEL" = objproof ]; then
  FCBASE="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4"
  rm -rf pristine patched
  mkdir -p pristine patched
  cp "$WRF_TREE/phys/module_sf_noahmplsm.F" pristine/module_sf_noahmplsm.F
  cp "$WRF_TREE/phys/module_sf_gecros.F"    pristine/
  cp "$WRF_TREE/phys/module_sf_gecros.F"    patched/
  python3 "$HERE/visibility_patch_leaves.py" pristine/module_sf_noahmplsm.F \
          --out patched/module_sf_noahmplsm.F --check \
          --require-symbol PHENOLOGY --require-symbol PRECIP_HEAT
  for d in pristine patched; do
    ( cd "$d" \
      && gfortran -c $FCBASE $FCOPTIM module_sf_gecros.F      -o gecros.o \
      && gfortran -c $FCBASE $FCOPTIM module_sf_noahmplsm.F   -o "../$d.o" )
  done
  python3 "$HERE/compare_object_code_vegeflux.py" pristine.o patched.o --expect-total
  exit $?
fi

cp "$WRF_TREE/phys/module_sf_noahmplsm.F" ./pristine.F
cp "$WRF_TREE/phys/module_sf_gecros.F"    ./module_sf_gecros.F

python3 "$HERE/visibility_patch_leaves.py" --self-test

python3 "$HERE/visibility_patch_leaves.py" ./pristine.F \
        --out ./module_sf_noahmplsm_public.F \
        --check \
        --require-symbol PHENOLOGY \
        --require-symbol PRECIP_HEAT

cat > wrf_stubs.f90 <<'STUB'
! Minimal replacements for the WRF frame routines the physics module calls.
! Neither PHENOLOGY nor PRECIP_HEAT reaches any of them on any path; they exist
! only so that the module links.
subroutine wrf_error_fatal(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') 'wrf_error_fatal: '//trim(msg)
  stop 1
end subroutine wrf_error_fatal

subroutine wrf_error_fatal3(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') 'wrf_error_fatal3: '//trim(msg)
  stop 1
end subroutine wrf_error_fatal3

subroutine wrf_message(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') trim(msg)
end subroutine wrf_message

subroutine wrf_debug(lvl, msg)
  integer, intent(in) :: lvl
  character(len=*), intent(in) :: msg
end subroutine wrf_debug
STUB

gfortran -c $FCBASE $FCOPTIM module_sf_gecros.F           -o module_sf_gecros.o
gfortran -c $FCBASE $FCOPTIM module_sf_noahmplsm_public.F -o module_sf_noahmplsm.o
gfortran -c -w -O2 wrf_stubs.f90 -o wrf_stubs.o
gfortran -c $FCBASE $FCOPTIM "$HERE/run_vegprecip.F90" -o run_vegprecip.o
gfortran -o run_vegprecip run_vegprecip.o module_sf_noahmplsm.o \
         module_sf_gecros.o wrf_stubs.o -lm

echo "built $WORK/run_vegprecip  (optlevel=$OPTLEVEL, FCOPTIM='$FCOPTIM')"
