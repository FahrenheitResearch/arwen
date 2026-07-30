#!/bin/sh
# Build the WRF v4.6.1 Noah-MP WATER bitwise oracle: the driver at 5954-6261
# that sequences CANWATER, SNOWWATER and SOILWATER.
#
# Usage:  build_water.sh <wrf-tree> <workdir> [optlevel]
#
#   <wrf-tree>  pinned WRF checkout, e.g. /home/drew/wrf-stock-v461-gate-20260721
#               commit d66e442fccc04111067e29274c9f9eaccc3cef28
#   <workdir>   scratch directory (created if absent)
#   [optlevel]  "noopt" (default) builds at -O0 and is the fixture
#               "nocontract" -O0 with -ffp-contract=off
#               "snan" -O0 with -finit-real=snan -finit-integer=-2147483647
#                      -finit-logical=false
#               "wrf" reproduces WRF's own gfortran FCOPTIM (see below)
#               "objproof" does not build the oracle at all: it compiles the
#                      pristine and patched modules at -O0 and proves gfortran
#                      emits the same code for both.
#
# Why the fixture is -O0
# ----------------------
# WATER calls SOILWATER, and at WRF's own FCOPTIM
# (-O2 -ftree-vectorize -funroll-loops) gfortran 13.3.0 vectorises SOILWATER's
# frozen-fraction loop (7333-7337) into glibc's libmvec `_ZGVbN4v_expf` instead
# of scalar `expf`.  libmvec's 4-wide expf carries a 4-ULP accuracy contract
# rather than glibc's scalar implementation, so no port that calls expf can
# reproduce it.  See gpuwm/data/noahmp/oracle/PROVENANCE-soilwater.md.
#
# Stage 5 below runs `nm -u` on the compiled module and FAILS if any libmvec
# symbol is present.  That is not decoration: it is the check that would have
# caught the trap in the first place, and it is what makes the -O0 claim
# falsifiable rather than inherited.
#
# "nocontract" and "snan" are negative controls and must produce a
# byte-identical CSV: the first pins that no emitted value depends on compiler
# FP contraction, the second that none reads an uninitialised local.  The
# latter is load-bearing here -- WATER's QIN/QDIS are INTENT(OUT) scalars that
# no live statement writes, and SNOWWATER's DIVIDE leaves slots above ISNOW
# conditionally assigned.
#
# "wrf" is a *recorded divergence*, not a gate.  Diff it against the fixture
# and bound what differs; do not widen anything to make it match.
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
OPTLEVEL=${3:-noopt}

HERE=$(cd "$(dirname "$0")" && pwd)

PRISTINE_SHA=bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282

EXTRA=""
case "$OPTLEVEL" in
  noopt)      FCOPTIM="-O0" ;;
  nocontract) FCOPTIM="-O0 -ffp-contract=off" ;;
  snan)       FCOPTIM="-O0"
              EXTRA="-finit-real=snan -finit-integer=-2147483647 -finit-logical=false" ;;
  wrf)        FCOPTIM="-O2 -ftree-vectorize -funroll-loops" ;;
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

# Stage 1: the pinned identity of the physics source, before anything else.
GOT=$(sha256sum "$WRF_TREE/phys/module_sf_noahmplsm.F" | cut -d' ' -f1)
if [ "$GOT" != "$PRISTINE_SHA" ]; then
  echo "module_sf_noahmplsm.F is not the pinned file" >&2
  echo "  expected $PRISTINE_SHA" >&2
  echo "  got      $GOT" >&2
  exit 3
fi
echo "[1] pristine module_sf_noahmplsm.F sha256 $GOT"

if [ "$OPTLEVEL" = objproof ]; then
  rm -rf pristine patched
  mkdir -p pristine patched
  cp "$WRF_TREE/phys/module_sf_noahmplsm.F" pristine/module_sf_noahmplsm.F
  cp "$WRF_TREE/phys/module_sf_gecros.F"    pristine/
  cp "$WRF_TREE/phys/module_sf_gecros.F"    patched/
  python3 "$HERE/visibility_patch_leaves.py" pristine/module_sf_noahmplsm.F \
          --out patched/module_sf_noahmplsm.F --check \
          --require-symbol WATER --require-symbol CANWATER \
          --require-symbol SNOWWATER --require-symbol SOILWATER
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

# Stage 2: the checker's own negative controls, then the lift itself.
python3 "$HERE/visibility_patch_leaves.py" --self-test
echo "[2] visibility_patch_leaves.py self-test passed"

python3 "$HERE/visibility_patch_leaves.py" ./pristine.F \
        --out ./module_sf_noahmplsm_public.F \
        --check \
        --require-symbol WATER \
        --require-symbol CANWATER \
        --require-symbol SNOWWATER \
        --require-symbol SOILWATER
echo "[3] accessibility lift applied; diff is exactly private:: -> public::"

cat > wrf_stubs.f90 <<'STUB'
! Service-only stubs for the standalone Noah-MP WATER oracle harness.
!
! MODULE_SF_NOAHMPLSM references exactly three external procedures --
! wrf_message, wrf_error_fatal and wrf_debug -- all of which are WRF logging
! and termination services, not physics.  Nothing in this file computes a
! physical quantity, and wrf_error_fatal aborts rather than returning, so a
! harness case that trips a WRF fatal path fails loudly instead of emitting a
! fabricated row.
subroutine wrf_error_fatal(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') 'noahmp water oracle: WRF fatal: '//trim(msg)
  stop 1
end subroutine wrf_error_fatal

subroutine wrf_error_fatal3(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') 'noahmp water oracle: WRF fatal: '//trim(msg)
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

gfortran -c $FCBASE $FCOPTIM $EXTRA module_sf_gecros.F           -o module_sf_gecros.o
gfortran -c $FCBASE $FCOPTIM $EXTRA module_sf_noahmplsm_public.F -o module_sf_noahmplsm.o
gfortran -c -w -O2 wrf_stubs.f90 -o wrf_stubs.o
gfortran -c $FCBASE $FCOPTIM $EXTRA "$HERE/run_water.F90" -o run_water.o
gfortran -o run_water run_water.o module_sf_noahmplsm.o \
         module_sf_gecros.o wrf_stubs.o -lm
echo "[4] built $WORK/run_water  (optlevel=$OPTLEVEL, FCOPTIM='$FCOPTIM $EXTRA')"

# Stage 5: trap 3.  gfortran's auto-vectoriser silently swaps scalar libm for
# glibc's libmvec, which is a different function with a different accuracy
# contract.  Refuse to emit a fixture from a build that carries one.
UNDEF=$(nm -u module_sf_noahmplsm.o run_water.o | awk '{print $NF}' | sort -u)
VEC=$(printf '%s\n' "$UNDEF" | grep -E '^_ZGV' || true)
if [ "$OPTLEVEL" = wrf ]; then
  echo "[5] libmvec references at WRF FCOPTIM (recorded, not gated):"
  printf '%s\n' "${VEC:-  (none)}" | sed 's/^/      /'
else
  if [ -n "$VEC" ]; then
    echo "libmvec symbol(s) present in a build that claims scalar libm:" >&2
    printf '%s\n' "$VEC" | sed 's/^/  /' >&2
    exit 4
  fi
  echo "[5] nm -u: no libmvec symbol; every transcendental is scalar libm"
  printf '%s\n' "$UNDEF" | grep -E '^(expf|powf|logf|log10f|sqrtf|exp|pow|log)$' \
      | sed 's/^/      scalar: /' || true
fi

./run_water noahmp-water.csv noahmp-water-libm.csv
echo "[6] fixture: $(($(wc -l < noahmp-water.csv) - 1)) data rows"

python3 "$HERE/validate_water_oracle.py" \
        --fixture noahmp-water.csv --probe noahmp-water-libm.csv
echo "[7] validated"

sha256sum noahmp-water.csv noahmp-water-libm.csv
