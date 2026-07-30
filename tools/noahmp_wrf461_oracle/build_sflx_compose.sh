#!/bin/sh
# Build the WRF v4.6.1 Noah-MP ERROR bitwise oracle -- the balance check
# NOAHMP_SFLX runs at module_sf_noahmplsm.F:1049-1058.
#
# Usage:  build_sflx_compose.sh <wrf-tree> <workdir> [optlevel]
#
#   <wrf-tree>  pinned WRF checkout, e.g. /home/drew/wrf-stock-v461-gate-20260721
#               commit d66e442fccc04111067e29274c9f9eaccc3cef28
#   <workdir>   scratch directory (created if absent)
#   [optlevel]  "noopt" (default) builds at -O0 and is the fixture
#               "nocontract" -O0 with -ffp-contract=off
#               "snan" -O0 with -finit-real=snan -finit-integer=-2147483647
#                      -finit-logical=false
#               "wrf" reproduces WRF's own gfortran FCOPTIM
#
# All three non-default levels are negative controls and must produce a
# byte-identical CSV.  "snan" is the load-bearing one here: ERROR declares
# ERRWAT INTENT(OUT) and leaves it unassigned when IST == 1 and
# calculate_soil is .false. (case err_no_soil_substep), so the fixture must
# record whatever the caller's storage held and the port must be shown not to
# depend on it.  run_sflx_compose.F90 poisons ERRWAT before every call for the
# same reason.
#
# ERROR evaluates no transcendental at any option setting -- its body is nine
# add/multiply statements -- so the libmvec hazard recorded in
# gpuwm/data/noahmp/oracle/PROVENANCE-soilwater.md cannot arise.  Stage 5
# checks that claim rather than asserting it.
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
  *)          echo "unknown optlevel: $OPTLEVEL" >&2; exit 2 ;;
esac

# WRF arch/configure.defaults, "Linux x86_64, gfortran" serial block.
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

cp "$WRF_TREE/phys/module_sf_noahmplsm.F" ./pristine.F
cp "$WRF_TREE/phys/module_sf_gecros.F"    ./module_sf_gecros.F

# Stage 2: the checker's own negative controls, then the lift itself.
python3 "$HERE/visibility_patch_leaves.py" --self-test
echo "[2] visibility_patch_leaves.py self-test passed"

python3 "$HERE/visibility_patch_leaves.py" ./pristine.F \
        --out ./module_sf_noahmplsm_public.F \
        --check \
        --require-symbol ERROR
echo "[3] accessibility lift applied; diff is exactly private:: -> public::"

cat > wrf_stubs.f90 <<'STUB'
! Service-only stubs for the standalone Noah-MP ERROR oracle harness.
!
! ERROR reaches wrf_message on every diagnostic line of its three failure
! blocks and wrf_error_fatal at the end of each.  wrf_error_fatal must not
! return: cases 90-92 of run_sflx_compose.F90 exist precisely to reach it, and
! the build below requires each of them to terminate with status 1.
subroutine wrf_error_fatal(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') 'noahmp sflx oracle: WRF fatal: '//trim(msg)
  call exit(1)
end subroutine wrf_error_fatal

subroutine wrf_error_fatal3(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') 'noahmp sflx oracle: WRF fatal: '//trim(msg)
  call exit(1)
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
gfortran -c $FCBASE $FCOPTIM $EXTRA "$HERE/run_sflx_compose.F90" -o run_sflx_compose.o
gfortran -o run_sflx_compose run_sflx_compose.o module_sf_noahmplsm.o \
         module_sf_gecros.o wrf_stubs.o -lm
echo "[4] built $WORK/run_sflx_compose  (optlevel=$OPTLEVEL, FCOPTIM='$FCOPTIM $EXTRA')"

# Stage 5: ERROR calls no libm function at all.  Prove it on the harness object
# rather than on the whole module, which carries every other Noah-MP routine.
UNDEF=$(nm -u run_sflx_compose.o | awk '{print $NF}' | sort -u)
BAD=$(printf '%s\n' "$UNDEF" | grep -E '^(_ZGV|expf$|powf$|logf$|log10f$|tanhf$|sqrtf$)' || true)
if [ -n "$BAD" ]; then
  echo "the ERROR harness references a libm symbol; ERROR has no transcendental:" >&2
  printf '%s\n' "$BAD" | sed 's/^/  /' >&2
  exit 4
fi
echo "[5] nm -u: the ERROR harness references no libm symbol"

./run_sflx_compose noahmp-sflx-error.csv
echo "[6] fixture: $(($(wc -l < noahmp-sflx-error.csv) - 1)) data rows"

# Stage 7: the three abort gates.  Each must terminate; a gate that cannot be
# shown to fire is not a gate.  Run one per process, since wrf_error_fatal
# ends the process by construction.
for spec in '90:Stop in Noah-MP' \
            '91:Energy budget problem in NOAHMP LSM' \
            '92:Water budget problem in NOAHMP LSM'; do
  case_id=${spec%%:*}
  want=${spec#*:}
  if ./run_sflx_compose /dev/null "$case_id" >abort.out 2>abort.err; then
    echo "case $case_id returned instead of aborting" >&2
    exit 5
  fi
  if ! grep -qF "$want" abort.err; then
    echo "case $case_id aborted with the wrong message:" >&2
    sed 's/^/  /' abort.err >&2
    echo "  expected to contain: $want" >&2
    exit 6
  fi
  echo "[7] case $case_id aborted: $want"
done

python3 "$HERE/validate_sflx_error_oracle.py" noahmp-sflx-error.csv
echo "[8] validated"

sha256sum noahmp-sflx-error.csv
{
  echo "optlevel = $OPTLEVEL"
  echo "FCOPTIM  = $FCOPTIM $EXTRA"
  echo "FCBASE   = $FCBASE"
  gfortran --version | head -1
} > sflx-error-provenance.txt
cat sflx-error-provenance.txt
