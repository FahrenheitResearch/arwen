#!/bin/sh
# Build the WRF v4.6.1 Noah-MP ENERGY bitwise oracle and emit its fixture.
#
# Usage:  build_energy.sh <wrf-tree> <workdir> [optlevel]
#
#   <wrf-tree>  pinned WRF checkout, e.g. /home/drew/wrf-stock-v461-gate-20260721
#               commit d66e442fccc04111067e29274c9f9eaccc3cef28
#   <workdir>   scratch directory (created if absent)
#   [optlevel]  "noopt" (default) builds everything at -O0.  This is the level
#               the ENERGY fixture is pinned at, for the reason recorded in
#               gpuwm/data/noahmp/oracle/PROVENANCE-soilwater.md: at WRF's own
#               FCOPTIM gfortran vectorises SOILWATER's frozen-fraction loop
#               onto glibc libmvec's _ZGVbN4v_expf, a different function from
#               scalar expf that no scalar port can reproduce.  ENERGY does not
#               call SOILWATER, but it does call EXP/LOG/** inside soil- and
#               snow-layer loops (PHASECHANGE->FRH2O, THERMOPROP->TDFCND,
#               RADIATION), so the same hazard applies and this script proves
#               it did not fire by checking `nm -u` for libmvec symbols.
#               "wrf" builds at FCOPTIM as the negative control that says how
#               far the optimiser moves the answer.
#
# Fails closed, in this order: pristine hashes, visibility patch, driver-helper
# extraction, pinned parameter tables, compile, libmvec audit, run, validate.
set -eu

WRF_TREE=$1
WORK=$2
OPTLEVEL=${3:-noopt}

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

# Pinned sources.  PRISTINE_LSM_SHA is the same constant the visibility patch
# and every other fixture in this directory check.
PRISTINE_LSM_SHA=bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282
PRISTINE_DRV_SHA=9010a757da994ed8796c63ca97da354eaf60c5c732df4ea9acad5bc62a973890
PRISTINE_GEC_SHA=ad2864562e95678a25276df82ef96395cca61c3a1bd0ab48ddfb8402902cf2f6

case "$OPTLEVEL" in
  noopt) FCOPTIM="-O0" ;;
  wrf)   FCOPTIM="-O2 -ftree-vectorize -funroll-loops" ;;
  *)     echo "unknown optlevel: $OPTLEVEL" >&2; exit 2 ;;
esac

# WRF arch/configure.defaults, "Linux x86_64, gfortran" serial block.
# No -ffast-math and no -march: gfortran may neither reassociate nor contract,
# so the FP operation sequence is fixed by the source.
FCBASE="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4"

mkdir -p "$WORK"
cd "$WORK"

echo "=== pinned source hashes ==="
sha256sum -c --status <<EOF
$PRISTINE_LSM_SHA  $WRF_TREE/phys/module_sf_noahmplsm.F
$PRISTINE_DRV_SHA  $WRF_TREE/phys/module_sf_noahmpdrv.F
$PRISTINE_GEC_SHA  $WRF_TREE/phys/module_sf_gecros.F
EOF
echo "ok"

cp "$WRF_TREE/phys/module_sf_noahmplsm.F" ./pristine.F
cp "$WRF_TREE/phys/module_sf_gecros.F"    ./module_sf_gecros.F

# ---------------------------------------------------------------------------
# 1. accessibility lift -- the only change made to the physics source
# ---------------------------------------------------------------------------
python3 "$HERE/visibility_patch_leaves.py" ./pristine.F \
        --out ./module_sf_noahmplsm_public.F \
        --check \
        --require-symbol ENERGY \
        --require-symbol THERMOPROP \
        --require-symbol RADIATION \
        --require-symbol VEGE_FLUX \
        --require-symbol BARE_FLUX \
        --require-symbol TSNOSOI \
        --require-symbol PHASECHANGE \
        --require-symbol ATM \
        --require-symbol PHENOLOGY \
        --require-symbol PRECIP_HEAT

# ---------------------------------------------------------------------------
# 2. driver-side services, lifted verbatim rather than transcribed
# ---------------------------------------------------------------------------
python3 "$HERE/extract_drv_helpers.py" "$WRF_TREE/phys/module_sf_noahmpdrv.F" \
        --out ./noahmp_drv_helpers.F90 \
        --expect-sha "$PRISTINE_DRV_SHA"

cat > wrf_stubs.f90 <<'STUB'
! Minimal stand-ins for the WRF frame routines the physics module calls.
! ENERGY reaches wrf_error_fatal only on the FIRE <= 0 abort at
! module_sf_noahmplsm.F:2323-2329, which no fixture case takes; they exist so
! the module links.
subroutine wrf_message(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') trim(msg)
end subroutine wrf_message

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

subroutine wrf_debug(level, msg)
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug
STUB

# ---------------------------------------------------------------------------
# 3. the byte-pinned parameter tables, taken from gpuwm rather than from the
#    WRF checkout's run/ copies, so the oracle and gpuwm/core/noahmp.py parse
#    identical bytes
# ---------------------------------------------------------------------------
cp "$REPO/gpuwm/data/noahmp/MPTABLE.TBL" .
cp "$REPO/gpuwm/data/noahmp/SOILPARM.TBL" .
cp "$REPO/gpuwm/data/noahmp/GENPARM.TBL" .
sha256sum -c --status <<'EOF'
7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d  MPTABLE.TBL
1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f  SOILPARM.TBL
9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2  GENPARM.TBL
EOF

# ---------------------------------------------------------------------------
# 4. compile
# ---------------------------------------------------------------------------
gfortran $FCOPTIM $FCBASE -c module_sf_gecros.F            -o gecros.o
gfortran $FCOPTIM $FCBASE -c module_sf_noahmplsm_public.F  -o noahmp.o
gfortran $FCOPTIM $FCBASE -c wrf_stubs.f90                 -o stubs.o
gfortran $FCOPTIM $FCBASE -c noahmp_drv_helpers.F90        -o drvhelp.o
gfortran $FCOPTIM $FCBASE -c "$HERE/run_energy.F90"        -o run_energy.o
gfortran -o run_energy run_energy.o noahmp.o gecros.o drvhelp.o stubs.o

# ---------------------------------------------------------------------------
# 5. libmvec audit (the trap recorded in PROVENANCE-soilwater.md)
# ---------------------------------------------------------------------------
echo "=== libmvec audit ($OPTLEVEL) ==="
if nm -u noahmp.o | grep -E '_ZGV[a-z][0-9]+' ; then
  if [ "$OPTLEVEL" = noopt ]; then
    echo "FATAL: vectorised libm calls in a -O0 build" >&2
    exit 3
  fi
  echo "(expected at FCOPTIM; the fixture is pinned at -O0)"
else
  echo "no libmvec references"
fi

# ---------------------------------------------------------------------------
# 6. run and validate
# ---------------------------------------------------------------------------
./run_energy noahmp-energy.csv
echo "=== fixture rows: $(( $(wc -l < noahmp-energy.csv) - 1 )) ==="

python3 "$HERE/validate_energy_oracle.py" noahmp-energy.csv \
        --sflx "$REPO/gpuwm/data/noahmp/oracle/noahmp-sflx.csv"

sha256sum "$WRF_TREE/phys/module_sf_noahmplsm.F" \
          "$WRF_TREE/phys/module_sf_noahmpdrv.F" \
          "$WRF_TREE/phys/module_sf_gecros.F" \
          module_sf_noahmplsm_public.F noahmp_drv_helpers.F90 \
          "$HERE/run_energy.F90" \
          MPTABLE.TBL SOILPARM.TBL GENPARM.TBL \
          noahmp-energy.csv > energy-sha256sums.txt
{
  echo "optlevel   = $OPTLEVEL"
  echo "FCOPTIM    = $FCOPTIM"
  echo "FCBASE     = $FCBASE"
  gfortran --version | head -1
} > energy-provenance.txt
cat energy-sha256sums.txt energy-provenance.txt
