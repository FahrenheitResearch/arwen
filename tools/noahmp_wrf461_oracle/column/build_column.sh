#!/bin/sh
# Build the WRF v4.6.1 Noah-MP column oracle (run_column) against the
# pristine pinned sources, CPU only.
#
# Usage:  build_column.sh <wrf-tree> <workdir> [optlevel]
#
#   <wrf-tree>  pristine WRF v4.6.1 checkout
#               (commit d66e442fccc04111067e29274c9f9eaccc3cef28, submodule
#               phys/noahmp at 848f54a)
#   <workdir>   build directory (created if absent)
#   [optlevel]  "noopt" (default): -O0, the fixture.  Scalar libm only, no
#                      vector math library, no FP contraction on x86-64.
#               "wrf": WRF's own gfortran FCOPTIM
#                      (-O2 -ftree-vectorize -funroll-loops), the stock build's
#                      arithmetic.  A recorded comparison, never the gate:
#                      vectorised loops may call libmvec, which rounds
#                      differently from scalar libm.
set -eu

WRF_TREE=$1
WORK=$2
OPTLEVEL=${3:-noopt}

HERE=$(cd "$(dirname "$0")" && pwd)
ORACLE=$(cd "$HERE/.." && pwd)

PRISTINE_LSM_SHA=bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282
PRISTINE_DRV_SHA=9010a757da994ed8796c63ca97da354eaf60c5c732df4ea9acad5bc62a973890
PRISTINE_GLA_SHA=bf94f3522c3b9c2c9cfbb34fa7e485ff58519106db434520968793409a520579

case "$OPTLEVEL" in
  noopt) FCOPTIM="-O0" ;;
  wrf)   FCOPTIM="-O2 -ftree-vectorize -funroll-loops" ;;
  *)     echo "unknown optlevel: $OPTLEVEL" >&2; exit 2 ;;
esac
FCBASE="-w -cpp -ffree-form -ffree-line-length-none"

NMP="$WRF_TREE/phys/noahmp"
LSM="$NMP/src/module_sf_noahmplsm.F"
GLA="$NMP/src/module_sf_noahmp_glacier.F"
GW="$NMP/src/module_sf_noahmp_groundwater.F"
DRV="$NMP/drivers/wrf/module_sf_noahmpdrv.F"

check() {
  got=$(sha256sum "$1" | cut -d' ' -f1)
  if [ "$got" != "$2" ]; then
    echo "$1 is not the pinned file (got $got, want $2)" >&2
    exit 3
  fi
  echo "[1] pristine $(basename "$1") sha256 $got"
}
check "$LSM" "$PRISTINE_LSM_SHA"
check "$DRV" "$PRISTINE_DRV_SHA"
check "$GLA" "$PRISTINE_GLA_SHA"

mkdir -p "$WORK"
cd "$WORK"
cp "$WRF_TREE/share/module_model_constants.F" .
cp "$WRF_TREE/phys/module_sf_gecros.F" .
cp "$LSM" "$GLA" "$GW" "$DRV" .

gfortran -c $FCBASE $FCOPTIM "$ORACLE/stub_wrf.F90"
gfortran -c $FCBASE $FCOPTIM module_model_constants.F
gfortran -c $FCBASE $FCOPTIM module_sf_gecros.F
gfortran -c $FCBASE $FCOPTIM -I. module_sf_noahmplsm.F
gfortran -c $FCBASE $FCOPTIM -I. module_sf_noahmp_glacier.F
gfortran -c $FCBASE $FCOPTIM -I. module_sf_noahmp_groundwater.F
# EM_CORE=0 compiles out the DM_PARALLEL halo exchange in GROUNDWATER_INIT;
# -fallow-argument-mismatch matches WRF's own build of this file.
gfortran -c $FCBASE $FCOPTIM -DEM_CORE=0 -fallow-argument-mismatch \
         -I. module_sf_noahmpdrv.F
gfortran -c $FCBASE $FCOPTIM -I. "$HERE/run_column.F90"
gfortran -o run_column run_column.o module_sf_noahmpdrv.o \
         module_sf_noahmp_groundwater.o module_sf_noahmp_glacier.o \
         module_sf_noahmplsm.o module_sf_gecros.o module_model_constants.o \
         stub_wrf.o -lm
echo "[2] built $WORK/run_column (optlevel=$OPTLEVEL, FCOPTIM='$FCOPTIM')"

# WRF 4.6.1's own tables (phys/noahmp/parameters/MPTABLE.TBL, run/SOILPARM.TBL,
# run/GENPARM.TBL), which are byte-identical to WOOF's packaged copies.
cp "$NMP/parameters/MPTABLE.TBL" "$WRF_TREE/run/SOILPARM.TBL" \
   "$WRF_TREE/run/GENPARM.TBL" .
sha256sum -c --status <<'EOF'
7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d  MPTABLE.TBL
1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f  SOILPARM.TBL
9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2  GENPARM.TBL
EOF
echo "[3] tables staged (sha256 equal to gpuwm/data/noahmp)"

# Which compiled routines call a vector math entry point.  At -O0 there must
# be none; at WRF's FCOPTIM they are recorded, because each one is a place
# the stock build can round differently from scalar libm.
for obj in module_sf_noahmplsm.o module_sf_noahmp_glacier.o module_sf_noahmpdrv.o; do
  VEC=$(objdump -dr --no-show-raw-insn "$obj" \
        | awk '/^[0-9a-f]+ <.*>:/{fn=$2} /_ZGV/{print fn" "$NF}' | sort -u)
  if [ -n "$VEC" ]; then
    echo "[4] $obj vector-math calls (function, symbol):"
    echo "$VEC" | sed 's/^/      /'
    if [ "$OPTLEVEL" = noopt ]; then
      echo "unexpected libmvec reference at -O0" >&2
      exit 6
    fi
  else
    echo "[4] $obj: no vector-math call"
  fi
done
FMA=$(objdump -d run_column | grep -c -E 'vfmadd|vfmsub|vfnmadd|vfnmsub' || true)
echo "[5] FMA instructions in run_column: $FMA"
gfortran --version | head -1
