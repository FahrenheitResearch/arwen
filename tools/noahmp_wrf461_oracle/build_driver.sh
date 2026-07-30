#!/bin/sh
# Build the WRF v4.6.1 Noah-MP driver-side cold-start bitwise oracle:
# module_sf_noahmpdrv::SNOW_INIT and module_sf_noahmpdrv::NOAHMP_INIT.
#
# Usage:  build_driver.sh <wrf-tree> <workdir> [optlevel]
#
#   <wrf-tree>  pinned WRF checkout, e.g. /home/drew/wrf-stock-v461-gate-20260721
#               commit d66e442fccc04111067e29274c9f9eaccc3cef28
#   <workdir>   scratch directory (created if absent)
#   [optlevel]  "noopt" (default) builds at -O0 and is the fixture
#               "nocontract" -O0 with -ffp-contract=off
#               "snan" -O0 with -finit-real=snan -finit-integer=-2147483647
#                      -finit-logical=false
#               "wrf" reproduces WRF's own gfortran FCOPTIM (recorded
#                      divergence, never a gate)
#
# No visibility patch
# -------------------
# phys/module_sf_noahmpdrv.F contains no accessibility statement at all -- no
# `private`, no `public` -- so Fortran's default accessibility makes every one
# of its module procedures public and this harness calls SNOW_INIT and
# NOAHMP_INIT directly against the byte-unmodified source.  Stage [2] asserts
# that absence instead of assuming it, and stage [3] asserts the module is
# linked against the **pristine** module_sf_noahmplsm.F, not the
# leaf-visibility-patched one (which physically cannot compile against this
# driver -- see ADDING_A_LEAF.md section 9).
#
# Why -O0 is the fixture
# ----------------------
# NOAHMP_INIT's only transcendental is the supercooled-liquid initial guess at
# 2095-2096, `(...)**(-1/BEXP)`, a REAL**REAL that lowers to a call.  Stage [6]
# runs `nm -u` and fails closed if the object references any glibc libmvec
# symbol (`_ZGVbN*`), which is the trap that forced the soilwater fixture to
# -O0 (ADDING_A_LEAF.md, and gpuwm/data/noahmp/oracle/PROVENANCE-soilwater.md).
#
# "nocontract" and "snan" are negative controls and must produce a
# byte-identical CSV: the first pins that no emitted value depends on compiler
# FP contraction, the second that none reads an uninitialised local.  The
# latter is load-bearing here -- SNOW_INIT's DZSNO is zeroed only on the
# `SNODEP < 0.025` branch (2383) and NOAHMP_INIT leaves `cropcat` unwritten on
# the vegetated branch when iopt_crop=0.
set -eu

WRF_TREE=$1
WORK=$2
OPTLEVEL=${3:-noopt}

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

PRISTINE_LSM_SHA=bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282
PRISTINE_DRV_SHA=9010a757da994ed8796c63ca97da354eaf60c5c732df4ea9acad5bc62a973890

EXTRA=""
case "$OPTLEVEL" in
  noopt)      FCOPTIM="-O0" ;;
  nocontract) FCOPTIM="-O0 -ffp-contract=off" ;;
  snan)       FCOPTIM="-O0"
              EXTRA="-finit-real=snan -finit-integer=-2147483647 -finit-logical=false" ;;
  wrf)        FCOPTIM="-O2 -ftree-vectorize -funroll-loops" ;;
  *)          echo "unknown optlevel: $OPTLEVEL" >&2; exit 2 ;;
esac

# WRF arch/configure.defaults, "Linux x86_64, gfortran" block.
FCBASE="-w -cpp -ffree-form -ffree-line-length-none"

mkdir -p "$WORK"
cd "$WORK"

# --- [1] the pinned identity of both physics sources, before anything else.
GOT_LSM=$(sha256sum "$WRF_TREE/phys/module_sf_noahmplsm.F" | cut -d' ' -f1)
GOT_DRV=$(sha256sum "$WRF_TREE/phys/module_sf_noahmpdrv.F" | cut -d' ' -f1)
if [ "$GOT_LSM" != "$PRISTINE_LSM_SHA" ]; then
  echo "module_sf_noahmplsm.F is not the pinned file" >&2
  echo "  expected $PRISTINE_LSM_SHA" >&2
  echo "  got      $GOT_LSM" >&2
  exit 3
fi
if [ "$GOT_DRV" != "$PRISTINE_DRV_SHA" ]; then
  echo "module_sf_noahmpdrv.F is not the pinned file" >&2
  echo "  expected $PRISTINE_DRV_SHA" >&2
  echo "  got      $GOT_DRV" >&2
  exit 3
fi
echo "[1] pristine module_sf_noahmplsm.F sha256 $GOT_LSM"
echo "[1] pristine module_sf_noahmpdrv.F sha256 $GOT_DRV"

# --- [2] the driver needs no visibility lift: prove it declares none.
NPRIV=$(grep -c -i '^ *private' "$WRF_TREE/phys/module_sf_noahmpdrv.F" || true)
NPUB=$(grep -c -i '^ *public' "$WRF_TREE/phys/module_sf_noahmpdrv.F" || true)
if [ "$NPRIV" != "0" ] || [ "$NPUB" != "0" ]; then
  echo "module_sf_noahmpdrv.F now carries accessibility statements" >&2
  echo "  private: $NPRIV  public: $NPUB  (both must be 0)" >&2
  exit 4
fi
echo "[2] module_sf_noahmpdrv.F declares 0 private and 0 public statements;"
echo "    SNOW_INIT and NOAHMP_INIT are public by default accessibility"

# --- [3] refuse to build if a leaf-visibility patch is anywhere in the tree.
if grep -q -i '^ *public *:: *ATM' "$WRF_TREE/phys/module_sf_noahmplsm.F"; then
  echo "module_sf_noahmplsm.F carries the leaf visibility lift; this harness" >&2
  echo "must link the pristine module (ADDING_A_LEAF.md section 9)" >&2
  exit 5
fi
echo "[3] linking the pristine module_sf_noahmplsm.F"

cp "$WRF_TREE/share/module_model_constants.F" .
cp "$WRF_TREE/phys/module_sf_gecros.F"        .
cp "$WRF_TREE/phys/module_sf_noahmplsm.F"     .
cp "$WRF_TREE/phys/module_sf_noahmp_glacier.F" .
cp "$WRF_TREE/phys/module_sf_noahmp_groundwater.F" .
cp "$WRF_TREE/phys/module_sf_noahmpdrv.F"     .

# --- [4] compile, byte-for-byte as the sources ship.
gfortran -c $FCBASE $FCOPTIM $EXTRA "$HERE/stub_wrf.F90"
gfortran -c $FCBASE $FCOPTIM $EXTRA module_model_constants.F
gfortran -c $FCBASE $FCOPTIM $EXTRA module_sf_gecros.F
gfortran -c $FCBASE $FCOPTIM $EXTRA -I. module_sf_noahmplsm.F
gfortran -c $FCBASE $FCOPTIM $EXTRA -I. module_sf_noahmp_glacier.F
gfortran -c $FCBASE $FCOPTIM $EXTRA -I. module_sf_noahmp_groundwater.F
# EM_CORE=0 compiles out the DM_PARALLEL halo exchange in GROUNDWATER_INIT;
# -fallow-argument-mismatch matches WRF's own build of this file.
gfortran -c $FCBASE $FCOPTIM $EXTRA -DEM_CORE=0 -fallow-argument-mismatch \
         -I. module_sf_noahmpdrv.F
gfortran -c $FCBASE $FCOPTIM $EXTRA -I. "$HERE/run_driver.F90"
gfortran -o run_driver run_driver.o module_sf_noahmpdrv.o \
         module_sf_noahmp_groundwater.o module_sf_noahmp_glacier.o \
         module_sf_noahmplsm.o module_sf_gecros.o module_model_constants.o \
         stub_wrf.o -lm
echo "[4] built $WORK/run_driver  (optlevel=$OPTLEVEL, FCOPTIM='$FCOPTIM $EXTRA')"

# --- [5] the byte-pinned parameter assets, not the WRF checkout's run/ copies,
#         so the oracle and gpuwm/core/noahmp.py parse identical bytes.
cp "$REPO/gpuwm/data/noahmp/MPTABLE.TBL" .
cp "$REPO/gpuwm/data/noahmp/SOILPARM.TBL" .
cp "$REPO/gpuwm/data/noahmp/GENPARM.TBL" .
sha256sum -c --status <<'EOF'
7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d  MPTABLE.TBL
1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f  SOILPARM.TBL
9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2  GENPARM.TBL
EOF
echo "[5] byte-pinned MPTABLE/SOILPARM/GENPARM staged"

# --- [6] ADDING_A_LEAF.md trap 3: no libmvec substitution on the fixture path.
#
# The interesting question is not "does the object reference libmvec" but "does
# either pinned routine reference it".  gfortran emits a relocation against the
# vector symbol inside the function that was vectorised, so objdump -dr locates
# it exactly.  At -O0 there is none at all; at WRF's own FCOPTIM there are
# three, and all three live in PEDOTRANSFER_SR2006, which module_sf_noahmpdrv.F
# reaches only from `IF(iopt_soil == 3)` at 983 and which neither SNOW_INIT nor
# NOAHMP_INIT can call.  Recording that is the point of the `wrf` control.
VEC_FUNCS=$(objdump -dr --no-show-raw-insn module_sf_noahmpdrv.o \
            | awk '/^[0-9a-f]+ <.*>:/{fn=$2} /_ZGV/{print fn}' | sort -u)
if [ -n "$VEC_FUNCS" ]; then
  echo "[6] libmvec references in module_sf_noahmpdrv.o, by owning function:"
  echo "$VEC_FUNCS" | sed 's/^/      /'
  if echo "$VEC_FUNCS" | grep -q -i -E 'snow_init|noahmp_init'; then
    echo "a pinned routine was vectorised into libmvec; its FP32 results are" >&2
    echo "not reproducible by any port that calls scalar libm" >&2
    exit 6
  fi
  if [ "$OPTLEVEL" != wrf ]; then
    echo "unexpected libmvec reference at optlevel=$OPTLEVEL" >&2
    exit 6
  fi
  echo "    none of them is SNOW_INIT or NOAHMP_INIT"
else
  echo "[6] nm -u module_sf_noahmpdrv.o: no libmvec reference at all"
fi
echo "    scalar libm calls: $(nm -u module_sf_noahmpdrv.o | grep -E 'powf|expf|logf|sqrtf' | tr -s ' ' | paste -sd' ' -)"

# --- [7] run and validate.
./run_driver noahmp-driver.csv
echo "[7] fixture: $(($(wc -l < noahmp-driver.csv) - 1)) data rows"

python3 "$HERE/validate_driver_oracle.py" --fixture noahmp-driver.csv
echo "[8] validated"

sha256sum noahmp-driver.csv
gfortran --version | head -1
