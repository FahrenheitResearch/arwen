#!/usr/bin/env bash
# Build the WRF 4.6.1 P3 column oracle from a pristine WRF source tree.
#
#   bash tools/p3_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
#
# module_mp_p3.F is compiled UNMODIFIED (sha256 pinned below; the same bytes
# ship in WRF v4.6.1, v4.7.1 and v4.8.0).  The driver is the existing
# evidence/p3-fortran-oracle-20260828/harness/p3_ref_driver.f90, which calls
# mp_p3_wrapper_wrf with the mp_physics=50 call shape.
#
# Arms:
#   A  -O0 -ffp-contract=off -fno-unsafe-math-optimizations
#      statement order as written, against this box's C library (glibc
#      2.39, whose tgammaf and log10f round incorrectly).  Reported.
#   CR arm A with tgammaf/log10f correctly rounded (crlibm_shim.c), which
#      is what WRF computes on a C library that rounds them correctly.
#      THE ORACLE for the 0 ULP claim; byte-identical to arm A built on
#      glibc 2.43 for all thirteen p3-fortref fixtures.
#   B  WRF 4.6.1's own GNU FCOPTIM: -O2 -ftree-vectorize -funroll-loops
#      (what a stock wrf.exe on this box would compute; vectorised loops
#      may call libmvec).  Reported, not the claim.
set -euo pipefail
WRF=${1:?usage: build.sh WRF_SOURCE_ROOT BUILD_DIR}
OUT=${2:?usage: build.sh WRF_SOURCE_ROOT BUILD_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
SRC="$WRF/phys/module_mp_p3.F"
DRV="$TREE/evidence/p3-fortran-oracle-20260828/harness/p3_ref_driver.f90"
TAB="$WRF/run/p3_lookupTable_1.dat-v5.4_2momI"

EXPECT_SRC=716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6
# WRF's run/ copy is LF; WOOF's packaged copy is the same text with CRLF
# (gpuwm/data/p3/tables/MANIFEST.sha256 pins 5cb47f2a...).  Both hash to
# be1ab6fb... once carriage returns are removed.
EXPECT_TAB=be1ab6fb03481e376e47c6c79d808af5d8ab069f2b242931e9c54801bad4ae84

got=$(sha256sum "$SRC" | cut -d' ' -f1)
[ "$got" = "$EXPECT_SRC" ] || { echo "FATAL: $SRC sha256 $got != $EXPECT_SRC" >&2; exit 1; }
got=$(tr -d '\r' < "$TAB" | sha256sum | cut -d' ' -f1)
[ "$got" = "$EXPECT_TAB" ] || { echo "FATAL: $TAB (LF) sha256 $got != $EXPECT_TAB" >&2; exit 1; }
woof_tab="$TREE/gpuwm/data/p3/tables/p3_lookupTable_1.dat-v5.4_2momI"
got=$(tr -d '\r' < "$woof_tab" | sha256sum | cut -d' ' -f1)
[ "$got" = "$EXPECT_TAB" ] || { echo "FATAL: WOOF table text differs from WRF's" >&2; exit 1; }

mkdir -p "$OUT/tables"
cp "$TAB" "$OUT/tables/"
FREE="-cpp -ffree-form -ffree-line-length-none"
build () {
  local arm=$1; shift
  local d="$OUT/build/$arm"
  rm -rf "$d"; mkdir -p "$d"
  ( cd "$d"
    gfortran $FREE "$@" -c "$SRC" -o module_mp_p3.o
    gfortran $FREE "$@" -c "$DRV" -o p3_ref_driver.o
    gfortran "$@" module_mp_p3.o p3_ref_driver.o -o p3_ref_driver )
  echo "built arm $arm: $*"
}
build A -O0 -ffp-contract=off -fno-unsafe-math-optimizations
build B -O2 -ftree-vectorize -funroll-loops
# Arm CR: arm A's flags and objects, with crlibm_shim.o linked ahead of libm
# so tgammaf/log10f are correctly rounded (current-glibc behaviour).
d="$OUT/build/CR"; rm -rf "$d"; mkdir -p "$d"
gcc -O2 -c "$HERE/crlibm_shim.c" -o "$d/crlibm_shim.o"
gfortran "$OUT/build/A/module_mp_p3.o" "$OUT/build/A/p3_ref_driver.o" "$d/crlibm_shim.o" -lquadmath -o "$d/p3_ref_driver"
echo "built arm CR: arm A objects + crlibm_shim.o"
{
  gfortran --version | sed -n 1p
  ldd --version | sed -n 1p
  echo "module_mp_p3.F $EXPECT_SRC"
  echo "driver $(sha256sum "$DRV" | cut -d' ' -f1)"
  echo "fma instructions in arm A: $(objdump -d "$OUT/build/A/p3_ref_driver" | grep -c 'vfmadd\|vfmsub\|vfnmadd' || true)"
  echo "fma instructions in arm B: $(objdump -d "$OUT/build/B/p3_ref_driver" | grep -c 'vfmadd\|vfmsub\|vfnmadd' || true)"
  echo "libmvec calls in arm B: $(objdump -d "$OUT/build/B/p3_ref_driver" | grep -c 'call.*_ZGV' || true)"
} | tee "$OUT/build/TOOLCHAIN.txt"
