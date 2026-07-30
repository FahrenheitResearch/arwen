#!/bin/sh
# Build and run the Noah-MP radiation-leaf oracle.
#
# Owned by the `radiation` lane.  Does not touch any shared oracle file.
#
#   ./build_radiation.sh [WRF_TREE] [OUTDIR]
#
# WRF_TREE defaults to the pinned gate tree; OUTDIR defaults to
# gpuwm/data/noahmp/oracle relative to the repository root.
#
# Must run under a glibc toolchain: the fixture's transcendental values are
# glibc's logf/expf/powf, and gpuwm/core/noahmp_libm.py transcribes exactly
# those.  Verified on gfortran 13.3.0 / glibc 2.39 (Ubuntu 24.04, WSL).
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
WRF=${1:-/home/drew/wrf-stock-v461-gate-20260721}
OUT=${2:-$REPO/gpuwm/data/noahmp/oracle}
WORK=${NOAHMP_RAD_WORK:-/tmp/noahmp-radiation-oracle}

FFLAGS="-ffree-form -ffree-line-length-none -cpp -O0 -ffp-contract=off -fno-fast-math -fno-range-check -g0"

rm -rf "$WORK"
mkdir -p "$WORK" "$OUT"

python3 "$HERE/make_public_radiation.py" \
    "$WRF/phys/module_sf_noahmplsm.F" \
    "$WORK/module_sf_noahmplsm_public.F"

cp "$WRF/phys/module_sf_gecros.F" "$WORK/"
cp "$HERE/run_radiation.F90" "$HERE/wrf_stubs_radiation.F90" "$WORK/"

cd "$WORK"
# shellcheck disable=SC2086
gfortran -c $FFLAGS module_sf_gecros.F              -o gecros.o
# shellcheck disable=SC2086
gfortran -c $FFLAGS module_sf_noahmplsm_public.F    -o noahmp.o
# shellcheck disable=SC2086
gfortran -c $FFLAGS wrf_stubs_radiation.F90         -o stubs.o
# shellcheck disable=SC2086
gfortran    $FFLAGS run_radiation.F90 noahmp.o gecros.o stubs.o -o run_radiation

./run_radiation "$OUT"

echo "toolchain: $(gfortran --version | head -1)"
echo "libc:      $(ldd --version | head -1)"
echo "flags:    $FFLAGS"
for f in "$OUT"/noahmp-radiation-*.csv; do
    echo "$(sha256sum "$f" | cut -c1-16)  $(wc -l < "$f") lines  $f"
done
