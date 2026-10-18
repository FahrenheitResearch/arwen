#!/usr/bin/env bash
# Build the WRF v4.6.1 classic Thompson (mp_physics=8) multi-step column
# driver twice from the pristine WRF source:
#
#   BUILD_DIR/oracle  -O2 -fno-tree-vectorize (the oracle flags; the build
#                     refuses a binary that links libmvec, see
#                     tools/thompson_wrf461_oracle/build.sh for why)
#   BUILD_DIR/stock   WRF's own GNU flags from arch/configure.defaults
#                     (-O2 -ftree-vectorize -funroll-loops), libmvec allowed:
#                     what a stock WRF build on this OS computes.  Reported
#                     beside the oracle, never graded as it.
#   BUILD_DIR/repaired      the oracle flags on a REPAIRED copy of the module
#   BUILD_DIR/repaired_rcg  (repair_wrf461.py: both defects, rain-collecting-
#   BUILD_DIR/repaired_lqg  graupel slot only, graupel TAU+1 flag only).  The
#                     pinned source is never edited; the copy is written
#                     into the variant's directory.
#
# The driver (main program) is compiled without -fconvert in every variant,
# so every unit is little endian and all read the same little-endian tables.
#
# usage: ./build.sh WRF_ROOT BUILD_DIR TABLE_DIR
#   WRF_ROOT   pristine WRF tag v4.6.1 (commit d66e442f) checkout
#   TABLE_DIR  qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat (the packaged
#              gpuwm-data tables); SHA-pinned and linked into BUILD_DIR/run
set -euo pipefail

fc=${FC:-gfortran}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
stub=$(CDPATH= cd -- "$here/../thompson_wrf461_oracle" && pwd)/stub_wrf.F90

if [ "$#" -ne 3 ]; then
  echo "usage: $0 WRF_ROOT BUILD_DIR TABLE_DIR" >&2
  exit 2
fi
phys=$(realpath "$1")/phys
build=$(realpath -m "$2")
tables=$(realpath "$3")

THOMPSON_SHA=fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802
RADAR_SHA=aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688
QR_ACR_QG_SHA=89b779855847b2acdca1b40e24c5f1bd89b0c6ed105ca91a5a076d80c2437c3f
QR_ACR_QS_SHA=47350be20bd59c9f31378dd5805ce7d35fd14bebcfafb4ade56626f6eed818d7
FREEZEH2O_SHA=c235d1ce6f8750a671b2273d0e216ed3acf9a869bfd52a14676826f87aab5c02

echo "$THOMPSON_SHA  $phys/module_mp_thompson.F" | sha256sum -c -
echo "$RADAR_SHA  $phys/module_mp_radar.F" | sha256sum -c -
echo "$QR_ACR_QG_SHA  $tables/qr_acr_qg_V4.dat" | sha256sum -c -
echo "$QR_ACR_QS_SHA  $tables/qr_acr_qsV2.dat" | sha256sum -c -
echo "$FREEZEH2O_SHA  $tables/freezeH2O.dat" | sha256sum -c -

VARIANTS="oracle stock repaired repaired_rcg repaired_lqg"
declare -A FLAGS=(
  [oracle]="-O2 -fno-tree-vectorize"
  [stock]="-O2 -ftree-vectorize -funroll-loops"
  [repaired]="-O2 -fno-tree-vectorize"
  [repaired_rcg]="-O2 -fno-tree-vectorize"
  [repaired_lqg]="-O2 -fno-tree-vectorize"
)
declare -A ONLY=([repaired]="" [repaired_rcg]="--only rcg"
                 [repaired_lqg]="--only lqg")
for variant in $VARIANTS; do
  f=${FLAGS[$variant]}
  mkdir -p "$build/$variant"
  cd "$build/$variant"
  src="$phys/module_mp_thompson.F"
  case $variant in
    repaired*)
      python3 "$here/repair_wrf461.py" "$src" module_mp_thompson_repaired.F \
        ${ONLY[$variant]} | tee SOURCE.txt
      src="$build/$variant/module_mp_thompson_repaired.F" ;;
  esac
  $fc -c $f -ffree-form -ffree-line-length-none "$stub"
  $fc -c $f -ffree-form -ffree-line-length-none "$phys/module_mp_radar.F"
  $fc -c $f -cpp -DWRF_CHEM=0 -ffree-form -ffree-line-length-none \
    "$src" -o module_mp_thompson.o
  $fc -c -O2 -fno-tree-vectorize -ffree-form -ffree-line-length-none \
    "$here/run_columns_classic_steps.F90"
  $fc $f -o run_columns_classic_steps stub_wrf.o module_mp_radar.o \
    module_mp_thompson.o run_columns_classic_steps.o
  mvec=$(nm -D run_columns_classic_steps 2>/dev/null | grep -c '_ZGV' || true)
  if [ "$variant" != stock ] && [ "$mvec" != 0 ]; then
    echo "libmvec SIMD math linked into the $variant build" >&2
    exit 3
  fi
  echo "$variant libmvec_symbols = $mvec" > LIBMVEC.txt
done

mkdir -p "$build/run"
for t in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat; do
  ln -sf "$tables/$t" "$build/run/$t"
done

{
  echo "wrf_commit = d66e442fccc04111067e29274c9f9eaccc3cef28 (tag v4.6.1)"
  echo "module_mp_thompson.F = $THOMPSON_SHA"
  echo "module_mp_radar.F = $RADAR_SHA"
  echo "driver = $(sha256sum "$here/run_columns_classic_steps.F90" | cut -d' ' -f1)"
  echo "stub = $(sha256sum "$stub" | cut -d' ' -f1)"
  echo "fortran = $($fc --version | head -1)"
  echo "libc = $(ldd --version | head -1)"
  for variant in $VARIANTS; do
    echo "$variant flags = ${FLAGS[$variant]}"
    [ -f "$build/$variant/SOURCE.txt" ] && cat "$build/$variant/SOURCE.txt"
    echo "$variant binary = $(sha256sum "$build/$variant/run_columns_classic_steps" | cut -d' ' -f1)"
    cat "$build/$variant/LIBMVEC.txt"
  done
} > "$build/BUILD-RECEIPT.txt"
cat "$build/BUILD-RECEIPT.txt"
