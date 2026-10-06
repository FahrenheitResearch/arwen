#!/usr/bin/env bash
# Build the HRRR v4.1.21 fork Thompson oracle (the WRFV3.9 generation of
# module_mp_thompson.F) on a Linux CPU with gfortran, generate the fork's
# lookup tables from the fork's own thompson_init, and link the batch column
# driver.
#
# usage:
#   ./build.sh FORK_PHYS_DIR BUILD_DIR CCN_ACTIVATE.BIN
#
# FORK_PHYS_DIR holds module_mp_thompson.F and module_mp_radar.F fetched
#   from https://raw.githubusercontent.com/NOAA-EMC/HRRR/v4.1.21/sorc/
#   hrrr_wrfarw.fd/WRFV3.9/phys/ ; their SHA-256s are pinned below.
# BUILD_DIR must not already hold table outputs (the build refuses to reuse
#   them, so every table it reports was computed by this build).
#
# Flags follow tools/thompson_wrf461_oracle/build.sh: -O2
# -fno-tree-vectorize, and the build refuses a binary that links libmvec
# (that file explains why the flag decides bit identity of the tables).
set -euo pipefail

fc=${FC:-gfortran}
opt_flags=${OPT_FLAGS:--O2 -fno-tree-vectorize}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ "$#" -ne 3 ]; then
  echo "usage: $0 FORK_PHYS_DIR BUILD_DIR CCN_ACTIVATE.BIN" >&2
  exit 2
fi
phys=$(realpath "$1")
build=$(realpath -m "$2")
ccn=$(realpath "$3")

THOMPSON_SHA=4d60011188443eb432294f7693beb64bdbc8f812541a15c7800013060c877283
RADAR_SHA=08329c87604b234efab53f7986163a4f6050f58823da9099cb464163c1920f08
CCN_SHA=f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd
echo "$THOMPSON_SHA  $phys/module_mp_thompson.F" | sha256sum -c -
echo "$RADAR_SHA  $phys/module_mp_radar.F" | sha256sum -c -
echo "$CCN_SHA  $ccn" | sha256sum -c -

mkdir -p "$build"
for output in qr_acr_qg.dat qr_acr_qs.dat freezeH2O.dat thompson_aux_tables.dat; do
  if [ -e "$build/$output" ]; then
    echo "refusing to reuse existing oracle output: $build/$output" >&2
    exit 2
  fi
done
cd "$build"

$fc -c $opt_flags -ffree-form -ffree-line-length-none "$here/stub_wrf.F90"
$fc -c $opt_flags -ffree-form -ffree-line-length-none "$phys/module_mp_radar.F"
$fc -c $opt_flags -cpp -DWRF_CHEM=0 -ffree-form -ffree-line-length-none \
  "$phys/module_mp_thompson.F" -o module_mp_thompson.o
for prog in generate_tables run_columns_fork; do
  src="$here/$prog.F90"
  $fc -c $opt_flags -ffree-form -ffree-line-length-none "$src"
  $fc $opt_flags -o "$prog" stub_wrf.o module_mp_radar.o \
    module_mp_thompson.o "$prog.o"
  if nm -D "$prog" 2>/dev/null | grep -q '_ZGV'; then
    echo "libmvec SIMD math linked into $prog; oracle is not invariant" >&2
    exit 3
  fi
done

t0=$(date +%s)
./generate_tables | tee generate.log
echo "table generation wall seconds: $(( $(date +%s) - t0 ))" | tee -a generate.log
cp -- "$ccn" CCN_ACTIVATE.BIN

sha_of() { sha256sum "$1" | cut -d' ' -f1; }
{
  echo "# HRRR v4.1.21 fork Thompson oracle receipt (tools/thompson_fork_oracle/build.sh)"
  echo
  echo "[fork_source]"
  echo "origin = https://github.com/NOAA-EMC/HRRR tag v4.1.21"
  echo "path = sorc/hrrr_wrfarw.fd/WRFV3.9/phys"
  echo "module_mp_thompson.F = $THOMPSON_SHA"
  echo "module_mp_radar.F = $RADAR_SHA"
  echo "CCN_ACTIVATE.BIN = $CCN_SHA"
  echo
  echo "[harness_source]"
  for f in build.sh stub_wrf.F90 generate_tables.F90 run_columns_fork.F90; do
    echo "$f = $(sha_of "$here/$f")"
  done
  echo
  echo "[toolchain]"
  echo "fortran = $($fc --version | head -1)"
  echo "libc = $(ldd --version | head -1)"
  echo "uname = $(uname -srm)"
  echo "opt_flags = $opt_flags"
  echo "libmvec_symbols = $(nm -D run_columns_fork | grep -c '_ZGV' || true)"
  echo
  echo "[binaries]"
  for b in generate_tables run_columns_fork; do echo "$b = $(sha_of "$b")"; done
  echo
  echo "[tables]"
  for t in qr_acr_qg.dat qr_acr_qs.dat freezeH2O.dat thompson_aux_tables.dat; do
    echo "$t = $(stat -c '%s' "$t") $(sha_of "$t")"
  done
} | tee PROVENANCE.txt
