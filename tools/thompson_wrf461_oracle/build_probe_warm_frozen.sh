#!/usr/bin/env bash
# Build and run probe_warm_frozen_aero.F90 -- the WP-07 oracle for WRF's
# always-run frozen collection block evaluated ABOVE FREEZING.
#
# This script deliberately does NOT rebuild module_mp_thompson.o and does NOT
# regenerate a single .dat.  It REUSES the build directory build_aero.sh
# already produced, so the object this probe links is byte-for-byte the one
# that produced the 19 committed aerosol column fixtures.  Rebuilding here
# would put the four classic .dat SHA-256 pins back in play for no reason.
#
# usage:
#   ./build_aero.sh /path/to/WRF-v4.6.1 /some/build-dir /path/CCN_ACTIVATE.BIN
#   ./build_probe_warm_frozen.sh /some/build-dir [/output/dir]
#
# The output directory defaults to <build-dir>/column-oracle-aero.  One CSV
# is written: wp07-warm-frozen-rates.csv.

set -euo pipefail

fc=${FC:-gfortran}

# -fno-tree-vectorize is LOAD-BEARING; see build_aero.sh for the full
# statement of the property and the measurement behind it.  In one line:
# -O2 implies -ftree-vectorize from GCC 12 on, a vectorised exp/pow/log loop
# links glibc's libmvec SIMD entry points instead of the scalar routines,
# libmvec is not bit-identical to scalar libm, and whether any given loop
# vectorises depends on how much unrelated source sits around it -- so the
# oracle's numbers move when the harness merely grows.  It already happened
# once, to the mp=8 harness.
opt_flags=${OPT_FLAGS:--O2 -fno-tree-vectorize}


if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 /build-dir-made-by-build_aero.sh [/output/dir]" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build_dir=$(realpath "$1")
out_dir="${2:-$build_dir/column-oracle-aero}"
out_dir=$(realpath -m "$out_dir")

for required in module_mp_thompson.o module_mp_radar.o stub_wrf.o \
                module_mp_thompson.mod CCN_ACTIVATE.BIN \
                qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat; do
  if [ ! -e "$build_dir/$required" ]; then
    echo "build directory is missing $required; run build_aero.sh first" >&2
    exit 2
  fi
done

mkdir -p "$out_dir"
cd "$build_dir"

$fc -c $opt_flags -ffree-form -ffree-line-length-none \
  "$script_dir/probe_warm_frozen_aero.F90"
$fc $opt_flags -o probe_warm_frozen_aero stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o probe_warm_frozen_aero.o

# The devectorised build must not reach libmvec at all.
for exe in "probe_warm_frozen_aero"; do
  if nm -D "$exe" 2>/dev/null | grep -q '_ZGV'; then
    echo "libmvec SIMD math linked into $exe; oracle is not invariant" >&2
    nm -D "$exe" | grep '_ZGV' >&2
    exit 3
  fi
done

# table_ccnAct reads CCN_ACTIVATE.BIN, which is big-endian because WRF builds
# with BYTESWAPIO.  Scope the conversion to unit 20 alone exactly as
# build_aero.sh does; the program aborts if unit 20 is not the lowest free
# unit.  The classic caches are read on hardcoded unit 63 and are unaffected.
export GFORTRAN_CONVERT_UNIT='big_endian:20'
./probe_warm_frozen_aero "$out_dir" | tee "$build_dir/warm_frozen_probe.log"
unset GFORTRAN_CONVERT_UNIT

sha256sum "$out_dir/wp07-warm-frozen-rates.csv"
