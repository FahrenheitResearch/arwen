#!/usr/bin/env bash
# Build and run the PER-KERNEL aerosol Fortran probes.
#
# build_aero.sh produces whole-scheme column fixtures: nineteen scenarios of
# mp_gt_driver, entry state in and exit state out.  Those fixtures pin the
# endpoint of nineteen coupled processes, so they cannot isolate a single WRF
# rate.  Two gpuwm test files therefore embed literal Fortran tables for
# individual blocks:
#
#   tests/test_thompson_aerosol_warm_gpu.py  _WARM_RATE_ORACLE (12348 rows)
#                                            _NCTEN_BALANCE_ORACLE (11025)
#   tests/test_thompson_aerosol_cold_gpu.py  _WRF_COLD_WARM_LOOP
#
# Until 2026-07-31 the programs that produced those tables lived only in an
# agent scratch directory.  They are now committed next to this script, and
# this script builds and runs them so any reader can re-derive every number:
#
#   probe_warm_rates_aero.F90       module_mp_thompson.F:2144-2232, :2996-3019
#   probe_cold_warm_loop_aero.F90   module_mp_thompson.F:1826-1842, :2144-2232
#                                   at five SUB-FREEZING temperatures
#
# Both LINK the same compiled module_mp_thompson.o that build_aero.sh builds,
# with build_aero.sh's exact flags, and both call thompson_init exactly as
# run_column_aero.F90 does.  Nothing under the WRF source tree is modified.
#
# usage:
#   ./build_aero_probes.sh /path/to/WRF-v4.6.1 /empty/build-dir \
#       /path/CCN_ACTIVATE.BIN [/dir/with/prebuilt/dats]
#
# Then verify the committed tests against the result:
#   python3 check_probe_oracles_aero.py <build-dir>/probe-oracle-aero

set -euo pipefail

fc=${FC:-gfortran}

# -fno-tree-vectorize is LOAD-BEARING, not a performance choice, and it is
# the same property build.sh states for the classic harness.
#
# GCC enables -ftree-vectorize at -O2 from GCC 12 on.  When a loop containing
# exp/pow/log vectorizes, GCC links glibc's libmvec SIMD entry points
# (_ZGVbN4v_expf, _ZGVbN4vv_powf, _ZGVbN2v_exp, _ZGVbN2vv_pow, _ZGVbN2v_log10)
# in place of the scalar routines.  libmvec is not bit-identical to scalar
# libm, so the oracle's answer moves by 1-2 ULP depending on which loops the
# vectoriser chose -- and that choice is a cost-model decision that depends on
# how much UNRELATED source surrounds the loop.
#
# This is not hypothetical here.  It already happened to the mp=8 harness:
# run_column.F90's base-state loop vectorised when the file was 227 lines and
# did not when it was ~1000, so three committed fixtures disagreed with the
# other 43 on p = p0*exp(-z/8000), which is scenario-independent arithmetic.
# run_column_aero.F90 builds its base state the same way (:231-232, :273) and
# is 729 lines; its 52 fixtures are currently self-consistent by where the
# cost model happens to land, not by construction.  One edit to that file can
# flip it silently.
#
# With -fno-tree-vectorize no libmvec symbol is linked at all, and the classic
# harness measured the result invariant: gfortran 12.5.0 / 13.4.0 / 14.3.0 /
# 15.2.0 at -O1, -O2 and -O3, with and without -ffp-contract=off, all produce
# a byte-identical 92-file set.
opt_flags=${OPT_FLAGS:--O2 -fno-tree-vectorize}


if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 /path/to/WRF-v4.6.1 /empty/build-directory /path/to/CCN_ACTIVATE.BIN [prebuilt-table-dir]" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
wrf_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
ccn_source=$(realpath "$3")
prebuilt_dir="${4:-}"
if [ -n "$prebuilt_dir" ]; then
  prebuilt_dir=$(realpath "$prebuilt_dir")
fi

radar_source="$wrf_root/phys/module_mp_radar.F"
thompson_source="$wrf_root/phys/module_mp_thompson.F"

for source in "$radar_source" "$thompson_source" "$ccn_source"; do
  if [ ! -f "$source" ]; then
    echo "missing required input: $source" >&2
    exit 2
  fi
done

# Same content pins build_aero.sh uses.  CCN_ACTIVATE.BIN is a WRF-DISTRIBUTED
# parcel-model product; no recompilation regenerates it.
CCN_SHA=f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd
QR_ACR_QG_SHA=89b779855847b2acdca1b40e24c5f1bd89b0c6ed105ca91a5a076d80c2437c3f
QR_ACR_QS_SHA=47350be20bd59c9f31378dd5805ce7d35fd14bebcfafb4ade56626f6eed818d7
FREEZEH2O_SHA=c235d1ce6f8750a671b2273d0e216ed3acf9a869bfd52a14676826f87aab5c02

actual_ccn_sha=$(sha256sum "$ccn_source" | awk '{print $1}')
if [ "$actual_ccn_sha" != "$CCN_SHA" ]; then
  echo "CCN_ACTIVATE.BIN SHA-256 $actual_ccn_sha; expected $CCN_SHA" >&2
  exit 2
fi

mkdir -p "$build_dir"
for output in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat \
              CCN_ACTIVATE.BIN; do
  if [ -e "$build_dir/$output" ]; then
    echo "refusing to reuse existing oracle input: $build_dir/$output" >&2
    exit 2
  fi
done

cd "$build_dir"

$fc -c $opt_flags -ffree-form -ffree-line-length-none \
  "$script_dir/stub_wrf.F90"
$fc -c $opt_flags -ffree-form -ffree-line-length-none "$radar_source"
$fc -c $opt_flags -cpp -DWRF_CHEM=0 -ffree-form \
  -ffree-line-length-none "$thompson_source"

if [ -n "$prebuilt_dir" ]; then
  for dat in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat; do
    if [ ! -f "$prebuilt_dir/$dat" ]; then
      echo "prebuilt table directory is missing $dat" >&2
      exit 2
    fi
    cp -- "$prebuilt_dir/$dat" "$build_dir/$dat"
  done
else
  $fc -c $opt_flags -ffree-form -ffree-line-length-none \
    "$script_dir/generate_tables.F90"
  $fc $opt_flags -o generate_tables stub_wrf.o module_mp_radar.o \
    module_mp_thompson.o generate_tables.o
  ./generate_tables | tee generate.log
fi

# Verify the NATIVE-endian classic caches before any endian request is made.
echo "$QR_ACR_QG_SHA  qr_acr_qg_V4.dat" | sha256sum -c -
echo "$QR_ACR_QS_SHA  qr_acr_qsV2.dat" | sha256sum -c -
echo "$FREEZEH2O_SHA  freezeH2O.dat" | sha256sum -c -

cp -- "$ccn_source" "$build_dir/CCN_ACTIVATE.BIN"

$fc -c $opt_flags -ffree-form -ffree-line-length-none \
  "$script_dir/probe_warm_rates_aero.F90"
$fc $opt_flags -o probe_warm_rates_aero stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o probe_warm_rates_aero.o

$fc -c $opt_flags -ffree-form -ffree-line-length-none \
  "$script_dir/probe_cold_warm_loop_aero.F90"
$fc $opt_flags -o probe_cold_warm_loop_aero stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o probe_cold_warm_loop_aero.o

# The devectorised build must not reach libmvec at all.  If it does, the
# fixtures below are not the invariant ones and must not be committed.
for exe in "generate_tables" "probe_cold_warm_loop_aero" "probe_warm_rates_aero"; do
  if nm -D "$exe" 2>/dev/null | grep -q '_ZGV'; then
    echo "libmvec SIMD math linked into $exe; oracle is not invariant" >&2
    nm -D "$exe" | grep '_ZGV' >&2
    exit 3
  fi
done

# CCN_ACTIVATE.BIN is big-endian (WRF builds with BYTESWAPIO); the classic
# caches this harness reads are native.  Request conversion for unit 20 only:
# table_ccnAct picks the lowest free unit in 20..99, which is 20 in a fresh
# process, and both probes assert that or abort.
export GFORTRAN_CONVERT_UNIT='big_endian:20'

mkdir -p probe-oracle-aero

# ONE FRESH PROCESS PER PROGRAM, for the same reason build_aero.sh uses one
# per scenario: table_ccnAct runs inside thompson_init's one-time micro_init
# block while is_aerosol_aware is reset on every entry.
./probe_warm_rates_aero probe-oracle-aero | tee probe-warm-rates.log
./probe_cold_warm_loop_aero probe-oracle-aero | tee probe-cold-warm-loop.log

unset GFORTRAN_CONVERT_UNIT

# Nothing in the probe path may have touched the classic caches.
echo "$QR_ACR_QG_SHA  qr_acr_qg_V4.dat" | sha256sum -c -
echo "$QR_ACR_QS_SHA  qr_acr_qsV2.dat" | sha256sum -c -
echo "$FREEZEH2O_SHA  freezeH2O.dat" | sha256sum -c -
echo "$CCN_SHA  CCN_ACTIVATE.BIN" | sha256sum -c -

wc -l probe-oracle-aero/*.csv
sha256sum probe-oracle-aero/*.csv | tee PROBE_ORACLE_SHA256SUMS

cat <<'NOTE'

Next: check the committed test tables against what was just generated.

    python3 tools/thompson_wrf461_oracle/check_probe_oracles_aero.py \
        <build-dir>/probe-oracle-aero
NOTE
