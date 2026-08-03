#!/usr/bin/env bash
# Aerosol-aware (mp_physics=28) sibling of build.sh.
#
# build.sh, run_column.F90, generate_tables.F90, dump_aux_tables.F90 and
# stub_wrf.F90 are NOT modified by this script.  WRF's is_aerosol_aware is
# a module-SAVEd flag set by optional-argument presence, so a single
# program cannot produce both families; keeping them separate is what keeps
# the four classic .dat SHA-256 pins and the 92 committed mp=8 column
# fixtures out of reach.
#
# usage:
#   ./build_aero.sh /path/to/WRF-v4.6.1 /empty/build-dir /path/CCN_ACTIVATE.BIN [/dir/with/prebuilt/dats]
#
# The optional fourth argument copies previously generated classic caches
# (qr_acr_qg_V4.dat, qr_acr_qsV2.dat, freezeH2O.dat) instead of spending
# ~3 minutes regenerating them.  Their SHA-256s are verified against the
# same pins the fresh path produces, so the shortcut cannot change results.

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

# CCN_ACTIVATE.BIN is a WRF-DISTRIBUTED parcel-model product.  No
# recompilation regenerates it, so it is pinned by content here.
CCN_SHA=f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd
actual_ccn_sha=$(sha256sum "$ccn_source" | awk '{print $1}')
if [ "$actual_ccn_sha" != "$CCN_SHA" ]; then
  echo "CCN_ACTIVATE.BIN SHA-256 $actual_ccn_sha; expected $CCN_SHA" >&2
  exit 2
fi
actual_ccn_bytes=$(stat -c '%s' "$ccn_source")
if [ "$actual_ccn_bytes" != "35288" ]; then
  echo "CCN_ACTIVATE.BIN is $actual_ccn_bytes bytes; expected 35288" >&2
  exit 2
fi

# The four classic-cache pins from gpuwm/core/thompson_contract.py.
QR_ACR_QG_SHA=89b779855847b2acdca1b40e24c5f1bd89b0c6ed105ca91a5a076d80c2437c3f
QR_ACR_QS_SHA=47350be20bd59c9f31378dd5805ce7d35fd14bebcfafb4ade56626f6eed818d7
FREEZEH2O_SHA=c235d1ce6f8750a671b2273d0e216ed3acf9a869bfd52a14676826f87aab5c02

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

# The classic caches are NATIVE endian.  Verifying them here, before any
# GFORTRAN_CONVERT_UNIT is exported, is what proves the big-endian request
# below is scoped to CCN_ACTIVATE.BIN alone.
echo "$QR_ACR_QG_SHA  qr_acr_qg_V4.dat" | sha256sum -c -
echo "$QR_ACR_QS_SHA  qr_acr_qsV2.dat" | sha256sum -c -
echo "$FREEZEH2O_SHA  freezeH2O.dat" | sha256sum -c -

# table_ccnAct OPENs 'CCN_ACTIVATE.BIN' by bare relative name from the
# working directory.  Copy, never symlink, so the build directory is a
# self-contained record of exactly which bytes were read.
cp -- "$ccn_source" "$build_dir/CCN_ACTIVATE.BIN"

$fc -c $opt_flags -ffree-form -ffree-line-length-none \
  "$script_dir/dump_ccn_table.F90"
$fc $opt_flags -o dump_ccn_table stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o dump_ccn_table.o

$fc -c $opt_flags -ffree-form -ffree-line-length-none \
  "$script_dir/probe_aero_functions.F90"
$fc $opt_flags -o probe_aero_functions stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o probe_aero_functions.o

$fc -c $opt_flags -ffree-form -ffree-line-length-none \
  "$script_dir/run_column_aero.F90"
$fc $opt_flags -o run_column_aero stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o run_column_aero.o

# The devectorised build must not reach libmvec at all.  If it does, the
# fixtures below are not the invariant ones and must not be committed.
for exe in "dump_ccn_table" "generate_tables" "probe_aero_functions" "run_column_aero"; do
  if nm -D "$exe" 2>/dev/null | grep -q '_ZGV'; then
    echo "libmvec SIMD math linked into $exe; oracle is not invariant" >&2
    nm -D "$exe" | grep '_ZGV' >&2
    exit 3
  fi
done

# ENDIANNESS.  CCN_ACTIVATE.BIN is big-endian because WRF builds with
# BYTESWAPIO; the classic caches this harness generates are native.  A
# global -fconvert=big-endian would rewrite all three .dat files at
# IDENTICAL sizes and fail every SHA pin above.  Instead request conversion
# for unit 20 only: table_ccnAct picks the lowest free unit in 20..99,
# which is 20 in a fresh process, and run_column_aero/dump_ccn_table/
# probe_aero_functions all assert that or abort.  The classic caches are
# read on hardcoded unit 63 and are unaffected.
export GFORTRAN_CONVERT_UNIT='big_endian:20'

mkdir -p column-oracle-aero

./dump_ccn_table column-oracle-aero/tnccn_act_native.bin \
  | tee ccn_table_dump.log

./probe_aero_functions column-oracle-aero | tee aero_probe.log

# ONE FRESH PROCESS PER SCENARIO, mandatory.  table_ccnAct is called inside
# thompson_init's one-time `if (micro_init)` block (call at 1013, block
# opens at 652) while is_aerosol_aware is reset on every thompson_init
# entry (468).  A second init in the same process therefore leaves
# tnccn_act at its all-ones prefill and activ_ncloud silently returns 100%
# activation with no error.
for scenario in aero-init-profile aero-sfc-emit aero-ccn-activate \
                aero-ccn-sweep aero-drop-evap aero-nc-auto \
                aero-nc-accrete aero-nc-effrad aero-nc-sed \
                aero-scav-rain aero-scav-frozen aero-ice-demott-dep \
                aero-ice-demott-idxin aero-ice-koop \
                aero-cloud-freeze-nc aero-nc-cap aero-warm-overlap \
                aero-cold-overlap aero-reduces-to-classic \
                wp08-nusweep wp08-melt wp08-freeze; do
  ./run_column_aero "$scenario" column-oracle-aero \
    | tee "column-$scenario.log"
done

unset GFORTRAN_CONVERT_UNIT

# Re-verify after every run: nothing in the aerosol path may have touched
# the classic caches.
echo "$QR_ACR_QG_SHA  qr_acr_qg_V4.dat" | sha256sum -c -
echo "$QR_ACR_QS_SHA  qr_acr_qsV2.dat" | sha256sum -c -
echo "$FREEZEH2O_SHA  freezeH2O.dat" | sha256sum -c -
echo "$CCN_SHA  CCN_ACTIVATE.BIN" | sha256sum -c -

stat -c '%n %s' qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat \
  CCN_ACTIVATE.BIN
sha256sum column-oracle-aero/*.csv | tee COLUMN_AERO_SHA256SUMS
sha256sum column-oracle-aero/tnccn_act_native.bin | tee CCN_TABLE_SHA256SUM

# The receipt.  Same contract build.sh writes for the classic oracle, and
# read back by tests/test_thompson_aerosol_oracle_provenance.py: the harness
# sources are hashed so editing run_column_aero.F90 without regenerating
# fails loudly at the edit rather than silently years later, every fixture is
# hashed, and the toolchain and the libmvec symbol count are recorded so a
# reader can tell whether the numbers came from scalar libm.
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
{
  echo "# WRF v4.6.1 aerosol-aware (mp_physics=28) column-oracle receipt"
  echo "# Written by tools/thompson_wrf461_oracle/build_aero.sh."
  echo
  echo "[wrf_source]"
  echo "commit = d66e442fccc04111067e29274c9f9eaccc3cef28"
  echo "tag = v4.6.1"
  echo "origin = https://github.com/wrf-model/WRF.git"
  echo "phys/module_mp_thompson.F = $(sha_of "$thompson_source")"
  echo "phys/module_mp_radar.F = $(sha_of "$radar_source")"
  echo
  echo "[aerosol_asset]"
  echo "CCN_ACTIVATE.BIN = $(sha_of "$build_dir/CCN_ACTIVATE.BIN")"
  echo "CCN_ACTIVATE.BIN.bytes = $(stat -c '%s' "$build_dir/CCN_ACTIVATE.BIN")"
  echo
  echo "[harness_source]"
  for f in build_aero.sh stub_wrf.F90 generate_tables.F90            dump_ccn_table.F90 probe_aero_functions.F90 run_column_aero.F90; do
    echo "$f = $(sha_of "$script_dir/$f")"
  done
  echo
  echo "[toolchain]"
  echo "fortran = $($fc --version | head -1)"
  echo "libc = $(ldd --version | head -1)"
  echo "uname = $(uname -srm)"
  echo "opt_flags = $opt_flags"
  echo "compile = $fc -c $opt_flags -ffree-form -ffree-line-length-none SOURCE"
  echo "compile_thompson = $fc -c $opt_flags -cpp -DWRF_CHEM=0 -ffree-form -ffree-line-length-none SOURCE"
  echo "link = $fc $opt_flags -o PROGRAM stub_wrf.o module_mp_radar.o module_mp_thompson.o PROGRAM.o"
  echo "libmvec_symbols = $(nm -D run_column_aero | grep -c '_ZGV' || true)"
  echo
  echo "[binaries]"
  for b in dump_ccn_table probe_aero_functions run_column_aero; do
    echo "$b = $(sha_of "$b")"
  done
  echo
  echo "[table_cache]"
  for t in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat; do
    echo "$t = $(sha_of "$t")"
  done
  echo
  echo "[fixtures]"
  echo "count = $(ls column-oracle-aero/*.csv | wc -l)"
  echo "rollup = $(cd column-oracle-aero && cat *.csv | sha256sum | cut -d' ' -f1)"
  echo
  echo "[fixture_sha256]"
  ( cd column-oracle-aero && for f in *.csv; do echo "$f = $(sha_of "$f")"; done )
} | tee PROVENANCE.txt
