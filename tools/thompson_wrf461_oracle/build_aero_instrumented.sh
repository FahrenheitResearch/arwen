#!/usr/bin/env bash
# Build and run the INSTRUMENTED aerosol oracle.
#
# Identical to build_aero.sh in every respect except one: module_mp_thompson.F
# is replaced by the copy instrument_aero_intermediates.py produces, which
# adds WRITE statements and diagnostic-only locals and changes no physics
# line.  That copy exposes the mid-call state
# tests/test_thompson_aerosol_cold_gpu.py's _WRF_COLD_REFERENCE is taken from,
# which no entry/exit column fixture can reach.
#
# FIDELITY PROOF.  After every scenario this script compares the resulting
# aero-*-column.csv and aero-*-surface.csv against the committed fixtures in
# gpuwm/data/thompson/oracle-aero/ BYTE FOR BYTE.  If the instrumentation had
# perturbed anything at all, that comparison would fail and so would this
# script.  Nothing under the WRF source tree is written.
#
# usage:
#   ./build_aero_instrumented.sh /path/to/WRF-v4.6.1 /empty/build-dir \
#       /path/CCN_ACTIVATE.BIN [/dir/with/prebuilt/dats]

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
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
wrf_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
ccn_source=$(realpath "$3")
prebuilt_dir="${4:-}"
if [ -n "$prebuilt_dir" ]; then
  prebuilt_dir=$(realpath "$prebuilt_dir")
fi

radar_source="$wrf_root/phys/module_mp_radar.F"
thompson_source="$wrf_root/phys/module_mp_thompson.F"
fixtures="$repo_root/gpuwm/data/thompson/oracle-aero"

for source in "$radar_source" "$thompson_source" "$ccn_source"; do
  if [ ! -f "$source" ]; then
    echo "missing required input: $source" >&2
    exit 2
  fi
done

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

python3 "$script_dir/instrument_aero_intermediates.py" \
  "$thompson_source" "$build_dir/module_mp_thompson_instrumented.F"

$fc -c $opt_flags -ffree-form -ffree-line-length-none \
  "$script_dir/stub_wrf.F90"
$fc -c $opt_flags -ffree-form -ffree-line-length-none "$radar_source"
$fc -c $opt_flags -cpp -DWRF_CHEM=0 -ffree-form \
  -ffree-line-length-none "$build_dir/module_mp_thompson_instrumented.F"

if [ -n "$prebuilt_dir" ]; then
  for dat in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat; do
    cp -- "$prebuilt_dir/$dat" "$build_dir/$dat"
  done
else
  $fc -c $opt_flags -ffree-form -ffree-line-length-none \
    "$script_dir/generate_tables.F90"
  $fc $opt_flags -o generate_tables stub_wrf.o module_mp_radar.o \
    module_mp_thompson_instrumented.o generate_tables.o
  ./generate_tables | tee generate.log
fi

echo "$QR_ACR_QG_SHA  qr_acr_qg_V4.dat" | sha256sum -c -
echo "$QR_ACR_QS_SHA  qr_acr_qsV2.dat" | sha256sum -c -
echo "$FREEZEH2O_SHA  freezeH2O.dat" | sha256sum -c -

cp -- "$ccn_source" "$build_dir/CCN_ACTIVATE.BIN"

$fc -c $opt_flags -ffree-form -ffree-line-length-none \
  "$script_dir/run_column_aero.F90"
$fc $opt_flags -o run_column_aero stub_wrf.o module_mp_radar.o \
  module_mp_thompson_instrumented.o run_column_aero.o

# The devectorised build must not reach libmvec at all.  If it does, the
# fixtures below are not the invariant ones and must not be committed.
for exe in "generate_tables" "run_column_aero"; do
  if nm -D "$exe" 2>/dev/null | grep -q '_ZGV'; then
    echo "libmvec SIMD math linked into $exe; oracle is not invariant" >&2
    nm -D "$exe" | grep '_ZGV' >&2
    exit 3
  fi
done

export GFORTRAN_CONVERT_UNIT='big_endian:20'

mkdir -p column-oracle-aero intermediates

# ONE FRESH PROCESS PER SCENARIO, exactly as build_aero.sh requires.
for scenario in aero-init-profile aero-sfc-emit aero-ccn-activate \
                aero-ccn-sweep aero-drop-evap aero-nc-auto \
                aero-nc-accrete aero-nc-effrad aero-nc-sed \
                aero-scav-rain aero-scav-frozen aero-ice-demott-dep \
                aero-ice-demott-idxin aero-ice-koop \
                aero-cloud-freeze-nc aero-nc-cap aero-warm-overlap \
                aero-cold-overlap aero-reduces-to-classic \
                wp08-nusweep wp08-melt wp08-freeze; do
  rm -f cold-network-intermediates.csv cloud-sed-intermediates.csv \
        phase-cleanup-intermediates.csv
  ./run_column_aero "$scenario" column-oracle-aero \
    | tee "column-$scenario.log"
  # A scenario whose every column is hydrometeor-free and sub-saturated
  # returns at module_mp_thompson.F:2020 before reaching any anchor, so it
  # legitimately emits nothing.  Record that as an empty file rather than
  # failing: every scenario the committed tables cover does emit.
  for probe in cold-network cloud-sed phase-cleanup; do
    if [ -f "$probe-intermediates.csv" ]; then
      mv "$probe-intermediates.csv" "intermediates/$scenario-$probe.csv"
    else
      : > "intermediates/$scenario-$probe.csv"
    fi
  done
done

unset GFORTRAN_CONVERT_UNIT

echo "$QR_ACR_QG_SHA  qr_acr_qg_V4.dat" | sha256sum -c -
echo "$QR_ACR_QS_SHA  qr_acr_qsV2.dat" | sha256sum -c -
echo "$FREEZEH2O_SHA  freezeH2O.dat" | sha256sum -c -
echo "$CCN_SHA  CCN_ACTIVATE.BIN" | sha256sum -c -

# THE FIDELITY PROOF.  Every regenerated fixture must be byte-identical to
# the committed one; that is what makes the intermediate columns above
# trustworthy as WRF's own.
status=0
for generated in column-oracle-aero/*.csv; do
  name=$(basename "$generated")
  if [ ! -f "$fixtures/$name" ]; then
    echo "NO COMMITTED FIXTURE: $name" >&2
    status=1
    continue
  fi
  if ! cmp -s "$generated" "$fixtures/$name"; then
    echo "INSTRUMENTATION IS NOT INERT: $name differs from the committed fixture" >&2
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  echo "FIDELITY PROOF FAILED" >&2
  exit 1
fi
echo "fidelity proof: all $(ls column-oracle-aero/*.csv | wc -l) regenerated fixtures are byte-identical to the committed ones"

wc -l intermediates/*.csv
sha256sum intermediates/*.csv | tee INTERMEDIATE_SHA256SUMS

cat <<'NOTE'

Next: check the committed _WRF_COLD_REFERENCE table against what was just
generated.

    python3 tools/thompson_wrf461_oracle/check_instrumented_tables_aero.py \
        <build-dir>/intermediates
NOTE
