#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 /path/to/WRF-v4.6.1 /empty/build-directory" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
wrf_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
radar_source="$wrf_root/phys/module_mp_radar.F"
thompson_source="$wrf_root/phys/module_mp_thompson.F"

fc=${FC:-gfortran}

# -fno-tree-vectorize is load-bearing, not a performance choice.
#
# GCC enables -ftree-vectorize at -O2 from GCC 12 on.  When a loop containing
# exp/pow/log vectorizes, GCC replaces glibc's scalar expf/powf/log10 with
# glibc's libmvec SIMD entry points (_ZGVbN4v_expf, _ZGVbN4vv_powf,
# _ZGVbN2v_exp, _ZGVbN2vv_pow, _ZGVbN2v_log10).  Those carry a looser accuracy
# contract than the scalar routines and are not bit-identical to them, so the
# oracle's answer moves by 1-2 ULP depending on which loops the vectoriser
# chose.  That choice depends on the cost model, which depends on how much
# UNRELATED source surrounds the loop -- so merely growing run_column.F90
# silently changed the fixtures once already: warm/mixed/ice were generated
# when the file was 227 lines and the base-state loop vectorised, and the
# other fixtures when it was ~1000 lines and it did not.  The committed set
# ended up self-contradictory: two fixtures at the same dz disagreed by 1 ULP
# on p = p0*exp(-z/8000), which is scenario-independent arithmetic.
#
# With -fno-tree-vectorize no libmvec symbol is linked at all (verify with
# `nm -D run_column | grep _ZGV`, which must print nothing), and the oracle
# becomes invariant: gfortran 12.5.0, 13.4.0, 14.3.0 and 15.2.0 at -O1, -O2
# and -O3, with and without -ffp-contract=off, all produce a byte-identical
# 92-file fixture set.  That is what makes this an authority a reader can
# re-derive rather than a record of one machine's optimiser.
opt_flags=${OPT_FLAGS:--O2 -fno-tree-vectorize}

for source in "$radar_source" "$thompson_source"; do
  if [ ! -f "$source" ]; then
    echo "missing official WRF source: $source" >&2
    exit 2
  fi
done
mkdir -p "$build_dir"
for output in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat \
              thompson_aux_tables.dat; do
  if [ -e "$build_dir/$output" ]; then
    echo "refusing to reuse existing oracle output: $build_dir/$output" >&2
    exit 2
  fi
done

cd "$build_dir"
$fc -c $opt_flags -ffree-form -ffree-line-length-none \
  "$script_dir/stub_wrf.F90"
$fc -c $opt_flags -ffree-form -ffree-line-length-none "$radar_source"
$fc -c $opt_flags -cpp -DWRF_CHEM=0 -ffree-form \
  -ffree-line-length-none "$thompson_source"
$fc -c $opt_flags -ffree-form -ffree-line-length-none \
  "$script_dir/generate_tables.F90"
$fc $opt_flags -o generate_tables stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o generate_tables.o
./generate_tables | tee generate.log

# A second call must take WRF's read path successfully before the cache is
# accepted.  This catches record-layout or short-write failures immediately.
./generate_tables | tee readback.log

$fc -c $opt_flags -ffree-form -ffree-line-length-none \
  "$script_dir/dump_aux_tables.F90"
$fc $opt_flags -o dump_aux_tables stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o dump_aux_tables.o
./dump_aux_tables | tee auxiliary.log

$fc -c $opt_flags -ffree-form -ffree-line-length-none \
  "$script_dir/run_column.F90"
$fc $opt_flags -o run_column stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o run_column.o

# The devectorised build must not reach libmvec at all.  If it does, the
# fixtures below are not the invariant ones and must not be committed.
if nm -D run_column | grep -q '_ZGV'; then
  echo "libmvec SIMD math linked into run_column; oracle is not invariant" >&2
  nm -D run_column | grep '_ZGV' >&2
  exit 3
fi

mkdir -p column-oracle
for scenario in warm mixed ice condense rain-sed ice-sed cloud-sed snow-sed \
                graupel-sed warm-auto rain-self warm-accrete ice-auto \
                rain-evap snow-subl graupel-subl ice-dep ice-nuc \
                snow-dep graupel-dep snow-melt graupel-melt rain-freeze \
                cloud-freeze graupel-rime snow-rime snow-ice rain-ice \
                rain-snow rain-graupel snow-rime-convert warm-overlap \
                rain-snow-graupel-overlap rain-ice-graupel-overlap \
                frozen-vapor-overlap cold-cloud-overlap \
                frozen-vapor-nucleation-overlap cold-ice-rain-overlap \
                cold-full-overlap cold-cloud-rain-overlap \
                cloud-condense-sed cloud-condense-nofall \
                cloud-rain-condense-sed cloud-rain-condense-nofall \
                condense-fall-attempt warm-frozen-subsat; do
  ./run_column "$scenario" column-oracle | tee "column-$scenario.log"
done

stat -c '%n %s' qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat \
  thompson_aux_tables.dat
sha256sum qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat \
  thompson_aux_tables.dat | tee SHA256SUMS
sha256sum column-oracle/*.csv | tee COLUMN_SHA256SUMS

# The receipt.  Everything a reader needs in order to re-derive this fixture
# set, and everything a test needs in order to notice that the harness moved
# out from under the fixtures.  `tests/test_thompson_oracle_provenance.py`
# reads the committed copy of this file back and fails when the harness
# sources or the fixtures no longer hash to what it records.
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
{
  echo "# WRF v4.6.1 Thompson column-oracle provenance receipt"
  echo "# Written by tools/thompson_wrf461_oracle/build.sh."
  echo
  echo "[wrf_source]"
  echo "commit = d66e442fccc04111067e29274c9f9eaccc3cef28"
  echo "tag = v4.6.1"
  echo "origin = https://github.com/wrf-model/WRF.git"
  echo "phys/module_mp_thompson.F = $(sha_of "$thompson_source")"
  echo "phys/module_mp_radar.F = $(sha_of "$radar_source")"
  echo
  echo "[harness_source]"
  for f in build.sh stub_wrf.F90 generate_tables.F90 dump_aux_tables.F90 \
           run_column.F90; do
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
  echo "libmvec_symbols = $(nm -D run_column | grep -c '_ZGV')"
  echo
  echo "[binaries]"
  for b in generate_tables dump_aux_tables run_column; do
    echo "$b = $(sha_of "$b")"
  done
  echo
  echo "[table_cache]"
  for t in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat \
           thompson_aux_tables.dat; do
    echo "$t = $(sha_of "$t")"
  done
  echo
  echo "[fixtures]"
  echo "count = $(ls column-oracle/*.csv | wc -l)"
  echo "rollup = $(cd column-oracle && cat *.csv | sha256sum | cut -d' ' -f1)"
  echo
  echo "[fixture_sha256]"
  ( cd column-oracle && for f in *.csv; do echo "$f = $(sha_of "$f")"; done )
} | tee PROVENANCE.txt
