#!/usr/bin/env bash
# Produce gpuwm/data/thompson/oracle-aero/aero-exit-temperature.csv, and PROVE
# that producing it changed nothing.
#
# The 19 committed column fixtures are, and stay, the output of an unmodified
# module_mp_thompson.F.  This script builds the harness TWICE from the same
# WRF tree -- once pristine, once with instrument_exit_temperature_aero.py's
# two write statements added -- runs all 19 scenarios through both, and
# refuses to emit the receipt unless all 44 pristine output files are BYTE FOR
# BYTE identical between the two builds.  The receipt is therefore WRF's own
# t1d, obtained without the fixtures ever depending on a patched compiler
# input.
#
# usage:
#   ./build_exit_temperature_aero.sh /path/to/WRF-v4.6.1 /empty/build-dir \
#       /path/CCN_ACTIVATE.BIN [/dir/with/prebuilt/dats]

set -euo pipefail

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

python_bin="${PYTHON:-python3}"

mkdir -p "$build_dir"
if [ -n "$(ls -A "$build_dir")" ]; then
  echo "refusing to write into a non-empty build directory: $build_dir" >&2
  exit 2
fi

# 1. The pristine reference run.
"$script_dir/build_aero.sh" "$wrf_root" "$build_dir/pristine" \
  "$ccn_source" ${prebuilt_dir:+"$prebuilt_dir"} \
  > "$build_dir/pristine.log" 2>&1
echo "pristine build ok"

# 2. The instrumented tree: a copy of the WRF sources with two write
#    statements added to module_mp_thompson.F and NOTHING else.
mkdir -p "$build_dir/instrumented-src/phys"
cp -- "$wrf_root/phys/module_mp_radar.F" "$build_dir/instrumented-src/phys/"
"$python_bin" "$script_dir/instrument_exit_temperature_aero.py" \
  "$wrf_root/phys/module_mp_thompson.F" \
  "$build_dir/instrumented-src/phys/module_mp_thompson.F"

# One fresh process per scenario is mandatory for the same reason build_aero.sh
# states, and the dumps append, so they are moved aside per scenario.
"$script_dir/build_aero.sh" "$build_dir/instrumented-src" \
  "$build_dir/instrumented" "$ccn_source" ${prebuilt_dir:+"$prebuilt_dir"} \
  > "$build_dir/instrumented.log" 2>&1
echo "instrumented build ok"

# 3. THE NEUTRALITY PROOF.  Every file the pristine harness emits must be
#    byte-identical under instrumentation, or the receipt is worthless.
differing=0
for file in "$build_dir/pristine/column-oracle-aero"/*; do
  name=$(basename "$file")
  if ! cmp -s "$file" "$build_dir/instrumented/column-oracle-aero/$name"; then
    echo "INSTRUMENTATION IS NOT NEUTRAL: $name differs" >&2
    differing=$((differing + 1))
  fi
done
if [ "$differing" -ne 0 ]; then
  echo "$differing file(s) changed under instrumentation; refusing to emit" >&2
  exit 1
fi
count=$(ls -1 "$build_dir/pristine/column-oracle-aero" | wc -l)
echo "neutrality proven: $count/$count files byte-identical"

# 4. Fold the dumps into one receipt.
"$python_bin" "$script_dir/make_exit_temperature_aero.py" \
  "$build_dir/instrumented" "$build_dir/aero-exit-temperature.csv"

echo "receipt: $build_dir/aero-exit-temperature.csv"
sha256sum "$build_dir/aero-exit-temperature.csv"
