#!/usr/bin/env bash
# Run every built Fortran arm on every fixture input.
# usage: run_oracle.sh BUILD_DIR FIXTURE_DIR OUT_ROOT
set -euo pipefail
build=$1; fixture=$2; out=$3
for arm in stock defined only2 only5 only6; do
    mkdir -p "${out}/${arm}"
    for f in "${fixture}"/*.in; do
        n=$(basename "$f" .in)
        "${build}/run_wdm6_${arm}" "$f" "${out}/${arm}/${n}.out" \
            "${out}/${arm}/constants-${n}.txt"
    done
done
