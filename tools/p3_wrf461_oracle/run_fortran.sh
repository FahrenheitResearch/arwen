#!/usr/bin/env bash
# Run every fixture through the built Fortran arms.
#   bash run_fortran.sh <BUILD_DIR> <FIXDIR> [ARMS...]   (default arms: A B)
# Dumps land in <BUILD_DIR>/out/<arm>/<fixture>.ref, FORMAT.md layout.
set -euo pipefail
OUT=${1:?BUILD_DIR}; FIX=${2:?FIXDIR}; shift 2
ARMS=${*:-A B}
FIX=$(cd "$FIX" && pwd)
for ARM in $ARMS; do
  D="$OUT/out/$ARM"; rm -rf "$D"; mkdir -p "$D"
  for CTL in "$FIX"/*.ctl; do
    N=$(basename "$CTL" .ctl)
    sed -e "s#@TABLEDIR@#$OUT/tables#" -e "s#@FIXDIR@#$FIX#" -e "s#@OUTDIR@#$D#" "$CTL" > "$D/$N.ctl"
    if "$OUT/build/$ARM/p3_ref_driver" "$D/$N.ctl" > "$D/$N.log" 2>&1; then
      echo "arm=$ARM $N ok $(sha256sum "$D/$N.ref" | cut -c1-16)"
    else
      echo "arm=$ARM $N FAILED rc=$?"; tail -5 "$D/$N.log"
    fi
  done
done
