#!/usr/bin/env bash
# Grade the CUDA WSM6 on the refute sets (run_refute.sh WORK_DIR first).
# GPU work: on a shared box run this only through the box's GPU mutex.
#
#   bash grade.sh WORK_DIR        (from the repository root, PYTHONPATH=.)
#
# Strict build (GPUWM_WRF_EXACT=1) against WRF -O0: the 0 ULP claim, every
# word of every step, at the launcher's own tier, at a forced 80 tier, and
# with the columns laid out 9 rows deep.  Default build against WRF -O0
# (reported, not claimed) and against WRF run under FTZ+DAZ.
set -uo pipefail
work=$(realpath "$1")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$work/results
mkdir -p "$out"
FULL="0,5,24;1,5,24;0,20,10;1,20,10;0,37.5,8;1,37.5,8;0,60,6;1,60,6;0,90,5;1,90,5;0,120,4;1,120,4;0,121,4;1,121,4;0,180,4;1,180,4;0,600,3;1,600,3;0,1800,2;1,1800,2"
SHORT="0,20,8;1,20,8;0,60,5;1,60,5;0,180,4;1,180,4;0,600,3;1,600,3"
grade() { python "$here/refute_run.py" "$@" 2>&1 | tail -1; }
for mode in strict default; do
    if [ "$mode" = strict ]; then export GPUWM_WRF_EXACT=1; else unset GPUWM_WRF_EXACT; fi
    grade "$work/sets/fuzz45" "$FULL" --out "$out/$mode-fuzz45.json"
    grade "$work/sets/fuzz45" "$FULL" --ny 9 --out "$out/$mode-fuzz45-ny9.json"
    grade "$work/sets/fuzz45" "$FULL" --tier 80 --out "$out/$mode-fuzz45-tier80.json"
    grade "$work/sets/sub45" "$FULL" --out "$out/$mode-sub45.json"
    for s in deep80 nz64 nz65 nz2 nz3; do
        grade "$work/sets/$s" "$SHORT" --out "$out/$mode-$s.json"
    done
done
unset GPUWM_WRF_EXACT
grade "$work/sets/fuzz45" "$FULL" --flavour FTZ --out "$out/default-fuzz45-vs-FTZ.json"
grade "$work/sets/deep80" "$SHORT" --flavour FTZ --out "$out/default-deep80-vs-FTZ.json"
