#!/usr/bin/env bash
# A MEASUREMENT COPY of a gpuwm tree that reproduces WRF v4.6.1's
# out-of-bounds rain-graupel table read, for the column oracle only.
#
# usage: make_racg_read_copy.sh TREE COPY
#
# WRF allocates tcg_racg, tmr_racg, tcr_gacr, tnr_racg and tnr_gacr with a
# graupel-density axis of extent 1 when the scheme is not hail aware
# (module_mp_thompson.F:465, :607-615) and reads them with idx_bg = 5
# (:2527-2545): in Fortran order that is the slab's index plus 4*37*37.
# WOOF reads the slab the tables hold (aaf_racg_index in
# gpuwm/core/kernels/thompson_aerosol_common.cuh); that is the declared
# rain-graupel divergence.  Running the oracle on this copy separates the
# read from everything else: with it reproduced, every remaining difference
# is a defect.
#
# Past the end of the arrays WRF reads unrelated memory, which no copy can
# reproduce; there the copy's index faults instead, so an oracle run that
# completes on the copy proves no such read occurred in its columns.
#
# The copy hard-links every file of TREE except the one header it rewrites
# (written to a new inode, so TREE is never touched).  It is an instrument:
# never commit it, never run a forecast from it.
set -euo pipefail
tree=$(realpath "$1")
copy=$(realpath -m "$2")
[ -e "$copy" ] && { echo "refusing: $copy exists" >&2; exit 2; }
cp -al "$tree" "$copy"
header=$copy/gpuwm/core/kernels/thompson_aerosol_common.cuh
python3 - "$header" <<'PY'
import os
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8").read()
old = """    return (size_t)(idx_g1 - 1) + (size_t)37 * ((size_t)(idx_g - 1)
        + (size_t)37 * ((size_t)(idx_r1 - 1)
        + (size_t)37 * (size_t)(idx_r - 1)));"""
new = """    const size_t wrf = (size_t)4 * 37 * 37 + (size_t)(idx_g1 - 1)
        + (size_t)37 * ((size_t)(idx_g - 1)
        + (size_t)37 * ((size_t)(idx_r1 - 1)
        + (size_t)37 * (size_t)(idx_r - 1)));
    return wrf < (size_t)37 * 37 * 37 * 37 ? wrf : ((size_t)1 << 40);"""
if text.count(old) != 1:
    raise SystemExit("aaf_racg_index has changed; update this script")
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8", newline="") as handle:
    handle.write(text.replace(old, new))
os.replace(tmp, path)
PY
if cmp -s "$tree/gpuwm/core/kernels/thompson_aerosol_common.cuh" "$header"; then
  echo "the copy's header is unchanged" >&2
  exit 3
fi
echo "$copy: WRF's rain-graupel read reproduced (measurement only)"
