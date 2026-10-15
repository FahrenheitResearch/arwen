#!/usr/bin/env bash
# Attribution control for WRF's undefined `tloc` read (NOT a fixture).
#
#   bash tools/urban_slucm_regime_oracle/tloc_control.sh WRF_SRC_DIR WORK_DIR OUT_DIR
#
# module_sf_urban.F writes the local-hour integer `tloc` only when
# AHOPTION == 1 (v4.6.1 :623-627), but the IRI_SCHEME == 1 irrigation arm
# (:874-883) reads it whatever AHOPTION is.  With AHOPTION = 0 and
# IRI_SCHEME = 1 WRF irrigates or not according to whatever the stack held;
# the -finit-real=zero fixture build makes that 0 (never irrigates).  The
# engine computes tloc by WRF's own formula whenever either arm needs it.
#
# This control copies the pinned source, widens that one guard to
# `ahoption==1 .or. iri_scheme==1`, and builds the regime oracle from the
# copy with the same flags.  Rows the engine misses against the unmodified
# fixture and matches against this control are decided by the undefined
# read alone.  The copy never becomes a fixture.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: tloc_control.sh WRF_SRC_DIR WORK_DIR OUT_DIR" >&2
    exit 2
fi
src=$(realpath "$1")
work=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$work/src"
cp "$src"/module_sf_urban.F "$src"/module_model_constants.F \
   "$src"/module_wrf_error.F "$src"/URBPARM.TBL "$src"/URBPARM_LCZ.TBL "$work/src/"
# Only the guard directly above the tloc formula (two other
# `if(ahoption==1) then` lines exist further down and are left alone).
perl -0pi -e 's/^   if\(ahoption==1\) then\n(     tloc=mod\(int\(OMG)/   if(ahoption==1 .or. iri_scheme==1) then\n$1/m or die "tloc guard not found\n"' \
    "$work/src/module_sf_urban.F"
( cd "$work/src" && sha256sum module_sf_urban.F module_model_constants.F \
      module_wrf_error.F URBPARM.TBL URBPARM_LCZ.TBL > SOURCES.sha256 )
diff "$src/module_sf_urban.F" "$work/src/module_sf_urban.F" > "$work/tloc.diff" || true
bash "$here/build.sh" "$work/src" "$work/build" "$3"
cp "$work/tloc.diff" "$3/CONTROL-tloc.diff"
