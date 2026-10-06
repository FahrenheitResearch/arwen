#!/usr/bin/env bash
# Build the tag's WPS (sorc/hrrr_wps.fd/WPSV3.9.1) the way
# sorc/build_hrrr_wps.sh does: ungrib and metgrid, the two programs the tag's
# makeguess script runs before real. It links the WRF tree build_wrf.sh built.
# No source edits: the tag's own configure.wps.optim is copied into place as
# its build script does.
set -euo pipefail
BASE=/opt/hrrr/sorc
export BASE
EXEC=/opt/hrrr/exec
LOGS=/opt/hrrr/build-logs
mkdir -p "$EXEC" "$LOGS"

# configure.wps.optim names a jasper include folder of an older machine that
# WCOSS2 does not have either; there the jasper module puts its headers on
# the compiler's search path. CPATH does the same here.
export CPATH="$PREFIX/include${CPATH:+:$CPATH}"

cd "$BASE/hrrr_wps.fd/WPSV3.9.1"
./clean -aa > /dev/null 2>&1 || true
./clean -a  > /dev/null 2>&1 || true
./clean     > /dev/null 2>&1 || true
cp configure.wps.optim configure.wps
./compile > "$LOGS/wps.log" 2>&1 || true
if [[ ! -x ungrib/src/ungrib.exe || ! -x metgrid/src/metgrid.exe ]]; then
  grep -n -B3 -A12 -E "Error|error #|catastrophic|undefined reference" "$LOGS/wps.log" | head -200
  tail -30 "$LOGS/wps.log"
  exit 1
fi
cp ungrib/src/ungrib.exe   "$EXEC/hrrr_wps_ungrib"
cp metgrid/src/metgrid.exe "$EXEC/hrrr_wps_metgrid"
gzip -f "$LOGS/wps.log"
ls -l "$EXEC"
