#!/usr/bin/env bash
# Build the tag's WRF-ARW (sorc/hrrr_wrfarw.fd/WRFV3.9) the way
# sorc/build_hrrr_wrfarw.sh does, then the two tools that link pieces of the
# built WRF tree: update_bc and ref2tten. No source edits: the tag's own
# configure.wrf.useme is copied into place as its build script does, and the
# WCOSS2 library paths that file names exist in this image
# (build_nceplibs.sh).
set -euo pipefail
JOBS="${1:-8}"
BASE=/opt/hrrr/sorc
export BASE
EXEC=/opt/hrrr/exec
LOGS=/opt/hrrr/build-logs
mkdir -p "$EXEC" "$LOGS"

cd "$BASE/hrrr_wrfarw.fd/WRFV3.9"
./clean -aa > /dev/null 2>&1 || true
./clean -a  > /dev/null 2>&1 || true
./clean     > /dev/null 2>&1 || true
cp configure.wrf.useme configure.wrf
export PNETCDF_QUILT=1
export WRFIO_NCD_LARGE_FILE_SUPPORT=1
export WRF_DFI_RADAR=1
export WRF_SMOKE=1
# The tag runs ./compile -j 1. A parallel compile alone does not finish this
# tree: main/real_em.o was compiled while dyn_em/module_initialize_real.o was
# still being built ("error #7002: Error in opening the compiled module
# file", MEASURED 2026-10-03 with 12 jobs), so real.exe never linked. The
# parallel pass builds everything it can; the closing pass is the tag's own
# serial compile, which rebuilds only what is missing.
./compile -j "$JOBS" em_real > "$LOGS/wrfarw-parallel.log" 2>&1 || true
gzip -f "$LOGS/wrfarw-parallel.log"
./compile -j 1 em_real > "$LOGS/wrfarw.log" 2>&1 || true
if [[ ! -x main/real.exe || ! -x main/wrf.exe ]]; then
  grep -n -B3 -A12 -E "Error|error #|catastrophic" "$LOGS/wrfarw.log" | head -200
  tail -30 "$LOGS/wrfarw.log"
  exit 1
fi
cp main/real.exe "$EXEC/hrrr_wrfarw_real"
cp main/wrf.exe  "$EXEC/hrrr_wrfarw_fcst"
gzip -f "$LOGS/wrfarw.log"

for tool in update_bc ref2tten; do
  cd "$BASE/hrrr_${tool}.fd"
  make clean
  make > "$LOGS/${tool}.log" 2>&1 || { tail -80 "$LOGS/${tool}.log"; exit 1; }
  make clean
done
ls -l "$EXEC"
