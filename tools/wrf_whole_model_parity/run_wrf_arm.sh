#!/usr/bin/env bash
# run_wrf_arm.sh ROOT CASE_DIR WRF_BUILD ARM LABEL
#   One stock WRF run of one arm on the CPU (serial build), from the per-arm
#   real.exe copy ROOT/inputs/ARM (prep_inputs.py) and ROOT/cases/ARM/namelist.input
#   (arms.py).  WRF_BUILD is a compiled WRF tree (main/wrf.exe, run/ tables);
#   LABEL names the build in the output path ROOT/wrf/ARM/LABEL.
set -euo pipefail
ROOT=$1; CASE=$2; EXE=$3; ARM=$4; LABEL=$5
D=$ROOT/wrf/$ARM/$LABEL
rm -rf "$D"; mkdir -p "$D"; cd "$D"
for f in "$EXE"/run/*; do
  b=$(basename "$f")
  case "$b" in namelist.input|wrf.exe|real.exe|ndown.exe|tc.exe|wrfinput_d01|wrfbdy_d01) ;; *) ln -s "$f" "$b";; esac
done
cp "$EXE/main/wrf.exe" ./wrf.exe   # a private copy: a rebuild elsewhere cannot change it mid-run
ln -s "$ROOT/inputs/$ARM/wrfinput_d01" wrfinput_d01
ln -s "$CASE/wrfbdy_d01" wrfbdy_d01
cp "$ROOT/cases/$ARM/namelist.input" namelist.input
sha256sum wrf.exe "$ROOT/inputs/$ARM/wrfinput_d01" "$CASE/wrfbdy_d01" namelist.input > inputs.sha256
ulimit -s unlimited
export OMP_NUM_THREADS=1
start=$(date +%s)
set +e
nice -n 10 ./wrf.exe > wrf.stdout 2> wrf.stderr
rc=$?
set -e
echo "{\"arm\":\"$ARM\",\"build\":\"$LABEL\",\"rc\":$rc,\"seconds\":$(( $(date +%s) - start ))}" > result.json
tail -1 wrf.stdout
