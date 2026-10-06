#!/usr/bin/env bash
# Build the tag's observation preprocessors that need only NCEPLIBS:
# process_mosaic (MRMS reflectivity to the model grid), process_cloud (NASA
# LaRC cloud product) and process_lightning. Each pass is the tag's own
# sorc/build_hrrr_<tool>.sh with the WCOSS2 module loads replaced by env.sh.
set -euo pipefail
BASE=/opt/hrrr/sorc
export BASE
EXEC=/opt/hrrr/exec
LOGS=/opt/hrrr/build-logs
mkdir -p "$EXEC" "$LOGS"
for tool in process_mosaic process_cloud process_lightning; do
  cd "$BASE/hrrr_${tool}.fd"
  make clean
  make > "$LOGS/${tool}.log" 2>&1 || { tail -80 "$LOGS/${tool}.log"; exit 1; }
  # Objects and modules go; the tracked tree stays as the tag ships it.
  make clean
done
ls -l "$EXEC"
