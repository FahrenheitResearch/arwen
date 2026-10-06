#!/usr/bin/env bash
# Build GSI (cloud analysis compiled in), EnKF, the EnKF preprocessor and
# initialens from the tag's sorc/hrrr_gsi.fd with no source edits. This is the
# tag's own sorc/build_hrrr_gsi.sh with the WCOSS2 module loads replaced by
# env.sh. Runs inside the "tools" stage.
set -euo pipefail
JOBS="${1:-8}"
BASE=/opt/hrrr/sorc
export BASE
EXEC=/opt/hrrr/exec
LOGS=/opt/hrrr/build-logs
mkdir -p "$EXEC" "$LOGS"

# sorc/build_hrrr_gsi.sh's cmake line, verbatim.
cd "$BASE/hrrr_gsi.fd"
rm -fr build && mkdir build && cd build
cmake -DENKF_MODE=WRF -DBUILD_ENKF_PREPROCESS_ARW=ON -DBUILD_GSDCLOUD_ARW=ON ../. > "$LOGS/gsi-cmake.log" 2>&1 \
  || { tail -80 "$LOGS/gsi-cmake.log"; exit 1; }
grep -E "Host is set|Setting|BUILD_CORELIBS|MPI version|netcdf_libs|compiler identification" "$LOGS/gsi-cmake.log" || true

# The tag runs a serial make. A parallel make alone does not finish this
# tree: the cloud analysis sources (gsdcloudanalysis.F90 and its siblings)
# sit wholly inside "#ifdef RR_CLOUDANALYSIS", and that define reaches the
# compiler through COMPILE_FLAGS, which CMake's Fortran dependency scanner
# does not read. Their "use" lines are therefore not dependencies, and a
# parallel make compiles them before gridmod and the other modules exist
# ("error #7002: Error in opening the compiled module file", MEASURED
# 2026-10-03 with 12 jobs). Two parallel passes that keep going build
# everything whose modules are ready; the closing pass is the tag's own
# serial make and must end clean.
make -j"$JOBS" -k > "$LOGS/gsi-make-parallel-1.log" 2>&1 || true
make -j"$JOBS" -k > "$LOGS/gsi-make-parallel-2.log" 2>&1 || true
make > "$LOGS/gsi-make.log" 2>&1 \
  || { grep -n -B5 -A25 -E "[Ee]rror" "$LOGS/gsi-make.log" | tail -160; exit 1; }
tail -5 "$LOGS/gsi-make.log"
gzip -f "$LOGS/gsi-make-parallel-1.log" "$LOGS/gsi-make-parallel-2.log"
cp bin/gsi.x        "$EXEC/hrrr_gsi"
cp bin/enkf_wrf.x   "$EXEC/hrrr_enkf"
cp bin/enspreproc.x "$EXEC/hrrr_process_enkf"
cp bin/initialens.x "$EXEC/hrrr_initialens"
gzip -f "$LOGS/gsi-make.log"
# The build tree is gigabytes of objects; the image keeps the executables
# and the logs, and the tracked tree stays as the tag ships it.
cd "$BASE/hrrr_gsi.fd" && rm -fr build
ls -l "$EXEC"
