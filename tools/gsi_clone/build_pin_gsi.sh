#!/usr/bin/env bash
# Build GSI and EnKF at a pin's commit, once per EnKF mode the pin lists,
# with no source edits. Runs inside Dockerfile.pin after build_pin_libs.sh.
# The cmake line is GSI's own (ush/build.sh) with the pin's GSI_MODE and
# ENKF_MODE, which is what the pinned workflow passes.
set -euo pipefail
pinfile="$1"; JOBS="${2:-8}"
# shellcheck disable=SC1090
. "$pinfile"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be positive to avoid an unlimited make build" >&2; exit 1; }
limit=$(nproc)
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r quota period < /sys/fs/cgroup/cpu.max
  if [[ "$quota" != max ]]; then limit=$((quota / period)); ((limit > 0)) || limit=1; fi
fi
((JOBS <= limit)) || JOBS=$limit
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SRC=$PIN_ROOT/gsi
EXEC=$PIN_ROOT/exec
LOGS=$PIN_ROOT/build-logs
mkdir -p "$EXEC" "$LOGS"

# CRTM finds OpenMP when it is built (its CMakeLists calls find_package(OpenMP)
# without REQUIRED) and exports a link to OpenMP::OpenMP_Fortran. GSI, built
# with OPENMP off as the pinned workflow builds it, never defines that
# imported target, and CMake's generate step stops on every GSI and EnKF
# target (MEASURED 2026-10-03). This include defines the target after each
# project() call; GSI's own sources get no OpenMP flag from it, and the
# executables link the OpenMP runtime only because CRTM's objects need it.
# GSI's C object library (crc32_c.c) includes zlib.h but is given no zlib
# include directory by GSI's CMake; on WCOSS2 the Cray compiler wrappers add
# every loaded module's include folder, zlib's among them. CPATH does the same
# here for the pin's own library prefix (MEASURED 2026-10-03: "cannot open
# source file zlib.h" without it).
export CPATH="$PIN_LIBS/include${CPATH:+:$CPATH}"
# The per-library prefixes build_pin_libs.sh wrote.
CMAKE_PREFIX_PATH="$(tr ";" ":" < "$PIN_LIBS/cmake-prefixes.txt")"   # the environment form is colon-separated
export CMAKE_PREFIX_PATH
: "${TMPDIR:?set TMPDIR inside the build workspace}"
mkdir -p "$TMPDIR"
openmp_include="$TMPDIR/gsi-pin-openmp-target.cmake"
echo 'find_package(OpenMP COMPONENTS Fortran)' > "$openmp_include"

for mode in $ENKF_MODES; do
  build="$TMPDIR/gsi-build-$mode"
  rm -rf "$build"
  # shellcheck disable=SC2086
  cmake -S "$SRC" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PIN_ROOT/install-$mode" \
    -DGSI_MODE="$GSI_MODE" -DENKF_MODE="$mode" -DCMAKE_PROJECT_INCLUDE="$openmp_include" $GSI_CMAKE_EXTRA > "$LOGS/cmake-$mode.log" 2>&1 \
    || { tail -60 "$LOGS/cmake-$mode.log"; exit 1; }
  grep -E "GSI_MODE|ENKF_MODE|USE_GSDCLOUD|OPENMP|ENABLE_MKL|BUILD_MGBF|compiler identification" "$LOGS/cmake-$mode.log" || true
  make -C "$build" -j"$JOBS" > "$LOGS/make-$mode.log" 2>&1 \
    || { grep -n -B5 -A20 -E "[Ee]rror" "$LOGS/make-$mode.log" | tail -120; exit 1; }
  make -C "$build" install > "$LOGS/install-$mode.log" 2>&1
  lower=$(tr '[:upper:]' '[:lower:]' <<< "$mode")
  cp "$PIN_ROOT/install-$mode/bin/gsi.x" "$EXEC/gsi.x.enkf-$lower-build"
  cp "$PIN_ROOT/install-$mode/bin/enkf.x" "$EXEC/enkf_$lower.x" 2>/dev/null \
    || cp "$build"/bin/enkf*.x "$EXEC/enkf_$lower.x"
  gzip -f "$LOGS/make-$mode.log"
  rm -rf "$build"
done
test -z "$(git -C "$SRC" status --porcelain --untracked-files=no)"
ls -l "$EXEC"
sha256sum "$EXEC"/* > "$PIN_ROOT/EXEC.sha256"
