#!/usr/bin/env bash
# Build NOAA's RDASApp (JEDI for RRFS v2) at a pin, MPAS bundle only, with no
# source edits beyond the ones RDASApp's own build.sh applies by default.
# Runs inside the pin's CONTAINER with a work folder mounted at /work:
#
#   podman run --rm -v <work>:/work <CONTAINER> bash /work/recipe/build_rdasapp.sh \
#       /work/recipe/<pin>.pin <jobs>
#
# RDASApp's build.sh cannot run here unchanged: it loads an Lmod module file
# per NOAA machine and has none for a generic host at this commit. This
# script performs the same steps for "build.sh -m MPAS -t NO -b NO" with the
# container's spack-stack environment in place of that module:
#   * the configure options build.sh adds (MPAS only, GSIbec on, test data
#     off, RDAS tools off for MPAS, the CRTM test-file path, 120 ranks max);
#   * the workaround file copies build.sh makes by default (-w YES);
#   * make in the bundle's build folder.
# JCB (-b NO) is skipped: it only generates test YAMLs from RRFS test data,
# which -t NO leaves out anyway.
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SRC=/work/RDASApp
BUILD=/work/build
LOGS=/work/logs
export TMPDIR="${TMPDIR:-/work/tmp}"
mkdir -p "$LOGS" "$TMPDIR"
export GIT_LFS_SKIP_SMUDGE=1

if [[ ! -d "$SRC/.git" ]]; then
  git init -q "$SRC"
  git -C "$SRC" fetch -q --depth 1 "$RDASAPP_REPO" "$RDASAPP_COMMIT"
  git -C "$SRC" checkout -q FETCH_HEAD
fi
test "$(git -C "$SRC" rev-parse HEAD)" = "$RDASAPP_COMMIT"
# Submodules at the commits RDASApp records (not their branch tips).
git -C "$SRC" submodule update --init --recursive --depth 1 --jobs 8 > "$LOGS/submodules.log" 2>&1 \
  || git -C "$SRC" submodule update --init --recursive --jobs 8 >> "$LOGS/submodules.log" 2>&1
git -C "$SRC" submodule status --recursive > "$LOGS/submodule-commits.txt"
if grep -q '^[+-U]' "$LOGS/submodule-commits.txt"; then
  echo "a submodule differs from the RDASApp pin; refusing a mixed-source build" >&2
  exit 1
fi

# build.sh's default workaround copies (lines 264-293 at this commit).
mkdir -p "$BUILD"; cd "$BUILD"
W=../RDASApp/sorc/_workaround_
S=../RDASApp/sorc
if [[ -d "$W" ]]; then
  for f in CMakeLists.txt:fv3jedi/CMakeLists.txt Fields/fv3jedi_field_mod.f90:fv3jedi/Fields/fv3jedi_field_mod.f90 \
           Geometry/fv3jedi_geom_mod.f90:fv3jedi/Geometry/fv3jedi_geom_mod.f90 \
           IO/FV3Restart/IOFms.h:fv3jedi/IO/FV3Restart/IOFms.h IO/FV3Restart/IOFms.cc:fv3jedi/IO/FV3Restart/IOFms.cc \
           IO/FV3Restart/IOFms.interface.F90:fv3jedi/IO/FV3Restart/IOFms.interface.F90 \
           IO/FV3Restart/IOFms.interface.h:fv3jedi/IO/FV3Restart/IOFms.interface.h \
           IO/FV3Restart/fv3jedi_io_fms2_mod.f90:fv3jedi/IO/FV3Restart/fv3jedi_io_fms2_mod.f90 \
           IO/FV3Restart/module_fv3lam_stats.f90:fv3jedi/IO/FV3Restart/module_fv3lam_stats.f90 \
           IO/FV3Restart/m_TwoPhaseScatterGather.f90:fv3jedi/IO/FV3Restart/m_TwoPhaseScatterGather.f90; do
    cp "$W/fv3-jedi-io/${f%%:*}" "$S/fv3-jedi/src/${f##*:}"
  done
  cp "$W/ufo/CMakeLists.txt" "$S/ufo/src/ufo/operators/sfccorrected/."
  cp "$W"/ufo/EvalSurface* "$S/ufo/src/ufo/operators/sfccorrected/."
  cp "$W"/ufo/ObsSfcCorrected* "$S/ufo/src/ufo/operators/sfccorrected/."
  cp "$W/fv3-jedi/fv3jedi_state_mod.F90" "$S/fv3-jedi/src/fv3jedi/State/."
  cp "$W/fv3-jedi/FieldsMetadataDefault.h" "$S/fv3-jedi/src/fv3jedi/FieldMetadata/."
  cp "$W/ufo/DuplicateThinning.cc" "$S/ufo/src/ufo/filters/."
fi

bash "$(dirname "$0")/verify_sources.sh" "$SRC" > "$LOGS/source-check.txt"

CRTM_DATA=/work/RDASApp/bundle/test-data-release/crtm/2.4.0
# Ubuntu's gcc links with --as-needed by default; NOAA's RHEL-family hosts
# do not. MPAS's libsmiol.so carries no DT_NEEDED entry for PnetCDF, so with
# --as-needed the linker drops the executable's -lpnetcdf (no object file
# names it) and mpas_init_atmosphere fails on every ncmpi_* symbol (MEASURED
# 2026-10-03). --no-as-needed restores the linking NOAA's hosts do.
cmake -DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed" -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-as-needed" \
  -DFV3_DYCORE=OFF -DMPAS_DYCORE=ON -DSKIP_DOWNLOAD_TEST_DATA=ON \
  -DMPIEXEC_EXECUTABLE="$(command -v mpirun)" -DMPIEXEC_NUMPROC_FLAG=-n -DBUILD_GSIBEC=ON \
  -DMACHINE_ID=container -DWORKFLOW_TESTS=OFF -DBUILD_RDAS_TOOLS=OFF \
  -DMPIEXEC_MAX_NUMPROCS:STRING=120 -DBUILD_SUPER_EXE=NO -DBUILD_RRFS_TEST="$BUILD_RRFS_TEST" \
  -DUFO_CRTM_TESTFILES_PATH="$CRTM_DATA" \
  ../RDASApp/bundle > "$LOGS/cmake.log" 2>&1 || { tail -60 "$LOGS/cmake.log"; exit 1; }
make -j "$JOBS" > "$LOGS/make.log" 2>&1 || { grep -n -B3 -A15 -E "[Ee]rror" "$LOGS/make.log" | tail -100; exit 1; }
tail -3 "$LOGS/make.log"
ls bin | grep -E "mpasjedi|mpas_" | head -40
ctest -N > "$LOGS/ctest-list.txt" 2>&1 || true
tail -1 "$LOGS/ctest-list.txt"
