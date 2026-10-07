#!/usr/bin/env bash
# Box only. Source arrives as a local export, never a fetch from a private repo.
# The packed DA deck (docs: DECK-RECENT, lane/da-rerun-286). Stage beside
# box_env.sh, mps_owned.sh and run-recent-packed.sh under $DA_ROOT.
set -Eeuo pipefail
DA_ROOT=${DA_ROOT:-/workspace/da-rerun-286}
DA_ENGINE=${DA_ENGINE:-$DA_ROOT/engine}
REV=${1:?full engine export SHA}
[[ "$REV" =~ ^[0-9a-f]{40}$ ]]
test -f "$DA_ENGINE/.engine-export-sha"
test "$(cat "$DA_ENGINE/.engine-export-sha")" = "$REV"
mkdir -p "$DA_ROOT/logs" "$DA_ROOT/tmp" "$DA_ROOT/cache/cupy" "$DA_ROOT/tables/thompson" "$DA_ROOT/tables/wif"
command -v cargo
command -v gcc
# The thompson-mp28 wrf_39_noaa profile reads the WRF 3.9 fork tables, which
# no public publisher carries: without a staged set the runtime builds them
# from the pinned Fortran source, and a box without gfortran failed the
# first packed member step (5090 box, 2026-10-05). Install it up front.
if ! command -v gfortran > /dev/null; then
  if [[ $(id -u) == 0 ]] && command -v apt-get > /dev/null; then
    apt-get update > "$DA_ROOT/logs/apt-gfortran.log" 2>&1 || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y gfortran >> "$DA_ROOT/logs/apt-gfortran.log" 2>&1 || true
  fi
  command -v gfortran > /dev/null || { echo 'gfortran is required to build the wrf_39_noaa Thompson fork tables (or stage them: gpuwm fetch-tables --thompson-fork-only --from DIR)' >&2; exit 2; }
fi
command -v pkg-config
command -v python3
command -v nvidia-smi
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
export GPUWM_BRIDGE_SOURCE_REV="$REV" CARGO_BUILD_JOBS=48 TMPDIR="$DA_ROOT/tmp"
python3 -m venv "$DA_ROOT/venv"
"$DA_ROOT/venv/bin/python" -m pip install --upgrade pip > "$DA_ROOT/logs/pip.log" 2>&1
"$DA_ROOT/venv/bin/python" -m pip install -e "$DA_ENGINE/gpuwm-data" -e "$DA_ENGINE[gpu-cu13]" 'cupy-cuda13x[ctk]==14.2.0' >> "$DA_ROOT/logs/pip.log" 2>&1
(cd "$DA_ENGINE/tools/grib1_bridge" && cargo build --release --locked --offline --bins --lib -j 16) > "$DA_ROOT/logs/build-grib.log" 2>&1 & P_GRIB=$!
(cd "$DA_ENGINE/tools/rustwx" && cargo build --release --locked --offline -p netcdf-writer -p static-fields -p rw-netcdf -p rw-fetch -p rw-wrfbatch -p obs-score -p rw-obs -p rw-nexrad -j 16) > "$DA_ROOT/logs/build-rustwx.log" 2>&1 & P_RUSTWX=$!
(cd "$DA_ENGINE/tools/rw_wps" && cargo build --release --locked --offline -j 16) > "$DA_ROOT/logs/build-rwwps.log" 2>&1 & P_WPS=$!
RC=0
wait "$P_GRIB" || RC=1
wait "$P_RUSTWX" || RC=1
wait "$P_WPS" || RC=1
[[ "$RC" == 0 ]]
(cd "$DA_ENGINE/tools/region_global_dealias" && cargo build --release --locked --offline -j 16) > "$DA_ROOT/logs/build-radar-dealias.log" 2>&1
source "$(dirname "$0")/box_env.sh"
unset GPUWM_NO_LOCAL_GPU CUDA_VISIBLE_DEVICES
# Classic tables (copied from the package into GPUWM_THOMPSON_TABLE_ROOT;
# an engine without that completion needs them staged by hand first), the
# WIF climatology, and the wrf_39_noaa fork set the deck's profile reads.
"$DA_VENV/bin/python" -m gpuwm.cli fetch-tables --wif --thompson-fork > "$DA_ROOT/logs/tables.log" 2>&1
# The doctor is a report of the whole install, not this deck's gate: it
# exits 1 for components the deck never uses (render extra, TUI, GOES and
# ODIM doors, MPAS engines, the default WPS_GEOG path) and stopped the
# build there on 2026-10-05. Keep its exit code in the log and go on.
DOCTOR_RC=0
"$DA_VENV/bin/python" -m gpuwm.cli doctor > "$DA_ROOT/logs/doctor.log" 2>&1 || DOCTOR_RC=$?
printf 'doctor exit %s (report only)\n' "$DOCTOR_RC" >> "$DA_ROOT/logs/doctor.log"
"$DA_VENV/bin/python" - <<'PY' > "$DA_ROOT/logs/environment.json"
import importlib.metadata as metadata, json, sys
print(json.dumps({'python': sys.version, 'packages': {
    x: metadata.version(x) for x in ['gpuwm', 'gpuwm-data', 'cupy-cuda13x', 'numpy', 'netCDF4']}}, indent=2))
PY
find "$DA_ENGINE/tools/grib1_bridge/target/release" "$DA_ENGINE/tools/rustwx/target/release" "$DA_ENGINE/tools/rw_wps/target/release" "$DA_ENGINE/tools/region_global_dealias/target/release" -maxdepth 1 -type f -executable -print0 | sort -z | xargs -0 sha256sum > "$DA_ROOT/logs/native-binaries.sha256"
printf 'Build complete from %s\n' "$REV"
