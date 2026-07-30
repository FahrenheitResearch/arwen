#!/usr/bin/env bash
# Run the pinned real.exe in a disposable directory to make canonical inputs.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(realpath -- "$SCRIPT_DIR/../..")
BUNDLE=/mnt/c/Users/drew/Downloads/WRF_1974_MP55_reference_bundle
ORACLE_MAIN=/home/drew/wrf-oracle-build/main
ORACLE_CASE=/home/drew/wrf-oracle-build/n1p5-case
WORK_ROOT=/home/drew/n5s-work
WORKDIR=
MET_DIR=
ENV_FILE="$SCRIPT_DIR/wsl-env.sh"
RUN_MINUTES=30
HISTORY_MINUTES=5
REAL_SHA=4b47e80e55009144410b5530e0bbbcd1a5426b427716301a078a39af7b6f4622

usage() {
  cat <<'EOF'
Usage: run_real.sh --met-dir DIR [options]

Options:
  --met-dir DIR          prep_met_em.sh work directory (required).
  --run-minutes N        Forecast duration, 30..75 (default 30).
  --history-minutes N    Common WRF/gpuwm history cadence (default 5).
  --bundle DIR           Read-only reference bundle.
  --oracle-main DIR      Directory containing pinned real.exe.
  --oracle-case DIR      Read-only oracle case/static-table directory.
  --work-root DIR        Disposable parent (default /home/drew/n5s-work).
  --workdir DIR          Explicit new/empty directory under work-root.
  --env-file FILE        Runtime environment profile (default tools/n5s/wsl-env.sh).
  -h, --help             Show this help.

The script SHA-verifies real.exe before execution, unsets the optional
instrumentation dump variable, and writes the five canonical restored inputs,
their digest, logs, and the complete bundle-departure provenance in workdir.
EOF
}

while (($#)); do
  case "$1" in
    --met-dir) MET_DIR=$2; shift 2 ;;
    --run-minutes) RUN_MINUTES=$2; shift 2 ;;
    --history-minutes) HISTORY_MINUTES=$2; shift 2 ;;
    --bundle) BUNDLE=$2; shift 2 ;;
    --oracle-main) ORACLE_MAIN=$2; shift 2 ;;
    --oracle-case) ORACLE_CASE=$2; shift 2 ;;
    --work-root) WORK_ROOT=$2; shift 2 ;;
    --workdir) WORKDIR=$2; shift 2 ;;
    --env-file) ENV_FILE=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$ENV_FILE" ]] || { echo "missing runtime environment file: $ENV_FILE" >&2; exit 2; }
# shellcheck source=/dev/null
source "$ENV_FILE"

[[ -n "$MET_DIR" ]] || { echo "--met-dir is required" >&2; exit 2; }
[[ "$RUN_MINUTES" =~ ^[0-9]+$ ]] && ((RUN_MINUTES >= 30 && RUN_MINUTES <= 75)) || {
  echo "--run-minutes must be an integer in 30..75" >&2; exit 2; }
[[ "$HISTORY_MINUTES" =~ ^[0-9]+$ ]] && ((HISTORY_MINUTES > 0)) || {
  echo "--history-minutes must be positive" >&2; exit 2; }
((RUN_MINUTES % HISTORY_MINUTES == 0)) || {
  echo "history cadence must divide run duration" >&2; exit 2; }
[[ -f "$ORACLE_MAIN/real.exe" ]] || { echo "missing $ORACLE_MAIN/real.exe" >&2; exit 2; }
ACTUAL_SHA=$(sha256sum "$ORACLE_MAIN/real.exe" | awk '{print $1}')
[[ "$ACTUAL_SHA" == "$REAL_SHA" ]] || {
  echo "real.exe SHA-256 mismatch: $ACTUAL_SHA != $REAL_SHA" >&2; exit 2; }

mkdir -p -- "$WORK_ROOT"
WORK_ROOT=$(realpath -- "$WORK_ROOT")
if [[ -z "$WORKDIR" ]]; then
  WORKDIR=$(mktemp -d "$WORK_ROOT/real.XXXXXX")
else
  mkdir -p -- "$WORKDIR"
  WORKDIR=$(realpath -- "$WORKDIR")
  [[ "$WORKDIR" == "$WORK_ROOT"/* ]] || {
    echo "--workdir must be below $WORK_ROOT" >&2; exit 2; }
  [[ -z "$(find "$WORKDIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo "--workdir must be empty: $WORKDIR" >&2; exit 2; }
fi
MET_DIR=$(realpath -- "$MET_DIR")

cd -- "$WORKDIR"
ln -s -- "$ORACLE_MAIN/real.exe" real.exe
for source in "$ORACLE_CASE"/*; do
  [[ -f "$source" || -L "$source" ]] || continue
  name=$(basename -- "$source")
  case "$name" in
    namelist.input|real.exe|wrf.exe|wrfinput_d*|wrfbdy_d01|met_em.*|rsl.*|*.log|*.bin|*.npz)
      continue ;;
  esac
  [[ -e "$name" || -L "$name" ]] || ln -s -- "$source" "$name"
done
for pattern in \
  met_em.d01.1974-04-03_12_00_00.nc \
  met_em.d01.1974-04-03_18_00_00.nc \
  met_em.d02.1974-04-03_12_00_00.nc \
  met_em.d03.1974-04-03_12_00_00.nc \
  met_em.d04.1974-04-03_12_00_00.nc; do
  [[ -f "$MET_DIR/$pattern" ]] || { echo "missing met input $MET_DIR/$pattern" >&2; exit 2; }
  ln -s -- "$MET_DIR/$pattern" "$pattern"
done

# Two-stage namelists: real.exe must ingest BOTH 6-hourly analysis times
# (12Z + 18Z) to build wrfbdy_d01, so its end time is start + interval;
# wrf.exe's forecast namelist bounds the run to RUN_MINUTES.
python3 "$SCRIPT_DIR/namelist.py" \
  --bundle-namelist "$BUNDLE/namelists/namelist.input" \
  --config "$REPO_ROOT/configs/real74_4dom.toml" \
  --run-minutes "$RUN_MINUTES" --history-minutes "$HISTORY_MINUTES" \
  --stage real \
  --output namelist.input --provenance real-namelist-provenance.json
python3 "$SCRIPT_DIR/namelist.py" \
  --bundle-namelist "$BUNDLE/namelists/namelist.input" \
  --config "$REPO_ROOT/configs/real74_4dom.toml" \
  --run-minutes "$RUN_MINUTES" --history-minutes "$HISTORY_MINUTES" \
  --stage forecast \
  --output namelist.input.forecast \
  --provenance forecast-namelist-provenance.json

# The inserted instrumentation is fully guarded by this variable.  Keeping it
# absent avoids an otherwise ~844 MB first-force dump in every ensemble run.
unset GPUWM_WRF_ORACLE_DUMP
set +e
./real.exe >real.stdout.log 2>real.stderr.log
STATUS=$?
set -e
printf '%s\n' "$STATUS" >exit.status
if ((STATUS != 0)); then
  echo "real.exe failed with status $STATUS in $WORKDIR" >&2
  exit "$STATUS"
fi
for name in wrfinput_d01 wrfinput_d02 wrfinput_d03 wrfinput_d04 wrfbdy_d01; do
  [[ -s "$name" ]] || { echo "real.exe did not produce $name" >&2; exit 1; }
done

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$WORKDIR" <<'PY'
import json, pathlib, sys
from gpuwm.verify.n5s_common import restored_input_sha256, sha256_file
root = pathlib.Path(sys.argv[1])
digest = restored_input_sha256(root)
(root / "restored_input_sha256.txt").write_text(digest + "\n", encoding="utf-8")
payload = {
    "schema": 1, "restored_input_sha256": digest,
    "artifacts": {name: {"sha256": sha256_file(root / name),
                          "bytes": (root / name).stat().st_size}
                  for name in ("wrfbdy_d01", "wrfinput_d01", "wrfinput_d02",
                               "wrfinput_d03", "wrfinput_d04")},
}
(root / "restored-input-manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

printf '%s\n' "$WORKDIR"
