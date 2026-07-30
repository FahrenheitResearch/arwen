#!/usr/bin/env bash
# Run the unperturbed CPU-WRF baseline and one-ULP member ensemble.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(realpath -- "$SCRIPT_DIR/../..")
ORACLE_MAIN=/home/drew/wrf-oracle-build/main
ORACLE_CASE=/home/drew/wrf-oracle-build/n1p5-case
RESTORED=
OUTPUT=
REGISTRATION=
MEMBERS=4
PARALLEL=1
RUN_MINUTES=30
HISTORY_MINUTES=5
ENV_FILE="$SCRIPT_DIR/wsl-env.sh"
WRF_SHA=f0fb585bf37b72fbdcece562047934cb8386db3958f153d6e4e6876e5fd997ac

usage() {
  cat <<'EOF'
Usage: run_ensemble.sh --restored-inputs DIR --output DIR [options]

Options:
  --registration FILE   Pre-look N5S registration JSON. If omitted, create it
                        from --run-minutes/--history-minutes before any run.
  --members M           Perturbed CPU members, 3 or 4 (default 4).
  --parallel N          Opt-in independent CPU-WRF concurrency (default 1).
  --run-minutes N       Registered duration, 30..75 (default 30).
  --history-minutes N   Registered common cadence (default 5).
  --oracle-main DIR     Pinned wrf.exe directory.
  --oracle-case DIR     Read-only static-table/case directory.
  --env-file FILE       Runtime environment profile (default tools/n5s/wsl-env.sh).
  -h, --help            Show this help.

CONTROLLER POLICY: no GPU work may run concurrently with this script, even
when --parallel is used.  The unperturbed/member WRF jobs are isolated in
fresh directories; parallel mode changes CPU-WRF concurrency only.

The optional instrumentation dump is disabled by unsetting
GPUWM_WRF_ORACLE_DUMP.  Every run retains stdout/stderr, rsl files, and an
exit-status record; the top-level exit-status-ledger.json summarizes them.
EOF
}

while (($#)); do
  case "$1" in
    --restored-inputs) RESTORED=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --registration) REGISTRATION=$2; shift 2 ;;
    --members) MEMBERS=$2; shift 2 ;;
    --parallel) PARALLEL=$2; shift 2 ;;
    --run-minutes) RUN_MINUTES=$2; shift 2 ;;
    --history-minutes) HISTORY_MINUTES=$2; shift 2 ;;
    --oracle-main) ORACLE_MAIN=$2; shift 2 ;;
    --oracle-case) ORACLE_CASE=$2; shift 2 ;;
    --env-file) ENV_FILE=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$ENV_FILE" ]] || { echo "missing runtime environment file: $ENV_FILE" >&2; exit 2; }
# shellcheck source=/dev/null
source "$ENV_FILE"

[[ -n "$RESTORED" && -n "$OUTPUT" ]] || {
  echo "--restored-inputs and --output are required" >&2; exit 2; }
[[ "$MEMBERS" =~ ^[0-9]+$ ]] && ((MEMBERS >= 3 && MEMBERS <= 4)) || {
  echo "--members must be 3 or 4" >&2; exit 2; }
[[ "$PARALLEL" =~ ^[0-9]+$ ]] && ((PARALLEL >= 1)) || {
  echo "--parallel must be positive" >&2; exit 2; }
RESTORED=$(realpath -- "$RESTORED")
[[ -f "$ORACLE_MAIN/wrf.exe" ]] || { echo "missing $ORACLE_MAIN/wrf.exe" >&2; exit 2; }
ACTUAL_SHA=$(sha256sum "$ORACLE_MAIN/wrf.exe" | awk '{print $1}')
[[ "$ACTUAL_SHA" == "$WRF_SHA" ]] || {
  echo "wrf.exe SHA-256 mismatch: $ACTUAL_SHA != $WRF_SHA" >&2; exit 2; }
# namelist.input in the restored dir is the REAL-stage (6 h) namelist;
# wrf.exe members must run the bounded FORECAST namelist (verify3 D1).
for name in wrfinput_d01 wrfinput_d02 wrfinput_d03 wrfinput_d04 wrfbdy_d01 \
            namelist.input namelist.input.forecast; do
  [[ -f "$RESTORED/$name" ]] || { echo "missing restored artifact $name" >&2; exit 2; }
done

mkdir -p -- "$OUTPUT"
OUTPUT=$(realpath -- "$OUTPUT")
[[ -z "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
  echo "--output must be empty: $OUTPUT" >&2; exit 2; }
if [[ -z "$REGISTRATION" ]]; then
  REGISTRATION="$OUTPUT/n5s-preregistration.json"
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m \
    gpuwm.verify.n5s_metrics register --run-minutes "$RUN_MINUTES" \
    --history-minutes "$HISTORY_MINUTES" --output "$REGISTRATION"
else
  REGISTRATION=$(realpath -- "$REGISTRATION")
  cp -- "$REGISTRATION" "$OUTPUT/n5s-preregistration.json"
  REGISTRATION="$OUTPUT/n5s-preregistration.json"
fi

prepare_run() {
  local run_dir=$1
  local ordinal=${2:-}
  mkdir -- "$run_dir"
  ln -s -- "$ORACLE_MAIN/wrf.exe" "$run_dir/wrf.exe"
  for source in "$ORACLE_CASE"/*; do
    [[ -f "$source" || -L "$source" ]] || continue
    local name
    name=$(basename -- "$source")
    case "$name" in
      namelist.input|real.exe|wrf.exe|wrfinput_d*|wrfbdy_d01|met_em.*|rsl.*|*.log|*.bin|*.npz)
        continue ;;
    esac
    [[ -e "$run_dir/$name" || -L "$run_dir/$name" ]] || \
      ln -s -- "$source" "$run_dir/$name"
  done
  cp -- "$RESTORED/namelist.input.forecast" "$run_dir/namelist.input"
  cp -- "$REGISTRATION" "$run_dir/n5s-preregistration.json"
  cp -- "$RESTORED/wrfbdy_d01" "$run_dir/wrfbdy_d01"
  for domain in 02 03 04; do
    cp -- "$RESTORED/wrfinput_d${domain}" "$run_dir/wrfinput_d${domain}"
  done
  if [[ -z "$ordinal" ]]; then
    cp -- "$RESTORED/wrfinput_d01" "$run_dir/wrfinput_d01"
  else
    python3 "$SCRIPT_DIR/perturb_ulp.py" \
      "$RESTORED/wrfinput_d01" "$run_dir/wrfinput_d01" \
      --field T --member-ordinal "$ordinal" \
      --record "$run_dir/perturbation.json"
  fi
  cp -- "$RESTORED/restored_input_sha256.txt" "$run_dir/restored_input_sha256.txt"
}

prepare_run "$OUTPUT/unperturbed"
RUN_DIRS=("$OUTPUT/unperturbed")
for ((index=0; index<MEMBERS; index++)); do
  printf -v label 'member-%02d' "$index"
  prepare_run "$OUTPUT/$label" "$index"
  RUN_DIRS+=("$OUTPUT/$label")
done

run_one() {
  local run_dir=$1
  cd -- "$run_dir"
  unset GPUWM_WRF_ORACLE_DUMP
  set +e
  ./wrf.exe >wrf.stdout.log 2>wrf.stderr.log
  local status=$?
  set -e
  printf '%s\n' "$status" >exit.status
  return 0
}

for ((offset=0; offset<${#RUN_DIRS[@]}; offset+=PARALLEL)); do
  pids=()
  for ((slot=0; slot<PARALLEL && offset+slot<${#RUN_DIRS[@]}; slot++)); do
    run_one "${RUN_DIRS[offset+slot]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
  "$OUTPUT" "${RUN_DIRS[@]}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
failed = False
for raw in sys.argv[2:]:
    run = pathlib.Path(raw)
    status = int((run / "exit.status").read_text(encoding="utf-8").strip())
    failed |= status != 0
    rows.append({
        "id": run.name, "exit_status": status,
        "stdout": str((run / "wrf.stdout.log").relative_to(root)),
        "stderr": str((run / "wrf.stderr.log").relative_to(root)),
        "rsl_files": sorted(str(path.relative_to(root)) for path in run.glob("rsl.*")),
    })
(root / "exit-status-ledger.json").write_text(
    json.dumps({"schema": 1, "runs": rows, "all_succeeded": not failed},
               indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(1 if failed else 0)
PY

printf '%s\n' "$OUTPUT"
