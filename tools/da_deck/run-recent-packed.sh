#!/usr/bin/env bash
# Rental only. Both recent-case seeds and paired arms share one time limit.
set -Eeuo pipefail
if [[ ${1:-} == --inner ]]; then
  shift
  RECENT_MANIFEST=${1:?completed public run manifest}; shift
  cd "$DA_ENGINE"
  python -m tools.da_recent_run --manifest "$RECENT_MANIFEST" --seed 20261003 --arm both --device-uuids "$@" --execute
  python -m tools.da_recent_run --manifest "$RECENT_MANIFEST" --seed 20261004 --arm both --device-uuids "$@" --execute
  exit
fi
RECENT_MANIFEST=${1:?completed public run manifest}
DA_ROOT=${DA_ROOT:-/workspace/da-rerun-286}
source "$DA_ROOT/box_env.sh"
mapfile -t RECENT_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
# Any number of cards: the packed step places the roster in waves.
test "${#RECENT_UUIDS[@]}" -ge 1
RECENT_TAG=$(date -u +%Y%m%dT%H%M%SZ)-$$
RECENT_MPS="$DA_ROOT/mps-recent-$RECENT_TAG"
mkdir -p "$DA_ROOT/logs" "$DA_ROOT/receipts"
RECENT_RENTAL_STARTED=${DA_RECENT_RENTAL_STARTED:-$(date +%s)}
[[ "$RECENT_RENTAL_STARTED" =~ ^[0-9]+$ ]]
RECENT_RUN_BUDGET=$((RECENT_RENTAL_STARTED + 7200 - $(date +%s) - 1200 - 30))
if (( RECENT_RUN_BUDGET > 4800 )); then RECENT_RUN_BUDGET=4800; fi
if (( RECENT_RUN_BUDGET <= 0 )); then
  echo 'No rental time remains after reserving scoring and cancellation' >&2
  exit 124
fi
bash "$DA_ROOT/mps_owned.sh" start "$RECENT_MPS"
source "$RECENT_MPS/client-env.sh"
export CUDA_VISIBLE_DEVICES="${RECENT_UUIDS[0]}"
RECENT_STARTED=$(date +%s)
cleanup() {
  RECENT_RC=$?
  trap - EXIT
  bash "$DA_ROOT/mps_owned.sh" stop "$RECENT_MPS" > "$DA_ROOT/logs/recent-mps-stop-$RECENT_TAG.log" 2>&1 || true
  python - "$DA_ROOT/receipts/recent-$RECENT_TAG.json" "$RECENT_RC" "$RECENT_STARTED" "$RECENT_MANIFEST" "$RECENT_RUN_BUDGET" <<'PY'
import json, pathlib, sys, time
path, code, started, manifest, budget = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps({"manifest": manifest, "exit_code": int(code),
    "wall_seconds": time.time()-int(started), "shared_budget_seconds": int(budget),
    "status": "commands-complete" if int(code) == 0 else "incomplete",
    "scientific_gate": "pending"}, indent=2)+"\n")
PY
  exit "$RECENT_RC"
}
trap cleanup EXIT
timeout --signal=INT --kill-after=30s "${RECENT_RUN_BUDGET}s" bash "$DA_ROOT/run-recent-packed.sh" --inner "$RECENT_MANIFEST" "${RECENT_UUIDS[@]}" \
  > "$DA_ROOT/logs/recent-$RECENT_TAG.log" 2>&1
