#!/usr/bin/env bash
# Run one controller step under tmux with durable logs and an exit record.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE="$SCRIPT_DIR/wsl-env.sh"
LOG_FILE=
STATUS_FILE=

usage() {
  cat <<'EOF'
Usage: tmux_worker.sh --log FILE --status FILE [--env-file FILE] -- COMMAND [ARG...]

The worker sources the checked WPS/WRF environment, redirects the command's
stdout/stderr to FILE without a pipe, records its exact status, and returns it.
EOF
}

while (($#)); do
  case "$1" in
    --env-file) ENV_FILE=$2; shift 2 ;;
    --log) LOG_FILE=$2; shift 2 ;;
    --status) STATUS_FILE=$2; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$LOG_FILE" && -n "$STATUS_FILE" && $# -gt 0 ]] || {
  echo "--log, --status, and a command are required" >&2
  usage >&2
  exit 2
}
for output in "$LOG_FILE" "$STATUS_FILE"; do
  mkdir -p -- "$(dirname -- "$output")"
done

: >"$STATUS_FILE"
record_status() {
  local worker_status=$?
  trap - EXIT
  printf '%s\n' "$worker_status" >"$STATUS_FILE"
  exit "$worker_status"
}
trap record_status EXIT
exec >"$LOG_FILE" 2>&1

[[ -f "$ENV_FILE" ]] || { echo "missing runtime environment file: $ENV_FILE" >&2; exit 2; }
# shellcheck source=/dev/null
source "$ENV_FILE"
"$@"
