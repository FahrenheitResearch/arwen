#!/usr/bin/env bash
# Run one case script inside an image, detached, with a log and a done
# marker, from the host side.
#
#   in_container.sh <image> <log> <done marker> <memory limit> -- <script> <args...>
#
# Mounts, by fixed name: $CASE_ROOT (the case and fix data folder) at /data,
# this folder at /scripts, read-only. Paths given to the script must be under
# /data. The container's peak memory is appended to the log. Host variables
# named in PASS_ENV (space separated) are passed into the container.
set -euo pipefail
image="$1"; log="$2"; marker="$3"; memory="$4"; shift 4
[[ "$1" == "--" ]] && shift
here="$(cd "$(dirname "$0")" && pwd)"
: "${CASE_ROOT:?set CASE_ROOT to the folder that holds the case data}"
rm -f "$marker"
env_args=""
for name in ${PASS_ENV:-}; do env_args+=" -e $name"; done
inner='. /opt/gsi-env.sh; started=$SECONDS; bash "$@"; rc=$?;'
inner+=' echo "elapsed_s=$((SECONDS - started)) rc=$rc";'
inner+=' cat /sys/fs/cgroup/memory.peak 2>/dev/null | sed "s/^/memory_peak_bytes=/"; exit $rc'
setsid nohup bash -c '
  podman run --rm --shm-size=16g --memory="$1" $8 \
    -v "$2":/data -v "$3":/scripts:ro "$4" bash -c "$5" inner "${@:9}" > "$6" 2>&1
  echo "rc=$?" > "$7"
' launcher "$memory" "$CASE_ROOT" "$here" "$image" "$inner" "$log" "$marker" "$env_args" "$@" \
  < /dev/null > /dev/null 2>&1 &
echo "started: $image $*"
