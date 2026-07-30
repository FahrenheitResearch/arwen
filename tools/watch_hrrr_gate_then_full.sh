#!/usr/bin/env bash
set -Eeuo pipefail

gate_version=${1:?usage: watch_hrrr_gate_then_full.sh GATE_VERSION FULL_VERSION}
full_version=${2:?missing FULL_VERSION}
repo=/workspace/gpuwm-hrrr-native-state-v1
proof=/workspace/gpuwm-hrrr-proof-20260718T00-v1
stage=/workspace/gpuwm-hrrr-proof-staging-v1
gate_job="$stage/jobs/native_two_domain_$gate_version"
gate_out="$proof/native/two-domain-$gate_version"
watch_job="$stage/jobs/watch_${gate_version}_then_${full_version}"
bridge="$proof/native/source-window-f00-f12-v1"
manifest=ec9e5f6013ed297a012fb21291d933063284a089302f85b7b297c906f274884d

[[ -d "$gate_job" ]] || { echo "missing gate job $gate_job" >&2; exit 2; }
[[ ! -e "$watch_job" ]] || { echo "refusing existing watcher $watch_job" >&2; exit 2; }
mkdir -p "$watch_job"
printf '%s\n' "$$" > "$watch_job/pid"

while [[ ! -f "$gate_job/exit_code" ]]; do
    sleep 5
done
if [[ "$(<"$gate_job/exit_code")" != 0 ]]; then
    printf '%s\n' "GATE_FAILED" > "$watch_job/status"
    exit 1
fi
if [[ ! -f "$gate_out/report.json" ]] \
        || ! grep -q '"status": "PASS"' "$gate_out/report.json"; then
    printf '%s\n' "GATE_RECEIPT_MISSING" > "$watch_job/status"
    exit 1
fi

cd "$repo"
bash tools/start_hrrr_two_domain_forecast.sh \
    "$full_version" 43200 "$bridge" "$manifest" 300 \
    > "$watch_job/launch.txt"
printf '%s\n' "FULL_LAUNCHED" > "$watch_job/status"
