#!/usr/bin/env bash
set -Eeuo pipefail

repo=/workspace/gpuwm-hrrr-native-state-v1
proof=/workspace/gpuwm-hrrr-proof-20260718T00-v1
stage=/workspace/gpuwm-hrrr-proof-staging-v1
watch_job="$stage/jobs/native_500_suite_watcher_v1"
gate_job="$stage/jobs/native_500_pipeline_gate5m-v1"
gate_report="$proof/native/single-500x500-gate5m-v1/report.json"
full_job="$stage/jobs/native_500_pipeline_full12h-compute-v1"
full_report="$proof/native/single-500x500-full12h-compute-v1/report.json"
full_bridge="$proof/native/source-window-f00-f12-500pipeline-full12h-compute-v1"
manifest=ec9e5f6013ed297a012fb21291d933063284a089302f85b7b297c906f274884d

[[ ! -e "$watch_job" ]] || {
    echo "refusing existing watcher $watch_job" >&2; exit 2; }
mkdir -p "$watch_job"

nohup bash -c '
    set -Eeuo pipefail
    repo=$1; proof=$2; stage=$3; watch_job=$4; gate_job=$5
    gate_report=$6; full_job=$7; full_report=$8; full_bridge=$9
    manifest=${10}
    while [[ ! -f "$gate_job/exit_code" ]]; do sleep 10; done
    [[ "$(head -1 "$gate_job/exit_code")" == 0 ]]
    jq -e '\''
        .status == "PASS" and
        .run_seconds == 300 and
        .health.initial.ok == true and
        .health.final.ok == true and
        .health.final_stability.nan == false and
        .health.final_stability.cfl <= .health.limits.max_cfl
    '\'' "$gate_report" >/dev/null
    "$repo/tools/start_hrrr_500_pipeline.sh" \
        full12h-compute-v1 43200 none 8
    while [[ ! -f "$full_job/exit_code" ]]; do sleep 20; done
    [[ "$(head -1 "$full_job/exit_code")" == 0 ]]
    jq -e --arg manifest "$manifest" '\''
        .status == "PASS" and
        .run_seconds == 43200 and
        .io_mode == "none" and
        .health.initial.ok == true and
        .health.final.ok == true and
        .health.final_stability.nan == false and
        .input.bridge_manifest_sha256 == $manifest
    '\'' "$full_report" >/dev/null
    "$repo/tools/start_hrrr_500_sealed.sh" \
        full12h-io-v1 43200 hourly "$full_bridge" "$manifest"
    printf "PASS\n" > "$watch_job/launch_chain_complete.tmp"
    mv "$watch_job/launch_chain_complete.tmp" \
        "$watch_job/launch_chain_complete"
' _ "$repo" "$proof" "$stage" "$watch_job" "$gate_job" \
    "$gate_report" "$full_job" "$full_report" "$full_bridge" \
    "$manifest" > "$watch_job/watcher.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$watch_job/pid"
echo "STARTED pid=$pid watcher=$watch_job"
