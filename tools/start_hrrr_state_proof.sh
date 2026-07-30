#!/usr/bin/env bash
set -Eeuo pipefail

version=${1:?usage: start_hrrr_state_proof.sh VERSION}
repo=/workspace/gpuwm-hrrr-native-state-v1
proof=/workspace/gpuwm-hrrr-proof-20260718T00-v1
stage=/workspace/gpuwm-hrrr-proof-staging-v1
job="$stage/jobs/native_state_d01_f00f01_$version"
output="$proof/native/state-proof-d01-f00f01-$version.json"
timing="$proof/native/state-proof-d01-f00f01-$version.time.txt"

[[ ! -e "$job" ]] || { echo "refusing existing job $job" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "refusing existing output $output" >&2; exit 2; }
mkdir -p "$job"

nohup bash -c '
    set +e
    repo=$1
    proof=$2
    stage=$3
    job=$4
    output=$5
    timing=$6
    cd "$repo"
    env PYTHONPATH=. /usr/bin/time -v -o "$timing" \
        /workspace/gpuwm-physics-install/venv/bin/python \
        tools/hrrr_state_proof.py \
        --bridge "$proof/native/source-window-v1" \
        --manifest-sha256 f41567ae577f5053b9ecfc8bda0a1672cc0c59735cb8fc1f8ad4487d46b833c6 \
        --namelist-wps "$proof/wps-aligned-v3-qifix/namelist.wps" \
        --namelist-input "$stage/namelist.input" \
        --geo-em "$proof/wps-aligned-v3-qifix/geo_em.d01.nc" \
        --output "$output" > "$job/job.log" 2>&1
    rc=$?
    printf "%s\n" "$rc" > "$job/exit_code.tmp"
    mv "$job/exit_code.tmp" "$job/exit_code"
    exit "$rc"
' _ "$repo" "$proof" "$stage" "$job" "$output" "$timing" \
    </dev/null >/dev/null 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$job/pid"
echo "STARTED pid=$pid job=$job output=$output"
