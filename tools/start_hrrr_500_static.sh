#!/usr/bin/env bash
set -Eeuo pipefail

repo=/workspace/gpuwm-hrrr-native-state-v1
proof=/workspace/gpuwm-hrrr-proof-20260718T00-v1
stage=/workspace/gpuwm-hrrr-proof-staging-v1
job="$stage/jobs/native_500_static_v1"
output="$proof/native/static-500x500-v1.npz"
receipt="$proof/evidence/static-500x500-v1.json"
timing="$proof/evidence/static-500x500-v1.time.txt"
geog=/workspace/gpuwm-real74-data/WRF_1974_MP55_reference_bundle/static/WPS_GEOG

for path in "$job" "$output" "$receipt"; do
    [[ ! -e "$path" ]] || { echo "refusing existing path $path" >&2; exit 2; }
done
mkdir -p "$job"
nohup bash -c '
    set +e
    repo=$1; job=$2; output=$3; receipt=$4; timing=$5; geog=$6
    cd "$repo"
    env PYTHONPATH=. /usr/bin/time -v -o "$timing" \
        /workspace/gpuwm-physics-install/venv/bin/python \
        tools/hrrr_build_native_static.py \
        --geog-root "$geog" --output "$output" --receipt "$receipt" \
        > "$job/job.log" 2>&1
    rc=$?
    printf "%s\n" "$rc" > "$job/exit_code.tmp"
    mv "$job/exit_code.tmp" "$job/exit_code"
    exit "$rc"
' _ "$repo" "$job" "$output" "$receipt" "$timing" "$geog" \
    </dev/null >/dev/null 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$job/pid"
echo "STARTED pid=$pid job=$job"
