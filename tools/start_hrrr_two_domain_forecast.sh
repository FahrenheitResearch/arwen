#!/usr/bin/env bash
set -Eeuo pipefail

version=${1:?usage: start_hrrr_two_domain_forecast.sh VERSION RUN_SECONDS BRIDGE MANIFEST_SHA256 [SAMPLE_INTERVAL_S]}
run_seconds=${2:?missing RUN_SECONDS}
bridge=${3:?missing BRIDGE}
manifest_sha=${4:?missing MANIFEST_SHA256}
sample_interval=${5:-60}
repo=/workspace/gpuwm-hrrr-native-state-v1
proof=/workspace/gpuwm-hrrr-proof-20260718T00-v1
stage=/workspace/gpuwm-hrrr-proof-staging-v1
job="$stage/jobs/native_two_domain_$version"
outdir="$proof/native/two-domain-$version"
timing="$proof/native/two-domain-$version.time.txt"

[[ ! -e "$job" ]] || { echo "refusing existing job $job" >&2; exit 2; }
[[ ! -e "$outdir" ]] || { echo "refusing existing output $outdir" >&2; exit 2; }
mkdir -p "$job"

nohup bash -c '
    set +e
    repo=$1
    proof=$2
    stage=$3
    job=$4
    outdir=$5
    timing=$6
    bridge=$7
    manifest_sha=$8
    run_seconds=$9
    sample_interval=${10}
    cd "$repo"
    env PYTHONPATH=. /usr/bin/time -v -o "$timing" \
        /workspace/gpuwm-physics-install/venv/bin/python \
        tools/hrrr_two_domain_forecast.py \
        --bridge "$bridge" \
        --manifest-sha256 "$manifest_sha" \
        --namelist-input "$stage/namelist.input" \
        --geo-d01 "$proof/wps-aligned-v3-qifix/geo_em.d01.nc" \
        --geo-d02 "$proof/wps-aligned-v3-qifix/geo_em.d02.nc" \
        --run-seconds "$run_seconds" \
        --sample-interval-s "$sample_interval" \
        --outdir "$outdir" > "$job/job.log" 2>&1
    rc=$?
    printf "%s\n" "$rc" > "$job/exit_code.tmp"
    mv "$job/exit_code.tmp" "$job/exit_code"
    exit "$rc"
' _ "$repo" "$proof" "$stage" "$job" "$outdir" "$timing" \
    "$bridge" "$manifest_sha" "$run_seconds" "$sample_interval" \
    </dev/null >/dev/null 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$job/pid"
echo "STARTED pid=$pid job=$job outdir=$outdir"
