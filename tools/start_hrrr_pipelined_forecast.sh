#!/usr/bin/env bash
set -Eeuo pipefail

version=${1:?usage: start_hrrr_pipelined_forecast.sh VERSION RUN_SECONDS [WORKERS] [SAMPLE_INTERVAL_S]}
run_seconds=${2:?missing RUN_SECONDS}
workers=${3:-8}
sample_interval=${4:-300}
repo=/workspace/gpuwm-hrrr-native-state-v1
proof=/workspace/gpuwm-hrrr-proof-20260718T00-v1
stage=/workspace/gpuwm-hrrr-proof-staging-v1
job="$stage/jobs/native_pipeline_$version"
bridge="$proof/native/source-window-f00-f12-pipeline-$version"
signals="$proof/evidence/pipeline-$version"
outdir="$proof/native/two-domain-pipeline-$version"
timing="$proof/native/two-domain-pipeline-$version.time.txt"

for path in "$job" "$bridge" "$signals" "$outdir"; do
    [[ ! -e "$path" ]] || { echo "refusing existing path $path" >&2; exit 2; }
done
mkdir -p "$job"

nohup bash -c '
    set +e
    repo=$1
    proof=$2
    stage=$3
    job=$4
    bridge=$5
    signals=$6
    outdir=$7
    timing=$8
    run_seconds=$9
    workers=${10}
    sample_interval=${11}
    cd "$repo"
    env PYTHONPATH=. /usr/bin/time -v -o "$timing" \
        /workspace/gpuwm-physics-install/venv/bin/python \
        tools/hrrr_two_domain_forecast.py \
        --bridge "$bridge" \
        --pipeline-series "$stage/hrrr-f00-f12-series-v1.tsv" \
        --pipeline-decoder "$repo/tools/hrrr_grib2_bridge-pipeline-v2" \
        --pipeline-signals "$signals" \
        --pipeline-workers "$workers" \
        --source-root "$proof/data" \
        --source-manifest "$proof/evidence/download-f00-f12-v2/SHA256SUMS" \
        --source-manifest-sha256 c0680b14a8c8dacef5d9013947f8cd63a3246bc0832102185261f625fc4cc85c \
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
' _ "$repo" "$proof" "$stage" "$job" "$bridge" "$signals" \
    "$outdir" "$timing" "$run_seconds" "$workers" "$sample_interval" \
    </dev/null >/dev/null 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$job/pid"
echo "STARTED pid=$pid job=$job outdir=$outdir"
