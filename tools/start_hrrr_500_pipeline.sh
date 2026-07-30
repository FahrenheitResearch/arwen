#!/usr/bin/env bash
set -Eeuo pipefail

version=${1:?usage: start_hrrr_500_pipeline.sh VERSION RUN_SECONDS [IO_MODE] [WORKERS]}
run_seconds=${2:?missing RUN_SECONDS}
io_mode=${3:-none}
workers=${4:-8}
repo=/workspace/gpuwm-hrrr-native-state-v1
proof=/workspace/gpuwm-hrrr-proof-20260718T00-v1
stage=/workspace/gpuwm-hrrr-proof-staging-v1
job="$stage/jobs/native_500_pipeline_$version"
bridge="$proof/native/source-window-f00-f12-500pipeline-$version"
signals="$proof/evidence/pipeline-500-$version"
outdir="$proof/native/single-500x500-$version"
timing="$proof/native/single-500x500-$version.time.txt"
static_cache="$proof/native/static-500x500-v1.npz"
static_receipt="$proof/evidence/static-500x500-v1.json"

for path in "$job" "$bridge" "$signals" "$outdir"; do
    [[ ! -e "$path" ]] || { echo "refusing existing path $path" >&2; exit 2; }
done
[[ -f "$static_cache" && -f "$static_receipt" ]] || {
    echo "native static cache/receipt is missing" >&2; exit 2; }
mkdir -p "$job"

nohup bash -c '
    set +e
    repo=$1; proof=$2; stage=$3; job=$4; bridge=$5; signals=$6
    outdir=$7; timing=$8; static_cache=$9; static_receipt=${10}
    run_seconds=${11}; io_mode=${12}; workers=${13}
    cd "$repo"
    env PYTHONPATH=. /usr/bin/time -v -o "$timing" \
        /workspace/gpuwm-physics-install/venv/bin/python \
        tools/hrrr_single_domain_benchmark.py \
        --bridge "$bridge" \
        --pipeline-series "$stage/hrrr-f00-f12-series-v1.tsv" \
        --pipeline-decoder "$repo/tools/hrrr_grib2_bridge-pipeline-v2" \
        --pipeline-signals "$signals" \
        --pipeline-workers "$workers" \
        --source-root "$proof/data" \
        --source-manifest "$proof/evidence/download-f00-f12-v2/SHA256SUMS" \
        --source-manifest-sha256 c0680b14a8c8dacef5d9013947f8cd63a3246bc0832102185261f625fc4cc85c \
        --static-cache "$static_cache" \
        --static-receipt "$static_receipt" \
        --namelist-input "$stage/namelist.input" \
        --run-seconds "$run_seconds" --io-mode "$io_mode" \
        --outdir "$outdir" > "$job/job.log" 2>&1
    rc=$?
    printf "%s\n" "$rc" > "$job/exit_code.tmp"
    mv "$job/exit_code.tmp" "$job/exit_code"
    exit "$rc"
' _ "$repo" "$proof" "$stage" "$job" "$bridge" "$signals" \
    "$outdir" "$timing" "$static_cache" "$static_receipt" \
    "$run_seconds" "$io_mode" "$workers" </dev/null >/dev/null 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$job/pid"
echo "STARTED pid=$pid job=$job outdir=$outdir"
