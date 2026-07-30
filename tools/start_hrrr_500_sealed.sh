#!/usr/bin/env bash
set -Eeuo pipefail

version=${1:?usage: start_hrrr_500_sealed.sh VERSION RUN_SECONDS IO_MODE BRIDGE MANIFEST_SHA256}
run_seconds=${2:?missing RUN_SECONDS}
io_mode=${3:?missing IO_MODE}
bridge=${4:?missing BRIDGE}
manifest_sha=${5:?missing MANIFEST_SHA256}
repo=/workspace/gpuwm-hrrr-native-state-v1
proof=/workspace/gpuwm-hrrr-proof-20260718T00-v1
stage=/workspace/gpuwm-hrrr-proof-staging-v1
job="$stage/jobs/native_500_sealed_$version"
outdir="$proof/native/single-500x500-$version"
timing="$proof/native/single-500x500-$version.time.txt"
static_cache="$proof/native/static-500x500-v1.npz"
static_receipt="$proof/evidence/static-500x500-v1.json"

for path in "$job" "$outdir"; do
    [[ ! -e "$path" ]] || { echo "refusing existing path $path" >&2; exit 2; }
done
[[ -d "$bridge" && -f "$static_cache" && -f "$static_receipt" ]] || {
    echo "sealed bridge or native static cache is missing" >&2; exit 2; }
mkdir -p "$job"

nohup bash -c '
    set +e
    repo=$1; stage=$2; job=$3; outdir=$4; timing=$5; bridge=$6
    manifest_sha=$7; static_cache=$8; static_receipt=$9
    run_seconds=${10}; io_mode=${11}
    cd "$repo"
    env PYTHONPATH=. /usr/bin/time -v -o "$timing" \
        /workspace/gpuwm-physics-install/venv/bin/python \
        tools/hrrr_single_domain_benchmark.py \
        --bridge "$bridge" --manifest-sha256 "$manifest_sha" \
        --static-cache "$static_cache" \
        --static-receipt "$static_receipt" \
        --namelist-input "$stage/namelist.input" \
        --run-seconds "$run_seconds" --io-mode "$io_mode" \
        --outdir "$outdir" > "$job/job.log" 2>&1
    rc=$?
    printf "%s\n" "$rc" > "$job/exit_code.tmp"
    mv "$job/exit_code.tmp" "$job/exit_code"
    exit "$rc"
' _ "$repo" "$stage" "$job" "$outdir" "$timing" "$bridge" \
    "$manifest_sha" "$static_cache" "$static_receipt" \
    "$run_seconds" "$io_mode" </dev/null >/dev/null 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$job/pid"
echo "STARTED pid=$pid job=$job outdir=$outdir"
