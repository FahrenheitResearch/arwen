#!/usr/bin/env bash
set -Eeuo pipefail

version=${1:?usage: start_hrrr_parallel_decode_benchmark.sh VERSION WORKERS}
workers=${2:?missing WORKERS}
repo=/workspace/gpuwm-hrrr-native-state-v1
proof=/workspace/gpuwm-hrrr-proof-20260718T00-v1
stage=/workspace/gpuwm-hrrr-proof-staging-v1
job="$stage/jobs/native_decode_$version"
outdir="$proof/native/source-window-f00-f12-$version"
timing="$proof/native/source-window-f00-f12-$version.time.txt"
binary="$repo/tools/hrrr_grib2_bridge-parallel-v1"
series="$stage/hrrr-f00-f12-series-v1.tsv"

[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo "workers must be positive" >&2; exit 2; }
[[ ! -e "$job" ]] || { echo "refusing existing job $job" >&2; exit 2; }
[[ ! -e "$outdir" ]] || { echo "refusing existing output $outdir" >&2; exit 2; }
[[ -x "$binary" ]] || { echo "missing executable $binary" >&2; exit 2; }
mkdir -p "$job"

nohup bash -c '
    set +e
    job=$1
    timing=$2
    binary=$3
    series=$4
    outdir=$5
    workers=$6
    nice -n 10 /usr/bin/time -v -o "$timing" \
        "$binary" --series-workers "$workers" "$series" "$outdir" \
        "2026-07-18 00:00:00" 781 987 315 521 \
        > "$job/job.log" 2>&1
    rc=$?
    printf "%s\n" "$rc" > "$job/exit_code.tmp"
    mv "$job/exit_code.tmp" "$job/exit_code"
    exit "$rc"
' _ "$job" "$timing" "$binary" "$series" "$outdir" "$workers" \
    </dev/null >/dev/null 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$job/pid"
echo "STARTED pid=$pid job=$job outdir=$outdir"
