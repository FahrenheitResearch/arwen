#!/usr/bin/env bash
set -eu
if [ "$#" -ne 2 ]; then
    printf '%s\n' 'Usage: run-full-scorer.sh OWNED_ENGINE_SOURCE OWNED_CARGO_HOME' >&2
    exit 2
fi
task_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
engine_source=$1
export CARGO_HOME=$2
export TMPDIR="$task_dir/tmp"
export CARGO_TARGET_DIR="$task_dir/full-target"
export CARGO_BUILD_JOBS=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
mkdir -p "$TMPDIR" "$task_dir/receipts"
cd -- "$task_dir"
if [ -f "$task_dir/receipts/full-build.done" ]; then
    printf 'deleted %s (%s bytes)\n' "$task_dir/receipts/full-build.done" "$(wc -c < "$task_dir/receipts/full-build.done")" >> "$task_dir/receipts/deleted-files.log"
    rm "$task_dir/receipts/full-build.done"
fi
status=0
{
    hostname
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    printf 'CARGO_BUILD_JOBS=%s TMPDIR=%s CARGO_TARGET_DIR=%s\n' "$CARGO_BUILD_JOBS" "$TMPDIR" "$CARGO_TARGET_DIR"
    python3 "$task_dir/configure_engine_deps.py" --engine-source "$engine_source" > "$task_dir/receipts/full-dependencies.json" &&
    cargo test --offline --locked --release --manifest-path "$task_dir/Cargo.toml" -- --test-threads=2 --nocapture &&
    cargo build --offline --locked --release --manifest-path "$task_dir/Cargo.toml" &&
    sha256sum "$CARGO_TARGET_DIR/release/solar-albedo-score"
} > "$task_dir/receipts/full-build.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$task_dir/receipts/full-build.done"
exit "$status"
