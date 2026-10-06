#!/usr/bin/env bash
set -eu
task_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export TMPDIR="$task_dir/tmp"
export CARGO_TARGET_DIR="$task_dir/target"
export CARGO_BUILD_JOBS=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
mkdir -p "$TMPDIR" "$task_dir/receipts"
if [ -f "$task_dir/receipts/tests.done" ]; then
    printf 'deleted %s (%s bytes)\n' "$task_dir/receipts/tests.done" "$(wc -c < "$task_dir/receipts/tests.done")" >> "$task_dir/receipts/deleted-files.log"
    rm "$task_dir/receipts/tests.done"
fi
status=0
{
    hostname
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    for quota_file in /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us; do
        if [ -f "$quota_file" ]; then
            printf '%s: ' "$quota_file"
            cat "$quota_file"
        fi
    done
    printf 'CARGO_BUILD_JOBS=%s TMPDIR=%s CARGO_TARGET_DIR=%s\n' "$CARGO_BUILD_JOBS" "$TMPDIR" "$CARGO_TARGET_DIR"
    "$HOME/.cargo/bin/cargo" test --offline --locked --manifest-path "$task_dir/checkpoint-tests/Cargo.toml" -- --test-threads=2 --nocapture
} > "$task_dir/receipts/tests.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$task_dir/receipts/tests.done"
exit "$status"
