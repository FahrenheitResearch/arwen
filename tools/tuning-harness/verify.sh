#!/usr/bin/env bash
set -euo pipefail

# Invoke from this crate on a CPU node. The caller starts this detached.
scratch_dir="${1:?usage: verify.sh SCRATCH_DIRECTORY}"
mkdir -p "$scratch_dir/tmp" "$scratch_dir/cargo-target"
export TMPDIR="$scratch_dir/tmp"
export CARGO_TARGET_DIR="$scratch_dir/cargo-target"
export CUDA_VISIBLE_DEVICES=
export GPUWM_NO_LOCAL_GPU=1
cpu_jobs=1
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    read -r quota period < /sys/fs/cgroup/cpu.max
    if [[ "$quota" != max ]]; then cpu_jobs=$(( quota / period )); fi
elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
    read -r quota < /sys/fs/cgroup/cpu/cpu.cfs_quota_us
    read -r period < /sys/fs/cgroup/cpu/cpu.cfs_period_us
    if (( quota > 0 )); then cpu_jobs=$(( quota / period )); fi
fi
if (( cpu_jobs < 1 )); then cpu_jobs=1; fi
export OMP_NUM_THREADS="$cpu_jobs"
export OPENBLAS_NUM_THREADS="$cpu_jobs"
export MKL_NUM_THREADS="$cpu_jobs"
export RAYON_NUM_THREADS="$cpu_jobs"
export CARGO_BUILD_JOBS="$cpu_jobs"
status=1
trap 'printf "%s\n" "$status" > "$scratch_dir/verify.done"' EXIT
{
    printf 'CPU build jobs: %s\n' "$cpu_jobs"
    if [[ ! -f Cargo.lock ]]; then nice -n 10 cargo generate-lockfile; fi
    nice -n 10 cargo fmt --check
    nice -n 10 cargo test --locked
    nice -n 10 cargo build --release --locked
} > "$scratch_dir/verify.log" 2>&1
status=0
