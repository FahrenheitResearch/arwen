#!/usr/bin/env bash
# Build the complete official em_fire case without changing its science inputs.
set -euo pipefail
source_root=$(realpath "${1:?WRF source directory}")
build_root=$(realpath -m "${2:?separate build directory}")
install_root=$(realpath -m "${3:?separate install directory}")
jobs=${4:-2}
case "$jobs" in ''|*[!0-9]*|0) echo 'Build jobs must be a positive integer.' >&2; exit 2;; esac
case "$build_root/" in "$source_root/"*) echo 'The build directory must be outside the WRF source.' >&2; exit 2;; esac
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
export OMP_NUM_THREADS="$jobs" OPENBLAS_NUM_THREADS="$jobs" MKL_NUM_THREADS="$jobs"
mkdir -p "$build_root" "$install_root" "$install_root/io_int" "$install_root/io_netcdf"
# A container's host core count is not its usable quota. Check every
# delegated cgroup ancestor and keep the requested cap when it is lower.
limit_jobs() {
  local quota period cap
  if [[ -f "$1/cpu.max" ]]; then
    read -r quota period < "$1/cpu.max"
    printf '%s %s %s\n' "$1/cpu.max" "$quota" "$period" >> "$build_root/cpu-limits.txt"
  elif [[ -f "$1/cpu.cfs_quota_us" && -f "$1/cpu.cfs_period_us" ]]; then
    quota=$(cat "$1/cpu.cfs_quota_us"); period=$(cat "$1/cpu.cfs_period_us")
    printf '%s %s %s\n' "$1" "$quota" "$period" >> "$build_root/cpu-limits.txt"
  else return; fi
  if [[ "$quota" != max && "$quota" -gt 0 ]]; then
    cap=$((quota / period)); if [[ "$cap" -lt 1 ]]; then cap=1; fi
    if [[ "$jobs" -gt "$cap" ]]; then jobs=$cap; fi
  fi
}
quota_path=/sys/fs/cgroup
while IFS=: read -r hierarchy controllers group_path; do
  if [[ "$hierarchy" == 0 ]]; then quota_path="/sys/fs/cgroup$group_path"; break; fi
done < /proc/self/cgroup
while [[ "$quota_path" == /sys/fs/cgroup* ]]; do
  limit_jobs "$quota_path"
  if [[ "$quota_path" == /sys/fs/cgroup ]]; then break; fi
  quota_path=$(dirname "$quota_path")
done
limit_jobs /sys/fs/cgroup/cpu
limit_jobs /sys/fs/cgroup/cpu,cpuacct
export OMP_NUM_THREADS="$jobs" OPENBLAS_NUM_THREADS="$jobs" MKL_NUM_THREADS="$jobs"
printf 'build_jobs=%s\n' "$jobs" >> "$build_root/cpu-limits.txt"
for module in diffwrf_int diffwrf WRF_Core WRF wrf; do mkdir -p "$install_root/modules/$module"; done
date -u +%FT%TZ > "$build_root/started.txt"
gfortran --version > "$build_root/compiler.txt"
cmake --version > "$build_root/cmake.txt"
nice -n 15 cmake -S "$source_root" -B "$build_root" \
  -DCMAKE_C_PREPROCESSOR=/usr/bin/cpp -DWRF_CORE=ARW -DWRF_CASE=EM_FIRE \
  -DWRF_NESTING=NONE -DUSE_MPI=OFF -DUSE_OPENMP=OFF \
  -DCMAKE_BUILD_TYPE=Release -DWRF_FCOPTIM='-O2 -ffp-contract=off -fno-tree-vectorize -fno-tree-slp-vectorize' \
  -DCMAKE_INSTALL_PREFIX="$install_root" -DCMAKE_Fortran_COMPILER=gfortran \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_Fortran_FLAGS=-fallow-argument-mismatch
nice -n 15 cmake --build "$build_root" -j "$jobs"
sha256sum "$build_root/main/ideal" "$build_root/main/wrf" > "$build_root/EXECUTABLES.sha256"
find "$source_root" -type f -name '*.F' -print0 | sort -z | xargs -0 sha256sum > "$build_root/FORTRAN_SOURCES.sha256"
find "$build_root" -name flags.make -print0 | sort -z | xargs -0 sha256sum > "$build_root/FLAGS.sha256"
date -u +%FT%TZ > "$build_root/finished.txt"
