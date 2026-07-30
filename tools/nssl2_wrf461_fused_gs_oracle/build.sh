#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 /path/to/official-WRF-v4.6.1 /new/build-directory" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
wrf_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
official_source="$wrf_root/phys/module_mp_nssl_2mom.F"
expected_commit=d66e442fccc04111067e29274c9f9eaccc3cef28
expected_source_sha=1eb1b138b75ff3b0cfe33c23779f4ec9b72e57a5455a53ef11c9e55ae0f42722

if [ ! -f "$official_source" ]; then
  echo "missing official WRF source: $official_source" >&2
  exit 2
fi
if ! git -C "$wrf_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "WRF source must be a fresh official git checkout" >&2
  exit 2
fi
actual_commit=$(git -C "$wrf_root" rev-parse HEAD)
if [ "$actual_commit" != "$expected_commit" ]; then
  echo "unexpected WRF commit: $actual_commit" >&2
  exit 2
fi
git -C "$wrf_root" diff --quiet
git -C "$wrf_root" diff --cached --quiet
actual_source_sha=$(sha256sum "$official_source" | awk '{print $1}')
if [ "$actual_source_sha" != "$expected_source_sha" ]; then
  echo "unexpected module_mp_nssl_2mom.F SHA-256: $actual_source_sha" >&2
  exit 2
fi
if [ -e "$build_dir" ]; then
  echo "refusing to reuse oracle build directory: $build_dir" >&2
  exit 2
fi

build_acceptance() {
  local destination=$1
  mkdir -p "$destination"
  cp "$official_source" "$destination/module_mp_nssl_2mom.F.orig"
  cp "$official_source" "$destination/module_mp_nssl_2mom.F"
  (
    cd "$destination"
    sed -i 's/\r$//' module_mp_nssl_2mom.F
    patch --fuzz=0 < "$script_dir/visibility.patch"
    touch namelist.input
    gfortran -c -O2 -cpp -DWRF_CHEM=0 -ffree-form \
      -ffree-line-length-none module_mp_nssl_2mom.F
    gfortran -c -O2 -ffree-form -ffree-line-length-none \
      "$script_dir/../nssl2_wrf461_oracle/stub_wrf.F90" \
      "$script_dir/fused_gs_oracle.F90"
    gfortran -O2 -o nssl2_fused_gs_oracle \
      module_mp_nssl_2mom.o stub_wrf.o fused_gs_oracle.o
    ./nssl2_fused_gs_oracle fused-gs.csv | tee oracle.log
  )
}

build_instrumented() {
  local destination=$1
  mkdir -p "$destination"
  cp "$official_source" "$destination/module_mp_nssl_2mom.F.orig"
  cp "$official_source" "$destination/module_mp_nssl_2mom.F"
  (
    cd "$destination"
    sed -i 's/\r$//' module_mp_nssl_2mom.F
    patch --fuzz=0 < "$script_dir/visibility.patch"
    git apply --check --unidiff-zero "$script_dir/instrumentation.patch"
    git apply --unidiff-zero "$script_dir/instrumentation.patch"
    touch namelist.input
    gfortran -c -O2 -cpp -DWRF_CHEM=0 -ffree-form \
      -ffree-line-length-none module_mp_nssl_2mom.F
    gfortran -c -O2 -ffree-form -ffree-line-length-none \
      "$script_dir/../nssl2_wrf461_oracle/stub_wrf.F90" \
      "$script_dir/fused_gs_oracle.F90"
    gfortran -O2 -o nssl2_fused_gs_oracle_instrumented \
      module_mp_nssl_2mom.o stub_wrf.o fused_gs_oracle.o
    ./nssl2_fused_gs_oracle_instrumented \
      fused-gs-instrumented.csv fused-gs-instrumented.raw.csv \
      | tee oracle-instrumented.log
  )
}

mkdir -p "$build_dir"
build_acceptance "$build_dir/acceptance-a"
build_acceptance "$build_dir/acceptance-b"
cmp "$build_dir/acceptance-a/fused-gs.csv" \
    "$build_dir/acceptance-b/fused-gs.csv"
build_instrumented "$build_dir/instrumented"
cmp "$build_dir/acceptance-a/fused-gs.csv" \
    "$build_dir/instrumented/fused-gs-instrumented.csv"
cp "$build_dir/acceptance-a/fused-gs.csv" "$build_dir/fused-gs.csv"
cp "$build_dir/instrumented/fused-gs-instrumented.raw.csv" \
   "$build_dir/fused-gs-instrumented.raw.csv"
python3 "$script_dir/normalize_diagnostics.py" \
  "$build_dir/fused-gs-instrumented.raw.csv" \
  "$build_dir/fused-gs.csv" \
  "$build_dir/fused-gs-diagnostics.csv"
python3 "$script_dir/validate_oracle.py" \
  "$build_dir/fused-gs.csv" "$build_dir/fused-gs-diagnostics.csv" \
  | tee "$build_dir/VALIDATION.json"

printf '%s  %s\n' "$actual_source_sha" phys/module_mp_nssl_2mom.F \
  > "$build_dir/SOURCE_SHA256"
printf '%s\n' "$actual_commit" > "$build_dir/WRF_COMMIT"
gfortran --version | head -1 > "$build_dir/TOOLCHAIN.txt"
sha256sum "$build_dir/acceptance-a/fused-gs.csv" \
  "$build_dir/acceptance-b/fused-gs.csv" > "$build_dir/REBUILD_SHA256"
sha256sum "$build_dir/fused-gs.csv" > "$build_dir/ORACLE_SHA256"
sha256sum "$build_dir/fused-gs-instrumented.raw.csv" \
  > "$build_dir/INSTRUMENTATION_SHA256"
sha256sum "$build_dir/fused-gs-diagnostics.csv" \
  > "$build_dir/DIAGNOSTICS_SHA256"

echo "NSSL2_FUSED_GS_DETERMINISTIC_REBUILD_COMPLETE $build_dir"
