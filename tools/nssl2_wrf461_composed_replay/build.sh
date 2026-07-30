#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 /path/to/WRF-v4.6.1 /new/build-directory" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
wrf_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
official_source="$wrf_root/phys/module_mp_nssl_2mom.F"
expected_source_sha=5aaae368289694c929d38365d77d445e4f22291a30a48555df7a21d470b72ae3

if [ ! -f "$official_source" ]; then
  echo "missing official WRF source: $official_source" >&2
  exit 2
fi
actual_source_sha=$(sha256sum "$official_source" | awk '{print $1}')
if [ "$actual_source_sha" != "$expected_source_sha" ]; then
  echo "unexpected module_mp_nssl_2mom.F SHA-256: $actual_source_sha" >&2
  exit 2
fi
if [ -e "$build_dir" ]; then
  echo "refusing to reuse composed replay build directory: $build_dir" >&2
  exit 2
fi

mkdir -p "$build_dir"
cp "$official_source" "$build_dir/module_mp_nssl_2mom.F"
cd "$build_dir"
sed -i 's/\r$//' module_mp_nssl_2mom.F
touch namelist.input

gfortran -c -O2 -cpp -DWRF_CHEM=0 -ffree-form \
  -ffree-line-length-none module_mp_nssl_2mom.F
gfortran -c -O2 -ffree-form -ffree-line-length-none \
  "$script_dir/stub_wrf.F90" "$script_dir/composed_column.F90"
gfortran -O2 -o nssl2_composed_column \
  module_mp_nssl_2mom.o stub_wrf.o composed_column.o

printf '%s  %s\n' "$actual_source_sha" module_mp_nssl_2mom.F > SOURCE_SHA256
gfortran --version | head -n 1 > TOOLCHAIN.txt
printf '%s\n' "$script_dir" > HARNESS_SOURCE.txt
