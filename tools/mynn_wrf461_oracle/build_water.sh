#!/usr/bin/env bash
# Build and run the over-water ISFTCFLX MYNN surface-layer oracle.
#
# Same contract as build.sh and build_coarse.sh: the WRF checkout must be the
# pinned v4.6.1 commit and phys/module_sf_mynn.F must hash to the pinned bytes,
# so the two CSVs this writes are produced by the UNMODIFIED module.  It builds
# only the water harness and never touches surface-layer.csv,
# surface-layer-wide.csv, surface-layer-wrapper.csv or surface-layer-coarse.csv.
#
# `nm -u module_sf_mynn.o` is recorded: at -O2 GCC 13 enables -ftree-vectorize,
# and a vectorized EXP/LOG call binds libmvec (_ZGVbN4v_expf, 4-ULP contract)
# instead of scalar glibc.  The receipt is what makes the max_ulp 0 claim
# against gpuwm/core/noahmp_libm.py checkable rather than assumed.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 /path/to/WRF-v4.6.1 /new/build-directory" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
wrf_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
source_file="$wrf_root/phys/module_sf_mynn.F"
expected_source_sha256=86395534a6c9bfc79dcad50094bce290eff05756777a95794b2673795f9761c3
expected_wrf_commit=d66e442fccc04111067e29274c9f9eaccc3cef28

if [ ! -f "$source_file" ]; then
  echo "missing official WRF source: $source_file" >&2
  exit 2
fi
if [ "$(git -C "$wrf_root" rev-parse HEAD)" != "$expected_wrf_commit" ]; then
  echo "WRF checkout is not the pinned v4.6.1 commit" >&2
  exit 2
fi
if [ "$(sha256sum "$source_file" | cut -d ' ' -f 1)" != \
     "$expected_source_sha256" ]; then
  echo "module_sf_mynn.F bytes differ from the pinned source" >&2
  exit 2
fi
if [ -e "$build_dir" ]; then
  echo "refusing to reuse oracle build directory: $build_dir" >&2
  exit 2
fi
mkdir -p "$build_dir"

cd "$build_dir"
gfortran --version | head -n 1 | tee GFORTRAN_VERSION
flags=(-O2 -ffree-form -ffree-line-length-none -fcheck=all \
       -ffpe-trap=invalid,zero,overflow)
gfortran "${flags[@]}" -c "$script_dir/stub_wrf.F90"
gfortran "${flags[@]}" -c "$source_file"
nm -u module_sf_mynn.o | tee MODULE_UNDEFINED_SYMBOLS
if grep -q '_ZGV' MODULE_UNDEFINED_SYMBOLS; then
  echo "module_sf_mynn.o binds libmvec; the scalar-libm oracle claim is void" >&2
  exit 2
fi
gfortran "${flags[@]}" -c "$script_dir/run_surface_layer_water.F90"
gfortran "${flags[@]}" -o run_surface_layer_water \
  stub_wrf.o module_sf_mynn.o run_surface_layer_water.o
./run_surface_layer_water surface-layer-water.csv surface-layer-water-leaf.csv
PYTHONPATH="$repo_root" python3 \
  "$script_dir/validate_water_oracle.py" \
  surface-layer-water.csv surface-layer-water-leaf.csv

sha256sum \
  "$source_file" \
  "$script_dir/stub_wrf.F90" \
  "$script_dir/run_surface_layer_water.F90" \
  surface-layer-water.csv \
  surface-layer-water-leaf.csv | tee SHA256SUMS_WATER
