#!/usr/bin/env bash
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
gfortran "${flags[@]}" -c "$script_dir/run_surface_layer.F90"
gfortran "${flags[@]}" -o run_surface_layer \
  stub_wrf.o module_sf_mynn.o run_surface_layer.o
./run_surface_layer surface-layer.csv
python3 "$script_dir/validate_oracle.py" surface-layer.csv

gfortran "${flags[@]}" -c "$script_dir/run_surface_layer_wide.F90"
gfortran "${flags[@]}" -o run_surface_layer_wide \
  stub_wrf.o module_sf_mynn.o run_surface_layer_wide.o
./run_surface_layer_wide surface-layer-wide.csv surface-layer-wrapper.csv
PYTHONPATH="$repo_root" python3 "$script_dir/validate_wide_oracle.py" \
  surface-layer-wide.csv surface-layer-wrapper.csv

sha256sum \
  "$source_file" \
  "$script_dir/stub_wrf.F90" \
  "$script_dir/run_surface_layer.F90" \
  "$script_dir/run_surface_layer_wide.F90" \
  surface-layer.csv \
  surface-layer-wide.csv \
  surface-layer-wrapper.csv | tee SHA256SUMS
