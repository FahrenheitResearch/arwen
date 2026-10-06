#!/usr/bin/env bash
# Build the GSL WRF 3.9 fork MYNN surface-layer oracle and write its CSV.
#   ./build.sh /path/to/fork/module_sf_mynn.F /new/build-directory
# The source is NOAA-EMC/HRRR tag v4.1.21,
# sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_sf_mynn.F, pinned by SHA-256.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 /path/to/module_sf_mynn.F /new/build-directory" >&2; exit 2
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_file=$(realpath "$1")
build_dir=$(realpath -m "$2")
expected=47bc9943f505c853e7dbb09b24d61889b4fc85ccc72a53b60903f6f46458a214
if [ "$(sha256sum "$source_file" | cut -d ' ' -f 1)" != "$expected" ]; then
  echo "module_sf_mynn.F bytes differ from the pinned fork source" >&2; exit 2
fi
if [ -e "$build_dir" ]; then
  echo "refusing to reuse oracle build directory: $build_dir" >&2; exit 2
fi
mkdir -p "$build_dir"
cd "$build_dir"
gfortran --version | head -n 1 | tee GFORTRAN_VERSION
flags=(-O2 -ffree-form -ffree-line-length-none -fcheck=all)
gfortran "${flags[@]}" -c "$script_dir/stub_wrf.F90"
cp "$source_file" module_sf_mynn.F
gfortran "${flags[@]}" -c module_sf_mynn.F
gfortran "${flags[@]}" -c "$script_dir/run_surface_layer_fork.F90"
gfortran "${flags[@]}" -o run_surface_layer_fork \
  stub_wrf.o module_sf_mynn.o run_surface_layer_fork.o
python3 "$script_dir/make_columns.py" columns.txt 6
./run_surface_layer_fork columns.txt surface-layer-gsl-wrf39.csv
gzip -9 -n -k surface-layer-gsl-wrf39.csv
gfortran "${flags[@]}" -c "$script_dir/run_zolri_fork.F90"
gfortran "${flags[@]}" -o run_zolri_fork \
  stub_wrf.o module_sf_mynn.o run_zolri_fork.o
./run_zolri_fork "$script_dir/zolri-columns.txt" zolri-gsl-wrf39.csv
sha256sum module_sf_mynn.F "$script_dir/stub_wrf.F90" \
  "$script_dir/run_surface_layer_fork.F90" "$script_dir/make_columns.py" \
  "$script_dir/run_zolri_fork.F90" "$script_dir/zolri-columns.txt" \
  columns.txt surface-layer-gsl-wrf39.csv zolri-gsl-wrf39.csv \
  | sed -E 's@  .*/@  @' | tee SHA256SUMS
