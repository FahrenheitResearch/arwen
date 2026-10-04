#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
source_file=$(realpath "$1")
build=$(realpath -m "$2")
expected=3265f810d08dcbddfaf198371dc7f652e78e8d3a788f703a515c555a3bbb2a12
actual=$(sha256sum "$source_file" | cut -d' ' -f1)
if [[ "$actual" != "$expected" ]]; then
  echo 'RUC oracle requires the pinned WRF v4.6.1 module; another source changes the reference physics.' >&2
  exit 2
fi
mkdir -p "$build"
cd "$build"
export TMPDIR="$build/tmp"
mkdir -p "$TMPDIR"
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none "$root/tools/ruc_wrf461_oracle/stub_wrf.F90"
gfortran -c -O0 -cpp -DEM_CORE=0 -Dwrf_chem=0 -fallow-argument-mismatch -ffree-form -ffree-line-length-none "$source_file"
gfortran -O0 -ffree-form -ffree-line-length-none -o run_surface stub_wrf.o module_sf_ruclsm.o "$root/tools/ruc_mosaic_wrf461_oracle/run_surface.F90"
python3 "$root/tools/ruc_mosaic_wrf461_oracle/generate.py" run_driver.F90
gfortran -O0 -ffree-form -ffree-line-length-none -o run_driver stub_wrf.o module_sf_ruclsm.o run_driver.F90
cp "$root/gpuwm/data/noah_tables/VEGPARM.TBL" "$root/gpuwm/data/noah_tables/SOILPARM.TBL" "$root/gpuwm/data/noah_tables/GENPARM.TBL" .
./run_surface surface.csv
./run_driver driver.csv
sha256sum "$source_file" run_driver.F90 surface.csv driver.csv > sha256sums.txt
