#!/usr/bin/env bash
set -euo pipefail
source_root=$(realpath "$1")
native=$(realpath "$2")
build=$(realpath -m "$3")
script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$build" == "$source_root" || "$build" == "$source_root/"* ]]; then
  echo 'Landuse control must not modify immutable WRF source' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$build"
cd "$build"
python "$script/extract_landuse.py" "$source_root/phys/module_physics_init.F" control.F90 > extraction.json
flags=(-O0 -cpp -ffp-contract=off -fcheck=all -fbacktrace -ffree-form -ffree-line-length-none -fallow-argument-mismatch)
gfortran -c "${flags[@]}" -I "$native" control.F90
gfortran -c "${flags[@]}" -I "$native" "$script/run_landuse.F90"
gfortran -o run "$native/stub_wrf.o" "$native/module_wrf_error.o" "$native/oracle_io.o" control.o run_landuse.o
cp "$source_root/run/LANDUSE.TBL" LANDUSE.TBL
nice -n 10 ./run "$build/fixtures" > run.log
nice -n 10 ./run "$build/fixtures" --correct-zero > corrected-zero-water.log
if nice -n 10 ./run "$build/negative" --zero-water > negative-zero-water.log 2>&1; then
  echo 'Original zero-water no-data control unexpectedly passed' >&2
  exit 2
fi
gfortran --version | head -1 > compiler.txt
ldd --version | head -1 > libc.txt
printf '%s\n' "${flags[*]}" > compiler-flags.txt
(cd "$source_root" && sha256sum phys/module_physics_init.F run/LANDUSE.TBL) > source-sha256sums.txt
(cd "$script" && sha256sum extract_landuse.py run_landuse.F90 build_landuse.sh) > oracle-sha256sums.txt
find fixtures -type f -print0 | sort -z | xargs -0 sha256sum > fixture-sha256sums.txt
