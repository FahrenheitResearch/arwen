#!/usr/bin/env bash
# Compile the byte-extracted driver routine with native WRF support objects.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo 'usage: build_wind.sh WRF_SOURCE_ROOT NATIVE_ORACLE_BUILD WIND_BUILD' >&2
    exit 2
fi
source_root=$(realpath "$1")
native=$(realpath "$2")
build=$(realpath -m "$3")
script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$build" == "$source_root" || "$build" == "$source_root/"* ]]; then
    echo 'Wind build inside WRF source would change the immutable reference' >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$build"
cd "$build"
python "$script/extract_wind.py" "$source_root/phys/module_fr_fire_driver.F" module_sfire_wind_oracle.F90 > extraction.json
flags=(-O0 -cpp -ffp-contract=off -fcheck=all -fbacktrace -ffree-form -ffree-line-length-none -fallow-argument-mismatch)
gfortran -c "${flags[@]}" -I "$native" module_sfire_wind_oracle.F90
gfortran -c "${flags[@]}" -I "$native" -I "$build" "$script/run_wind.F90"
objects=(stub_wrf module_model_constants module_wrf_error module_fr_fire_util)
paths=()
for object in "${objects[@]}"; do paths+=("$native/$object.o"); done
gfortran -o run_wind "${paths[@]}" "$native/oracle_io.o" module_sfire_wind_oracle.o run_wind.o
nm -u module_sfire_wind_oracle.o > undefined-wind.txt
if grep -q '_ZGV' undefined-wind.txt; then
    echo 'Wind object uses vector libm; scalar binary32 oracle is invalid' >&2
    exit 4
fi
nice -n 10 ./run_wind "$build/fixtures" > run-wind-stdout.txt
gfortran --version | head -1 > compiler.txt
ldd --version | head -1 > libc.txt
printf '%s\n' "${flags[*]}" > compiler-flags.txt
(cd "$source_root" && sha256sum phys/module_fr_fire_driver.F) > source-sha256sums.txt
(cd "$source_root" && sha256sum share/module_model_constants.F phys/module_fr_fire_util.F) > support-source-sha256sums.txt
(cd "$native" && sha256sum stub_wrf.o module_model_constants.o module_wrf_error.o module_fr_fire_util.o oracle_io.o) > support-object-sha256sums.txt
(cd "$script" && sha256sum run_wind.F90 extract_wind.py build_wind.sh) > oracle-sha256sums.txt
find fixtures -type f -print0 | sort -z | xargs -0 sha256sum > fixture-sha256sums.txt
echo "Native atmosphere-to-fire wind fixtures: $build/fixtures"
