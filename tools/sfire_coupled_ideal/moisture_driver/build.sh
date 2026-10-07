#!/usr/bin/env bash
set -euo pipefail
source_root=$(realpath "$1")
native=$(realpath "$2")
build=$(realpath -m "$3")
script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$build" == "$source_root" || "$build" == "$source_root/"* ]]; then
  echo 'Moisture oracle must not change the immutable WRF source' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$build"
cd "$build"
python "$script/extract.py" "$source_root/phys/module_fr_fire_driver.F" control.F90 > extraction.json
flags=(-O0 -cpp -ffp-contract=off -fcheck=all -fbacktrace -ffree-form -ffree-line-length-none)
gfortran -c "${flags[@]}" -I "$native" control.F90
python "$script/allocated_classes.py" "$script/run.F90" "$build/run.F90" "${4:-5}" > allocation.json
gfortran -c "${flags[@]}" -I "$native" -I "$build" "$build/run.F90"
gfortran -o run "$native/stub_wrf.o" "$native/module_model_constants.o" "$native/module_wrf_error.o" \
  "$native/module_fr_fire_util.o" "$native/module_fr_fire_phys.o" "$native/oracle_io.o" control.o run.o
cp "$script/../../sfire_wrf471_oracle/namelist.fire" namelist.fire
nice -n 10 ./run "$build/fixtures" > run.log
gfortran --version | head -1 > compiler.txt
ldd --version | head -1 > libc.txt
printf '%s\n' "${flags[*]}" > compiler-flags.txt
(cd "$source_root" && sha256sum phys/module_fr_fire_driver.F phys/module_fr_fire_phys.F phys/module_fr_fire_util.F) > source-sha256sums.txt
(cd "$script" && sha256sum extract.py run.F90 build.sh allocated_classes.py) > oracle-sha256sums.txt
sha256sum run.F90 >> oracle-sha256sums.txt
(cd "$native" && sha256sum stub_wrf.o module_model_constants.o module_wrf_error.o module_fr_fire_util.o module_fr_fire_phys.o oracle_io.o) > support-object-sha256sums.txt
find fixtures -type f -print0 | sort -z | xargs -0 sha256sum > fixture-sha256sums.txt
