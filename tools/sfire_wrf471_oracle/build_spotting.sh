#!/usr/bin/env bash
set -euo pipefail
source_root=$(realpath "$1")
native=$(realpath "$2")
build=$(realpath -m "$3")
script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$build" == "$source_root" || "$build" == "$source_root/"* ]]; then
  echo 'Firebrand controls must not change immutable WRF sources' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$build"
cd "$build"
python "$script/extract_spotting.py" "$source_root/phys/module_firebrand_spotting.F" module_spotting_oracle.F90 > extraction.json
flags=(-O0 -cpp -ffp-contract=off -fcheck=all -fbacktrace -ffree-form -ffree-line-length-none)
gfortran -c "${flags[@]}" module_spotting_oracle.F90
gfortran -c "${flags[@]}" -I "$native" -I "$build" "$script/run_spotting.F90"
gfortran -o run_spotting "$native/stub_wrf.o" "$native/oracle_io.o" module_spotting_oracle.o run_spotting.o
nm -u module_spotting_oracle.o > undefined.txt
if grep -q '_ZGV' undefined.txt; then
  echo 'Firebrand controls unexpectedly use vector libm' >&2
  exit 3
fi
nice -n 10 ./run_spotting "$build/fixtures" > run.log
gfortran --version | head -1 > compiler.txt
ldd --version | head -1 > libc.txt
printf '%s\n' "${flags[*]}" > compiler-flags.txt
(cd "$source_root" && sha256sum phys/module_firebrand_spotting.F phys/module_firebrand_spotting_mpi.F) > source-sha256sums.txt
(cd "$script" && sha256sum extract_spotting.py run_spotting.F90 build_spotting.sh) > oracle-sha256sums.txt
(cd "$native" && sha256sum stub_wrf.o oracle_io.o) > support-object-sha256sums.txt
find fixtures -type f -print0 | sort -z | xargs -0 sha256sum > fixture-sha256sums.txt
