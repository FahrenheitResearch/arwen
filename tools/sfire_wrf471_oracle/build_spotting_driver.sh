#!/usr/bin/env bash
set -euo pipefail
source_root=$(realpath "$1")
native=$(realpath "$2")
build=$(realpath -m "$3")
script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$build" == "$source_root" || "$build" == "$source_root/"* ]]; then
  echo 'Spotting driver builds must not change immutable WRF source' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$(realpath "$script/../..")"
mkdir -p "$build"
cd "$build"
python "$script/correct_spotting.py" "$source_root/phys/module_firebrand_spotting.F" corrected.F > corrections.json
flags=(-O0 -cpp -ffp-contract=off -fcheck=all -fbacktrace -ffree-form -ffree-line-length-none -DSTUBMPI)
gfortran -c "${flags[@]}" "$script/spotting_services.F90"
python "$script/extract_spotting_columns.py" corrected.F columns.F90 > columns-extraction.json
gfortran -c "${flags[@]}" -I "$build" columns.F90
cp "$native/oracle_io.o" "$native/oracle_io.mod" .
for kind in original corrected; do
  mkdir -p "$kind"
  cd "$kind"
  module="$source_root/phys/module_firebrand_spotting.F"
  if [[ "$kind" == corrected ]]; then module="$build/corrected.F"; fi
  gfortran -c "${flags[@]}" -I "$build" "$module" -o module_firebrand_spotting.o
  gfortran -c "${flags[@]}" -I "$build" -I "$PWD" "$script/run_spotting_driver.F90"
  gfortran -o run "$build/spotting_services.o" "$build/oracle_io.o" "$build/columns.o" module_firebrand_spotting.o run_spotting_driver.o
  set +e
  nice -n 10 ./run "$PWD/fixtures" > run.log 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > run.status
  if [[ "$kind" == corrected && "$rc" != 0 ]]; then cat run.log; exit "$rc"; fi
  cd "$build"
done
gfortran --version | head -1 > compiler.txt
ldd --version | head -1 > libc.txt
printf '%s\n' "${flags[*]}" > compiler-flags.txt
(cd "$source_root" && sha256sum phys/module_firebrand_spotting.F phys/module_firebrand_spotting_mpi.F) > source-sha256sums.txt
(cd "$script" && sha256sum correct_spotting.py extract_spotting_columns.py spotting_services.F90 run_spotting_driver.F90 build_spotting_driver.sh) > oracle-sha256sums.txt
for kind in original corrected; do
  cp compiler.txt libc.txt compiler-flags.txt source-sha256sums.txt oracle-sha256sums.txt "$kind/"
  (cd "$kind" && find fixtures -type f -print0 | sort -z | xargs -0 sha256sum > fixture-sha256sums.txt)
done
