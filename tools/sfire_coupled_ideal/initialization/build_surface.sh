#!/usr/bin/env bash
set -euo pipefail
source_root=$(realpath "$1")
native=$(realpath "$2")
build=$(realpath -m "$3")
script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$build" == "$source_root" || "$build" == "$source_root/"* ]]; then
  echo 'Surface control must not modify immutable WRF source' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$build"
cd "$build"
python "$script/extract_surface.py" "$source_root/dyn_em/module_initialize_fire.F" control.F90 > extraction.json
flags=(-O0 -cpp -ffp-contract=off -fcheck=all -fbacktrace -ffree-form -ffree-line-length-none)
gfortran -c "${flags[@]}" -I "$native" control.F90
gfortran -c "${flags[@]}" -I "$native" "$script/run_surface.F90"
gfortran -o run "$native/oracle_io.o" control.o run_surface.o
nice -n 10 ./run "$build/fixtures" > run.log
gfortran --version | head -1 > compiler.txt
ldd --version | head -1 > libc.txt
printf '%s\n' "${flags[*]}" > compiler-flags.txt
(cd "$source_root" && sha256sum dyn_em/module_initialize_fire.F) > source-sha256sums.txt
(cd "$script" && sha256sum extract_surface.py run_surface.F90 build_surface.sh) > oracle-sha256sums.txt
find fixtures -type f -print0 | sort -z | xargs -0 sha256sum > fixture-sha256sums.txt
