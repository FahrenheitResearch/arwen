#!/usr/bin/env bash
set -euo pipefail
source_root=$(realpath "$1")
native=$(realpath "$2")
build=$(realpath -m "$3")
script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$build" == "$source_root" || "$build" == "$source_root/"* ]]; then
  echo 'MPI control build must not change immutable WRF source' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$build"
cd "$build"
flags=(-O0 -cpp -ffp-contract=off -fcheck=all -fbacktrace -ffree-form -ffree-line-length-none -DDM_PARALLEL)
if [[ -n "${SFIRE_MPI_SDK:-}" ]]; then
  flags+=(-I "$SFIRE_MPI_SDK/usr/lib/x86_64-linux-gnu/fortran/gfortran-mod-16/openmpi"
          -I "$SFIRE_MPI_SDK/usr/lib/x86_64-linux-gnu/openmpi/include")
  export LIBRARY_PATH="$SFIRE_MPI_SDK/usr/lib/x86_64-linux-gnu/openmpi/lib:${LIBRARY_PATH:-}"
fi
mpifort -c "${flags[@]}" "$script/spotting_mpi_services.F90"
mpifort -c "${flags[@]}" -I "$build" "$source_root/phys/module_firebrand_spotting_mpi.F"
mpifort -c "${flags[@]}" -I "$native" -I "$build" "$script/run_spotting_mpi.F90"
mpifort -o run "$native/oracle_io.o" spotting_mpi_services.o module_firebrand_spotting_mpi.o run_spotting_mpi.o
# Nine ranks run on two allowed physical cores. One thread per rank,
# no GPU context, and only small packets keep this CPU control bounded.
nice -n 10 taskset -c "${SFIRE_MPI_CPUS:-0,1}" mpirun --oversubscribe --bind-to none -np 9 ./run "$build/fixtures" > run.log 2>&1
mpifort --version | head -1 > compiler.txt
dpkg-query -W openmpi-bin > mpi.txt
ldd --version | head -1 > libc.txt
printf '%s\n' "${flags[*]}" > compiler-flags.txt
(cd "$source_root" && sha256sum phys/module_firebrand_spotting_mpi.F) > source-sha256sums.txt
(cd "$script" && sha256sum spotting_mpi_services.F90 run_spotting_mpi.F90 build_spotting_mpi.sh) > oracle-sha256sums.txt
find fixtures -type f -print0 | sort -z | xargs -0 sha256sum > fixture-sha256sums.txt
