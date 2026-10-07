#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
    echo 'usage: build.sh WRF_SOURCE CONFIGURE_MOD NATIVE_FIRE_ORACLE_BUILD NEW_BUILD' >&2
    exit 2
fi
source=$(realpath "$1")
config=$(realpath "$2")
native=$(realpath "$3")
build=$(realpath -m "$4")
script=$(cd "$(dirname "$0")" && pwd)
if [[ "$build" == "$source" || "$build" == "$source/"* ]]; then
    echo 'The oracle build must not change the immutable WRF source' >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$build"
python3 "$script/../../bigstep_wrf471_oracle/prep_build.py" "$source/dyn_em/module_big_step_utilities_em.F" "$build"
python3 "$script/extract.py" "$build/prep_exact.F90" "$build"
cd "$build"
cp "$config" module_configure.mod
cp "$native/module_model_constants.mod" "$native/oracle_io.mod" .
flags=(-O0 -cpp -ffp-contract=off -ffree-form -ffree-line-length-none -fcheck=all -fbacktrace)
gfortran "${flags[@]}" -c rhs_native.F90 "$script/run.F90"
gfortran -o run_open_phi "$native/module_model_constants.o" "$native/oracle_io.o" rhs_native.o run.o
nice -n 10 ./run_open_phi "$build/fixtures" > run.log
gfortran --version | head -1 > compiler.txt
ldd --version | head -1 > libc.txt
printf '%s\n' "${flags[*]}" > compiler-flags.txt
(cd "$source" && sha256sum dyn_em/module_big_step_utilities_em.F share/module_model_constants.F) > source-sha256sums.txt
(cd "$script" && sha256sum build.sh extract.py run.F90) > oracle-sha256sums.txt
sha256sum module_configure.mod rhs_native.F90 run_open_phi > artifacts.sha256
find fixtures -type f -print0 | sort -z | xargs -0 sha256sum > fixture-sha256sums.txt
echo 'Native open geopotential reference completed'
