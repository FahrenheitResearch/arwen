#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
common_source="${source_root}/phys/module_bl_mynn_common.F"
pbl_source="${source_root}/phys/module_bl_mynn.F"

test -f "${common_source}"
test -f "${pbl_source}"
mkdir -p "${build_dir}"
cd "${build_dir}"

gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none \
    "${script_dir}/stub_wrf.F90"
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${common_source}"
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${pbl_source}"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_level2.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_pblh_scale.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_mixlength.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_turbulence.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_predict.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_condensation.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_esat_blend.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_tendencies.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_initialize.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_dmp_mf.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_tendencies_mf.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_stfunc.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_driver.F90"
gfortran -o run_level2 stub_wrf.o module_bl_mynn_common.o \
    module_bl_mynn.o run_level2.o
gfortran -o run_pblh_scale stub_wrf.o module_bl_mynn_common.o \
    module_bl_mynn.o run_pblh_scale.o
gfortran -o run_mixlength stub_wrf.o module_bl_mynn_common.o \
    module_bl_mynn.o run_mixlength.o
gfortran -o run_turbulence stub_wrf.o module_bl_mynn_common.o \
    module_bl_mynn.o run_turbulence.o
gfortran -o run_predict stub_wrf.o module_bl_mynn_common.o \
    module_bl_mynn.o run_predict.o
gfortran -o run_condensation stub_wrf.o module_bl_mynn_common.o \
    module_bl_mynn.o run_condensation.o
gfortran -o run_esat_blend stub_wrf.o module_bl_mynn_common.o \
    module_bl_mynn.o run_esat_blend.o
gfortran -o run_tendencies stub_wrf.o module_bl_mynn_common.o \
    module_bl_mynn.o run_tendencies.o
gfortran -o run_initialize stub_wrf.o module_bl_mynn_common.o \
    module_bl_mynn.o run_initialize.o
gfortran -o run_dmp_mf stub_wrf.o module_bl_mynn_common.o \
    module_bl_mynn.o run_dmp_mf.o
gfortran -o run_tendencies_mf stub_wrf.o module_bl_mynn_common.o \
    module_bl_mynn.o run_tendencies_mf.o
gfortran -o run_stfunc stub_wrf.o module_bl_mynn_common.o \
    module_bl_mynn.o run_stfunc.o
gfortran -o run_driver stub_wrf.o module_bl_mynn_common.o \
    module_bl_mynn.o run_driver.o

./run_level2 pbl-level2.csv
./run_pblh_scale pblh-scale.csv
./run_mixlength mixlength.csv
./run_turbulence turbulence.csv
./run_predict predict.csv
./run_condensation condensation.csv
./run_esat_blend esat-blend.csv
./run_tendencies tendencies-nomf.csv
./run_initialize initialize.csv
./run_dmp_mf dmp-mf.csv
./run_tendencies_mf tendencies-mf.csv
./run_stfunc stfunc.csv
./run_driver driver.csv
python3 "${script_dir}/validate_oracle.py" pbl-level2.csv
python3 "${script_dir}/validate_pblh_oracle.py" pblh-scale.csv
python3 "${script_dir}/validate_mixlength_oracle.py" mixlength.csv
PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}" \
    python3 "${script_dir}/validate_turbulence_oracle.py" turbulence.csv
PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}" \
    python3 "${script_dir}/validate_predict_oracle.py" predict.csv
PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}" \
    python3 "${script_dir}/validate_condensation_oracle.py" condensation.csv
PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}" \
    python3 "${script_dir}/validate_esat_blend_oracle.py" esat-blend.csv
PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}" \
    python3 "${script_dir}/validate_tendencies_oracle.py" tendencies-nomf.csv
PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}" \
    python3 "${script_dir}/validate_initialize_oracle.py" initialize.csv
PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}" \
    python3 "${script_dir}/validate_dmp_mf_oracle.py" dmp-mf.csv
PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}" \
    python3 "${script_dir}/validate_tendencies_mf_oracle.py" tendencies-mf.csv
PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}" \
    python3 "${script_dir}/validate_stfunc_oracle.py" stfunc.csv
PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}" \
    python3 "${script_dir}/validate_driver_oracle.py" driver.csv
sha256sum "${common_source}" "${pbl_source}" \
    "${script_dir}/stub_wrf.F90" "${script_dir}/run_level2.F90" \
    "${script_dir}/run_pblh_scale.F90" "${script_dir}/run_mixlength.F90" \
    "${script_dir}/run_turbulence.F90" \
    "${script_dir}/run_predict.F90" \
    "${script_dir}/run_condensation.F90" \
    "${script_dir}/run_esat_blend.F90" \
    "${script_dir}/run_tendencies.F90" \
    "${script_dir}/run_initialize.F90" \
    "${script_dir}/run_dmp_mf.F90" \
    "${script_dir}/run_tendencies_mf.F90" \
    "${script_dir}/run_stfunc.F90" \
    "${script_dir}/run_driver.F90" \
    pbl-level2.csv pblh-scale.csv mixlength.csv turbulence.csv predict.csv \
    condensation.csv esat-blend.csv tendencies-nomf.csv initialize.csv \
    dmp-mf.csv tendencies-mf.csv stfunc.csv driver.csv \
    > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
