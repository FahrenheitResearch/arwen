#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ruc_source="${source_root}/phys/module_sf_ruclsm.F"

test -f "${ruc_source}"
mkdir -p "${build_dir}"
cd "${build_dir}"

gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none \
    "${script_dir}/stub_wrf.F90"
gfortran -c -O0 -cpp -DEM_CORE=0 -Dwrf_chem=0 \
    -fallow-argument-mismatch -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${ruc_source}"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_init.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_step.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_soilvegin.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_soilprop.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_transf.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_soilmoist.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_soiltemp.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_soil.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_snowtemp.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_snowsoil.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_snowseaice.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_qsn.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_sice.F90"
gfortran -o run_init stub_wrf.o module_sf_ruclsm.o run_init.o
gfortran -o run_step stub_wrf.o module_sf_ruclsm.o run_step.o
gfortran -o run_soilvegin stub_wrf.o module_sf_ruclsm.o run_soilvegin.o
gfortran -o run_soilprop stub_wrf.o module_sf_ruclsm.o run_soilprop.o
gfortran -o run_transf stub_wrf.o module_sf_ruclsm.o run_transf.o
gfortran -o run_soilmoist stub_wrf.o module_sf_ruclsm.o run_soilmoist.o
gfortran -o run_soiltemp stub_wrf.o module_sf_ruclsm.o run_soiltemp.o
gfortran -o run_soil stub_wrf.o module_sf_ruclsm.o run_soil.o
gfortran -o run_snowtemp stub_wrf.o module_sf_ruclsm.o run_snowtemp.o
gfortran -o run_snowsoil stub_wrf.o module_sf_ruclsm.o run_snowsoil.o
gfortran -o run_snowseaice stub_wrf.o module_sf_ruclsm.o run_snowseaice.o
gfortran -o run_qsn stub_wrf.o module_sf_ruclsm.o run_qsn.o
gfortran -o run_sice stub_wrf.o module_sf_ruclsm.o run_sice.o

cp "${source_root}/run/VEGPARM.TBL" .
cp "${source_root}/run/SOILPARM.TBL" .
cp "${source_root}/run/GENPARM.TBL" .
./run_init ruc-init.csv
./run_step ruc-step.csv
./run_soilvegin ruc-soilvegin.csv
./run_soilprop ruc-soilprop.csv
./run_transf ruc-transf.csv
./run_soilmoist ruc-soilmoist.csv
./run_soiltemp ruc-soiltemp.csv ruc-tbq.csv
./run_soil ruc-soil.csv
./run_snowtemp ruc-snowtemp.csv
./run_snowsoil ruc-snowsoil.csv
./run_snowseaice ruc-snowseaice.csv
./run_qsn ruc-qsn.csv
./run_sice ruc-sice.csv
export PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}"
python3 "${script_dir}/validate_oracle.py" ruc-init.csv
python3 "${script_dir}/validate_step_oracle.py" ruc-step.csv
python3 "${script_dir}/validate_soilvegin_oracle.py" ruc-soilvegin.csv
python3 "${script_dir}/validate_soilprop_oracle.py" ruc-soilprop.csv
python3 "${script_dir}/validate_transf_oracle.py" ruc-transf.csv
python3 "${script_dir}/validate_soilmoist_oracle.py" ruc-soilmoist.csv
python3 "${script_dir}/validate_soiltemp_oracle.py" ruc-soiltemp.csv
python3 "${script_dir}/validate_tbq_oracle.py" ruc-tbq.csv
python3 "${script_dir}/validate_soil_oracle.py" ruc-soil.csv
python3 "${script_dir}/validate_snowtemp_oracle.py" ruc-snowtemp.csv
python3 "${script_dir}/validate_snowsoil_oracle.py" ruc-snowsoil.csv
python3 "${script_dir}/validate_snowseaice_oracle.py" ruc-snowseaice.csv
python3 "${script_dir}/validate_qsn_oracle.py" ruc-qsn.csv
python3 "${script_dir}/validate_sice_oracle.py" ruc-sice.csv
sha256sum "${ruc_source}" "${script_dir}/stub_wrf.F90" \
    "${script_dir}/run_init.F90" "${script_dir}/run_step.F90" \
    "${script_dir}/run_soilvegin.F90" "${script_dir}/run_soilprop.F90" \
    "${script_dir}/run_transf.F90" "${script_dir}/run_soilmoist.F90" \
    "${script_dir}/run_soiltemp.F90" \
    "${script_dir}/run_soil.F90" \
    "${script_dir}/run_snowtemp.F90" "${script_dir}/run_snowsoil.F90" \
    "${script_dir}/run_snowseaice.F90" \
    "${script_dir}/run_qsn.F90" "${script_dir}/run_sice.F90" \
    "${script_dir}/validate_qsn_oracle.py" \
    "${script_dir}/validate_sice_oracle.py" \
    VEGPARM.TBL SOILPARM.TBL GENPARM.TBL \
    ruc-init.csv ruc-step.csv ruc-soilvegin.csv ruc-soilprop.csv \
    ruc-transf.csv ruc-soilmoist.csv ruc-soiltemp.csv ruc-tbq.csv \
    ruc-soil.csv ruc-snowtemp.csv ruc-snowsoil.csv ruc-snowseaice.csv \
    ruc-qsn.csv ruc-sice.csv \
    > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt

# --- snowseaice argument-contract oracle -------------------------------------
# oracle/snowseaice.csv holds myj, snowfrac, meltfactor, rainf, ilnb, snwe > 0
# and xlv flat across every row, so it cannot discriminate a port that drops
# any of them.  This second unmodified-module build moves each one off its
# identity or into the branch it selects.  It is appended rather than folded
# into the block above so the pinned oracle-sha256sums.txt ordering above is
# unchanged.
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_snowseaice_contract.F90"
gfortran -o run_snowseaice_contract \
    stub_wrf.o module_sf_ruclsm.o run_snowseaice_contract.o
./run_snowseaice_contract ruc-snowseaice-contract.csv
python3 "${script_dir}/validate_snowseaice_contract_oracle.py" \
    ruc-snowseaice-contract.csv
sha256sum "${script_dir}/run_snowseaice_contract.F90" \
    "${script_dir}/validate_snowseaice_contract_oracle.py" \
    ruc-snowseaice-contract.csv \
    >> oracle-sha256sums.txt

# --------------------------------------------------------------------------
# sfctmp snow preparation (phys/module_sf_ruclsm.F:1400-1766)
#
# The preparation block is straight-line code inside sfctmp with no argument
# list of its own, and the dispatch below it overwrites every result before
# sfctmp returns, so it is pinned by reading the running, UNMODIFIED module
# with gdb rather than by calling a subroutine.  That needs a -g build, which
# is kept in its own subdirectory so nothing above is disturbed; the physics
# source is still compiled exactly as it ships.
#
# Requires gdb (apt-get install gdb).  Everything above this line runs
# without it.
# --------------------------------------------------------------------------
prep_dir="${build_dir}/sfctmp_prep"
mkdir -p "${prep_dir}"
(
  cd "${prep_dir}"
  gfortran -c -O0 -g -cpp -ffree-form -ffree-line-length-none \
      "${script_dir}/stub_wrf.F90"
  gfortran -c -O0 -g -cpp -DEM_CORE=0 -Dwrf_chem=0 \
      -fallow-argument-mismatch -ffree-form -ffree-line-length-none \
      -I "${prep_dir}" "${ruc_source}"
  gfortran -c -O0 -g -ffree-form -ffree-line-length-none \
      -I "${prep_dir}" "${script_dir}/run_sfctmp_prep.F90"
  gfortran -g -o run_sfctmp_prep \
      stub_wrf.o module_sf_ruclsm.o run_sfctmp_prep.o
  cp "${source_root}/run/VEGPARM.TBL" "${source_root}/run/SOILPARM.TBL" \
      "${source_root}/run/GENPARM.TBL" .
  gdb -batch -x "${script_dir}/sfctmp_prep.gdb" > sfctmp-prep-gdb-stdout.txt 2>&1
  grep -E '^(case_index,|[0-9]+,)' sfctmp-prep-gdb.log > ruc-sfctmp-prep.csv
  cp ruc-sfctmp-prep.csv ruc-sfctmp-prep-inputs.csv ..
  gdb --version | head -1 > ../gdb-version.txt
)
export PYTHONPATH="${script_dir}/../..:${PYTHONPATH:-}"
python3 "${script_dir}/validate_sfctmp_prep_oracle.py" \
    ruc-sfctmp-prep.csv ruc-sfctmp-prep-inputs.csv
sha256sum "${script_dir}/run_sfctmp_prep.F90" \
    "${script_dir}/sfctmp_prep.gdb" \
    "${script_dir}/validate_sfctmp_prep_oracle.py" \
    ruc-sfctmp-prep.csv ruc-sfctmp-prep-inputs.csv \
    >> oracle-sha256sums.txt
