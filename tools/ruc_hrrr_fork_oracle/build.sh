#!/usr/bin/env bash
# Rebuild gpuwm/data/ruc/oracle_fork/: the WRF v4.6.1 RUC oracle harnesses
# (tools/ruc_wrf461_oracle/run_*.F90, unchanged) compiled against the
# operational RAP/HRRR branch's module_sf_ruclsm.F instead of WRF v4.6.1's.
#
#   build.sh HRRR_WRF_ROOT BUILD_DIR [OUT_DIR]
#
# HRRR_WRF_ROOT is the branch's WRF tree, NOAA-EMC/HRRR v4.1.21
# sorc/hrrr_wrfarw.fd/WRFV3.9: phys/module_sf_ruclsm.F and the run/ tables
# (VEGPARM.TBL, SOILPARM.TBL, GENPARM.TBL).  OUT_DIR defaults to BUILD_DIR/out
# and receives the CSVs under the names tests/test_ruc_fork_oracle.py reads.
# The harness inputs are the v4.6.1 oracle's, so each case runs the same
# column through the other lineage.  soilvegin is not built: its harness
# passes an argument (myj) the branch's SOILVEGIN does not take, and the two
# SOILVEGINs are otherwise identical.
#
# Reproducibility, measured with gfortran 15.2: every CSV is byte-identical
# to the committed one except snowtemp_contract.csv's qcg_before column in the
# melt_dense_dry_evap case, which the harness prints from a local it never
# sets (stack residue, 0 in the committed file).  snowtemp overwrites qcg
# before reading it, so no output moves.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh HRRR_WRF_ROOT BUILD_DIR [OUT_DIR]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
out_dir=$(realpath -m "${3:-${build_dir}/out}")
harness=$(cd "$(dirname "${BASH_SOURCE[0]}")/../ruc_wrf461_oracle" && pwd)
ruc_source="${source_root}/phys/module_sf_ruclsm.F"
test -f "${ruc_source}"
mkdir -p "${build_dir}" "${out_dir}"
cd "${build_dir}"

flags="-O0 -ffree-form -ffree-line-length-none"
gfortran -c ${flags} -cpp "${harness}/stub_wrf.F90"
gfortran -c ${flags} -cpp -DEM_CORE=0 -Dwrf_chem=0 -fallow-argument-mismatch \
    -I "${build_dir}" "${ruc_source}"
for name in step soilprop soil snowtemp snowsoil sfctmp lsmruc \
            snowtemp_contract snowsoil_contract; do
    gfortran -c ${flags} -I "${build_dir}" "${harness}/run_${name}.F90"
    gfortran -o "run_${name}" stub_wrf.o module_sf_ruclsm.o "run_${name}.o"
done
cp "${source_root}/run/VEGPARM.TBL" "${source_root}/run/SOILPARM.TBL" \
    "${source_root}/run/GENPARM.TBL" .

./run_step "${out_dir}/step.csv"
./run_soilprop "${out_dir}/soilprop.csv"
./run_soil "${out_dir}/soil.csv"
./run_snowtemp "${out_dir}/snowtemp.csv"
./run_snowsoil "${out_dir}/snowsoil.csv"
./run_snowtemp_contract "${out_dir}/snowtemp_contract.csv"
./run_snowsoil_contract "${out_dir}/snowsoil_contract.csv"
./run_sfctmp "${out_dir}/sfctmp.csv"
./run_sfctmp "${out_dir}/sfctmp_stackfill.csv" 12345.0
./run_lsmruc "${out_dir}/lsmruc.csv"
./run_lsmruc "${out_dir}/lsmruc_stackfill.csv" 12345.0

( sha256sum "${ruc_source}" VEGPARM.TBL SOILPARM.TBL GENPARM.TBL
  cd "${out_dir}" && sha256sum *.csv ) > "${out_dir}/oracle-sha256sums.txt"
gfortran --version | head -1 > "${out_dir}/compiler.txt"
