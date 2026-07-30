#!/usr/bin/env bash
# Standalone Noah-MP oracle harness against unmodified WRF v4.6.1 sources.
#
# The four pinned physics sources plus share/module_model_constants.F are
# compiled byte-for-byte as they ship.  stub_wrf.F90 supplies only services
# (logging, fatal stop, an opaque domain handle, and abort-on-call link targets
# for the urban/BEP packages this harness does not build).
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)

constants_source="${source_root}/share/module_model_constants.F"
gecros_source="${source_root}/phys/module_sf_gecros.F"
lsm_source="${source_root}/phys/module_sf_noahmplsm.F"
glacier_source="${source_root}/phys/module_sf_noahmp_glacier.F"
groundwater_source="${source_root}/phys/module_sf_noahmp_groundwater.F"
drv_source="${source_root}/phys/module_sf_noahmpdrv.F"

for f in "${constants_source}" "${gecros_source}" "${lsm_source}" \
         "${glacier_source}" "${groundwater_source}" "${drv_source}"; do
    test -f "${f}"
done

mkdir -p "${build_dir}"
cd "${build_dir}"

fflags=(-O0 -cpp -ffree-form -ffree-line-length-none)

gfortran -c "${fflags[@]}" "${script_dir}/stub_wrf.F90"
gfortran -c "${fflags[@]}" "${constants_source}"
gfortran -c "${fflags[@]}" "${gecros_source}"
gfortran -c "${fflags[@]}" -I "${build_dir}" "${lsm_source}"
gfortran -c "${fflags[@]}" -I "${build_dir}" "${glacier_source}"
gfortran -c "${fflags[@]}" -I "${build_dir}" "${groundwater_source}"
# EM_CORE=0 compiles out the DM_PARALLEL halo exchange in GROUNDWATER_INIT;
# -fallow-argument-mismatch matches WRF's own build of this file.
gfortran -c "${fflags[@]}" -DEM_CORE=0 -fallow-argument-mismatch \
    -I "${build_dir}" "${drv_source}"

gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_parameters.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_sflx.F90"

physics_objects="stub_wrf.o module_model_constants.o module_sf_gecros.o \
module_sf_noahmplsm.o module_sf_noahmp_glacier.o module_sf_noahmp_groundwater.o \
module_sf_noahmpdrv.o"

# shellcheck disable=SC2086
gfortran -o run_parameters ${physics_objects} run_parameters.o
# shellcheck disable=SC2086
gfortran -o run_sflx ${physics_objects} run_sflx.o

# The oracle consumes the byte-pinned gpuwm parameter assets, not the WRF
# checkout's run/ copies, so the oracle and gpuwm/core/noahmp.py parse the
# identical bytes.
cp "${repo_root}/gpuwm/data/noahmp/MPTABLE.TBL" .
cp "${repo_root}/gpuwm/data/noahmp/SOILPARM.TBL" .
cp "${repo_root}/gpuwm/data/noahmp/GENPARM.TBL" .
sha256sum -c --status <<'EOF'
7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d  MPTABLE.TBL
1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f  SOILPARM.TBL
9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2  GENPARM.TBL
EOF

./run_parameters noahmp-parameters.csv
./run_sflx noahmp-sflx.csv

python3 "${script_dir}/validate_parameters_oracle.py" noahmp-parameters.csv
python3 "${script_dir}/validate_sflx_oracle.py" noahmp-sflx.csv

sha256sum "${constants_source}" "${gecros_source}" "${lsm_source}" \
    "${glacier_source}" "${groundwater_source}" "${drv_source}" \
    "${script_dir}/stub_wrf.F90" \
    "${script_dir}/run_parameters.F90" "${script_dir}/run_sflx.F90" \
    MPTABLE.TBL SOILPARM.TBL GENPARM.TBL \
    noahmp-parameters.csv noahmp-sflx.csv \
    > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
cat oracle-sha256sums.txt compiler.txt
