#!/usr/bin/env bash
# Build and run the Noah sea-ice oracle against the byte-unmodified WRF v4.6.1
# phys/module_sf_noah_seaice_drv.F (seaice_noah) and module_sf_noah_seaice.F
# (SFLX_SEAICE), which USE module_sf_noahlsm (HSTEP, ROSR12 and the Noah
# constants), module_model_constants and module_wrf_error.  All five are
# compiled from the pinned tree.  The service stubs are those of the Noah LSM
# oracle (tools/noah_wrf461_oracle/stub_wrf.F90), plus stub_state.F90: the
# three urban package constants seaice_noah's sf_urban_physics test reads from
# module_state_description, a Registry-generated module that does not exist
# outside a configured WRF build.  run_seaice.F90 passes sf_urban_physics = 0,
# so no fixture row reaches the branch those constants select.
#
# Built at -O0 with WRF's own defines, and the build FAILS on any libmvec
# _ZGV* symbol in the -O0 objects, as the Noah LSM oracle does.
#
#   bash tools/noah_seaice_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lsm_dir="${script_dir}/../noah_wrf461_oracle"
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"

sources=(
    share/module_model_constants.F
    frame/module_wrf_error.F
    phys/module_sf_noahlsm.F
    phys/module_sf_noah_seaice.F
    phys/module_sf_noah_seaice_drv.F
)
for rel in "${sources[@]}"; do
    test -f "${source_root}/${rel}"
done
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${sources[@]}"; then
    echo "one of the sea-ice sources differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4"
defines="${defines} -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"

gfortran -c -O0 -cpp ${free} -fallow-argument-mismatch "${lsm_dir}/stub_wrf.F90"
gfortran -c -O0 ${free} "${script_dir}/stub_state.F90"
gfortran -c -O0 -cpp ${defines} ${free} \
    "${source_root}/share/module_model_constants.F"
gfortran -c -O0 -cpp ${defines} ${free} -fallow-argument-mismatch \
    "${source_root}/frame/module_wrf_error.F"
for src in module_sf_noahlsm module_sf_noah_seaice module_sf_noah_seaice_drv; do
    gfortran -c -O0 -cpp ${defines} ${free} -fallow-argument-mismatch \
        -I "${build_dir}" "${source_root}/phys/${src}.F"
done
gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/run_seaice.F90"
gfortran -o run_seaice stub_wrf.o stub_state.o module_model_constants.o \
    module_wrf_error.o module_sf_noahlsm.o module_sf_noah_seaice.o \
    module_sf_noah_seaice_drv.o run_seaice.o

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    for obj in module_sf_noahlsm module_sf_noah_seaice module_sf_noah_seaice_drv; do
        echo "# -O0 ${obj} libm symbols:"
        nm -u "${obj}.o" | sed 's/^ *//' | grep -E 'expf|powf|logf|_ZGV' || true
    done
} > libmvec-report.txt
if grep -q '_ZGV' libmvec-report.txt; then
    echo "an -O0 object pulled in a libmvec vector symbol" >&2
    exit 4
fi

./run_seaice noah-seaice.csv          F 0.65 > run-seaice-stdout.txt
./run_seaice noah-seaice-frpcpn.csv   T 0.65 > run-seaice-frpcpn-stdout.txt
./run_seaice noah-seaice-alb075.csv   F 0.75 > run-seaice-alb075-stdout.txt
for variant in frpcpn alb075; do
    if cmp -s noah-seaice.csv "noah-seaice-${variant}.csv"; then
        echo "noah-seaice-${variant}.csv is byte-identical to noah-seaice.csv" >&2
        exit 7
    fi
done

sha256sum "${source_root}/share/module_model_constants.F" \
    "${source_root}/frame/module_wrf_error.F" \
    "${source_root}/phys/module_sf_noahlsm.F" \
    "${source_root}/phys/module_sf_noah_seaice.F" \
    "${source_root}/phys/module_sf_noah_seaice_drv.F" \
    "${lsm_dir}/stub_wrf.F90" \
    "${script_dir}/stub_state.F90" \
    "${script_dir}/run_seaice.F90" \
    "${script_dir}/build.sh" \
    noah-seaice.csv noah-seaice-frpcpn.csv noah-seaice-alb075.csv \
    | sed -e "s#${source_root}#<wrf-4.6.1-checkout>#" \
          -e "s#${script_dir}/../noah_wrf461_oracle#<oracle-workspace>/tools/noah_wrf461_oracle#" \
          -e "s#${script_dir}#<oracle-workspace>/tools/noah_seaice_wrf461_oracle#" \
    > oracle-sha256sums.txt
echo "oracle built in ${build_dir}"
