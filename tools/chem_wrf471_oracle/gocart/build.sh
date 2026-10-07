#!/usr/bin/env bash
# Build and run the GOCART (chem_opt=300) column oracles against the
# byte-unmodified WRF v4.7.1 sources.
#
#   bash tools/chem_wrf471_oracle/gocart/build.sh WRF_SRC BUILD_DIR [FIXTURE_ROOT]
#
# WRF_SRC is a folder laid out like the WRF tree holding at least the files in
# SOURCES below (the program's wrf-src/ plus wrf-src/extra/ merged, or a WRF
# v4.7.1 checkout).  Every compiled WRF file is checked against its sha256
# before anything is compiled; a mismatch stops the build.
#
# What is compiled from WRF: the GOCART process modules and the data modules
# they USE, and share/module_model_constants.F (the constants the reference is
# measured in).  What is stubbed (stub_gocart.F90): the registry-generated
# module_state_description / module_configure (declarations only, values from
# the gocart_simple package), a calc_zenith that aborts if reached, and the
# wrf_debug / wrf_message / wrf_error_fatal service hooks.
#
# -O0, no vectoriser: WRF builds chem/ at -O2 -ftree-vectorize where gfortran
# may call glibc's libmvec vector transcendentals (4 ULP contract) instead of
# the scalar ones.  The reference is the -O0 object; this script fails on any
# _ZGV* symbol in it, and a positive control proves the check can fire.
#
# Every run_gocart_*.F90 in this directory is compiled against the objects and
# run once with FIXTURE_ROOT/<x>/ as its output root (<x> = the name after
# run_gocart_), so a lane adds a driver without editing this script.
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "usage: build.sh WRF_SRC BUILD_DIR [FIXTURE_ROOT]" >&2
    exit 2
fi
src=$(realpath "$1")
build=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fixtures=$(realpath -m "${3:-${build}/fixtures}")

# path  sha256 (WRF v4.7.1, commit f52c197ed39d12e087d02c50f412d90d418f6186)
SOURCES=$(cat "${here}/SOURCES.sha256")
while read -r sum rel; do
    [[ -z "${sum}" || "${sum}" == \#* ]] && continue
    got=$(sha256sum "${src}/${rel}" | cut -d' ' -f1)
    if [[ "${got}" != "${sum}" ]]; then
        echo "${rel}: sha256 ${got}, pinned ${sum}" >&2
        exit 3
    fi
done <<< "${SOURCES}"

mkdir -p "${build}" "${fixtures}"
# Byte-exact depvel extraction for the sink driver's aer_res_def input.
# module_dep_simple.F is checked in SOURCES.sha256 above.
sed -n '1520,1616p' "${src}/chem/module_dep_simple.F" > "${build}/sink_depvel.inc"
cd "${build}"
defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none -fallow-argument-mismatch"
fc="gfortran -c -O0 -cpp ${defines} -I ${build}"

${fc} ${free} "${here}/stub_gocart.F90"
${fc} ${free} "${here}/oracle_io.F90"
objs=(stub_gocart.o oracle_io.o)
compile() {  # compile one WRF file (free form) and remember its object
    local rel="$1" obj
    obj=$(basename "${rel}" .F).o
    ${fc} ${free} -o "${obj}" "${src}/${rel}"
    objs+=("${obj}")
}
compile share/module_model_constants.F
compile phys/module_data_gocart_dust.F
compile chem/module_data_gocart_seas.F
compile chem/module_data_gocartchem.F
compile chem/module_data_radm2.F
compile chem/module_data_sorgam.F
compile chem/module_gocart_dust.F
compile chem/module_gocart_dust_afwa.F
compile chem/module_gocart_seasalt.F
compile chem/module_gocart_settling.F
compile chem/module_gocart_drydep.F
compile chem/module_gocart_aerosols.F
compile chem/module_gocart_chem.F

: > undefined-O0.txt
for o in "${objs[@]}"; do nm -u "${o}" | sed 's/^ *//' >> undefined-O0.txt; done
sort -u -o undefined-O0.txt undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "an -O0 object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-O0.txt >&2
    exit 4
fi
gfortran -c -Ofast -ftree-vectorize -o libmvec_positive_control.o \
    "${here}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol; the -O0 check is vacuous" >&2
    exit 5
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# libm symbols the -O0 GOCART objects call (THE REFERENCE):"
    awk '{print $2}' undefined-O0.txt | grep -E '^(exp|log|log10|pow|sqrt|sin|cos|acos|atan|cbrt)f?$|^__powi[sd]f2$|_ZGV' || true
    echo "# positive control:"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt
cat libmvec-report.txt

shopt -s nullglob
for drv in "${here}"/run_gocart_*.F90; do
    name=$(basename "${drv}" .F90)
    key=${name#run_gocart_}
    # A driver whose build_<key>.sh is its own build (it compiles a module set
    # this script does not, as the optics driver does) runs through that
    # script, which checks its own pinned sources: linking it against the
    # objects above would fail on the modules it USEs.
    if [[ -f "${here}/build_${key}.sh" ]] && ! grep -q '/build.sh"' "${here}/build_${key}.sh"; then
        bash "${here}/build_${key}.sh" "${src}" "${build}/${key}" "${fixtures}/${key}" "${build}/${key}-tables"
        echo "fixture ${key}: $(find "${fixtures}/${key}" -name MANIFEST.txt | wc -l) cases (standalone build_${key}.sh)"
        continue
    fi
    ${fc} ${free} -o "${name}.o" "${drv}"
    gfortran -o "${name}" "${name}.o" "${objs[@]}"
    mkdir -p "${fixtures}/${key}"
    ( cd "${fixtures}/${key}" && "${build}/${name}" "${fixtures}/${key}" > "${build}/${name}-stdout.txt" )
    ( cd "${fixtures}/${key}" && find . -name '*.bin' -o -name 'MANIFEST.txt' | sort | xargs sha256sum > oracle-sha256sums.txt )
    echo "fixture ${key}: $(find "${fixtures}/${key}" -name MANIFEST.txt | wc -l) cases"
done
gfortran --version | head -1 > compiler.txt
echo "gocart oracle built in ${build}, fixtures in ${fixtures}"
