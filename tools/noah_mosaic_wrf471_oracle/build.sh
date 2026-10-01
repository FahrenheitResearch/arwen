#!/usr/bin/env bash
# WRF v4.7.1 Noah mosaic column oracle. All upstream inputs are hash-pinned.
# Usage: build.sh WRF_SOURCE_ROOT BUILD_DIR
# The core includes real urban modules for later sf_urban_physics=1 fixtures.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

core=(
    share/module_model_constants.F
    frame/module_wrf_error.F
    phys/module_sf_urban.F
    phys/module_bep_bem_helper.F
    phys/module_sf_bep.F
    phys/module_sf_bem.F
    phys/module_sf_bep_bem.F
    phys/module_sf_noahlsm.F
    phys/module_sf_noahlsm_glacial_only.F
    GFDLETA_CAL_MON_DAY
    phys/module_sf_noahdrv.F
)
tables=(run/URBPARM.TBL run/URBPARM_LCZ.TBL run/VEGPARM.TBL run/SOILPARM.TBL
        run/GENPARM.TBL run/LANDUSE.TBL)

extra=()
shopt -s nullglob
for list in "${script_dir}"/sources-*.list; do
    while IFS= read -r rel; do
        rel="${rel%%#*}"; rel="$(echo "${rel}" | xargs)"
        [[ -n "${rel}" ]] && extra+=("${rel}")
    done < "${list}"
done

# --- byte identity to the pinned sources ------------------------------------
pins=("${script_dir}"/SOURCES.sha256 "${script_dir}"/SOURCES-*.sha256)
declare -A pinned=()
for pin in "${pins[@]}"; do
    while read -r sum rel; do
        [[ -z "${sum}" || "${sum}" == \#* ]] && continue
        pinned["${rel}"]="${sum}"
    done < "${pin}"
done
check() {
    local rel="$1"
    if [[ -z "${pinned[${rel}]:-}" ]]; then
        echo "${rel} is compiled but pinned in no SOURCES*.sha256" >&2
        exit 3
    fi
    local have
    have=$(sha256sum "${source_root}/${rel}" | cut -d' ' -f1)
    if [[ "${have}" != "${pinned[${rel}]}" ]]; then
        echo "${rel}: sha256 ${have}, pinned ${pinned[${rel}]}" >&2
        exit 3
    fi
}
for rel in "${core[@]}" "${extra[@]}" "${tables[@]}" phys/module_ra_gfdleta.F; do
    [[ "${rel}" == GFDLETA_CAL_MON_DAY ]] && continue
    check "${rel}"
done
check phys/module_surface_driver.F

if [[ "${build_dir}" == "${source_root}" || "${build_dir}" == "${source_root}/"* ]]; then
    echo "BUILD_DIR inside WRF source would modify the read-only reference tree" >&2
    exit 7
fi
mkdir -p "${build_dir}"
cd "${build_dir}"

# Capture the actual post-SFCDIAGS UCM category predicate and assignments.
# Neither the surface driver nor the Noah driver is edited for this oracle.
sed -n '3004,3016p' "${source_root}/phys/module_surface_driver.F" \
    > mosaic_surface_driver_3004_3016.inc
cmp mosaic_surface_driver_3004_3016.inc \
    "${script_dir}/mosaic_surface_driver_3004_3016.inc"

defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4"
defines="${defines} -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"
fc="gfortran -c -O0 -cpp ${defines} ${free} -fallow-argument-mismatch -I ${build_dir}"

objects=(stub_wrf.o)
gfortran -c -O0 -cpp ${free} -fallow-argument-mismatch "${script_dir}/stub_wrf.F90"

compile_wrf() {
    local rel="$1"
    if [[ "${rel}" == GFDLETA_CAL_MON_DAY ]]; then
        # WRF's CAL_MON_DAY, extracted byte for byte, in a module of the name
        # module_sf_noahdrv USEs it from.
        {
            echo "module module_ra_gfdleta"
            echo "contains"
            awk '/^ *SUBROUTINE CAL_MON_DAY/,/^ *END SUBROUTINE CAL_MON_DAY/' \
                "${source_root}/phys/module_ra_gfdleta.F"
            echo "end module module_ra_gfdleta"
        } > module_ra_gfdleta_cal_mon_day.F
        ${fc} module_ra_gfdleta_cal_mon_day.F
        objects+=(module_ra_gfdleta_cal_mon_day.o)
        return
    fi
    local base
    base=$(basename "${rel}")
    base="${base%.*}"
    ${fc} -o "${base}.o" "${source_root}/${rel}"
    objects+=("${base}.o")
    nm -u "${base}.o" | sed 's/^ *//' | sort > "undefined-O0-${base}.txt"
    if grep -q '_ZGV' "undefined-O0-${base}.txt"; then
        echo "-O0 object ${base}.o pulled in a libmvec vector symbol:" >&2
        grep '_ZGV' "undefined-O0-${base}.txt" >&2
        exit 4
    fi
}
for rel in "${core[@]}" "${extra[@]}"; do
    compile_wrf "${rel}"
done

# Mosaic entry points must be exported by the actual upstream driver.
for pair in module_sf_noahdrv:lsm_mosaic module_sf_noahdrv:lsm_mosaic_init \
            module_sf_noahdrv:soil_veg_gen_parm; do
    obj="${pair%%:*}"; sym="${pair##*:}"
    if ! nm "${obj}.o" | grep -q "__${obj}_MOD_${sym}\$"; then
        echo "${obj}.o does not export ${sym}; is -Dwrfmodel set?" >&2
        exit 6
    fi
done

gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/oracle_io.F90"
objects+=(oracle_io.o)

# Positive control: the _ZGV grep must be able to fire on this toolchain.
gfortran -c -Ofast -ftree-vectorize \
    -o libmvec_positive_control.o "${script_dir}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol; the -O0 guard is" \
         "vacuous on this toolchain" >&2
    exit 5
fi

for table in "${tables[@]}"; do
    cp "${source_root}/${table}" .
done

drivers=("${script_dir}"/run_*.F90)
if [[ ${#drivers[@]} -eq 0 ]]; then
    echo "No run_*.F90 drivers: no oracle would execute" >&2
    exit 8
fi
mkdir -p fixtures
for driver in "${drivers[@]}"; do
    name=$(basename "${driver}" .F90)
    gfortran -c -O0 -cpp ${defines} ${free} -I "${build_dir}" "${driver}"
    gfortran -o "${name}" "${objects[@]}" "${name}.o"
    mkdir -p "fixtures/${name#run_}"
    "./${name}" "fixtures/${name#run_}" > "${name}-stdout.txt"
done

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 libm symbols per object (THE REFERENCE):"
    for f in undefined-O0-*.txt; do
        echo "## ${f#undefined-O0-}"
        grep -E 'expf|powf|logf|log10f|atanf|sqrtf|cbrtf|_ZGV' "${f}" || true
    done
    echo "# positive control (plain expf loop at -Ofast):"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt

# Check every -O0 object, including extracted CAL_MON_DAY and harness objects.
for obj in "${objects[@]}" run_*.o; do
    if nm -u "${obj}" | grep '_ZGV'; then
        echo "${obj}: -O0 libmvec symbol invalidates scalar float32 oracle" >&2
        exit 4
    fi
done
gfortran --version | head -1 > compiler.txt
{
    for pin in "${pins[@]}"; do cat "${pin}"; done
    sha256sum "${script_dir}"/*.F90 "${script_dir}/build.sh"
} > oracle-sha256sums.txt
echo "mosaic oracle built in ${build_dir}; fixtures in ${build_dir}/fixtures"
