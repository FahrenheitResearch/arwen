#!/usr/bin/env bash
# Build and run the urban WRF v4.7.1 column oracles.
#
#   bash tools/urban_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT is a clone of wrf-model/WRF tag v4.7.1 with phys/noahmp at
# e5c0859874407859936739e8be8741f9aed369ee and phys/physics_mmm at MMM-physics
# tag 20240626-MPASv8.2 (WRF's arch/Externals.cfg).  The tree is NOT checked by
# commit: every file this script compiles or copies must match its SHA-256 in
# SOURCES.sha256 (or in a lane's SOURCES-<lane>.sha256), and a mismatch stops
# the build before anything is compiled.
#
# WHAT IT COMPILES.  The core set below, from the pinned tree, byte-unmodified,
# at -O0 with WRF's own phys/ defines (arch/postamble): the constants, the error
# module, the three urban models and their helper, Noah and the Noah driver
# that calls them, and myjurb.  Then every extra WRF source a lane lists in a
# `sources-<lane>.list` file here (paths relative to WRF_SOURCE_ROOT, compiled
# in listed order, each hash-pinned), then oracle_io.F90, stub_wrf.F90 and
# every run_*.F90 in this directory.  A lane adds a driver or a source by
# adding files, never by editing this script.
#
# WHAT IT RUNS.  Each run_<name> executable, with BUILD_DIR/fixtures/<name> as
# its one argument (oracle_io's root).  Copy the fixture directory to
# gpuwm/data/urban/oracle/<lane>/ to publish it.
#
# libmvec.  WRF compiles phys/ at -O2 -ftree-vectorize, where gfortran can swap
# scalar expf/powf/logf for glibc's 4-ULP vector forms.  The reference is -O0
# and the script FAILS if any _ZGV* symbol appears in an -O0 object; a
# positive control proves the grep can see one on this toolchain at all.
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
    phys/module_bl_myjurb.F
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

mkdir -p "${build_dir}"
cd "${build_dir}"

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

# The urban routines the drivers call must actually be exported.
for pair in module_sf_urban:urban_param_init module_sf_urban:urban_var_init \
            module_sf_urban:urban module_sf_bep:bep module_sf_bep_bem:bep_bem \
            module_sf_noahdrv:lsm; do
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
mkdir -p fixtures
for driver in "${drivers[@]}"; do
    name=$(basename "${driver}" .F90)
    gfortran -c -O0 -cpp ${defines} ${free} -I "${build_dir}" "${driver}"
    gfortran -o "${name}" "${objects[@]}" "${name}.o"
    rm -rf "fixtures/${name#run_}"
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

gfortran --version | head -1 > compiler.txt
{
    for pin in "${pins[@]}"; do cat "${pin}"; done
    sha256sum "${script_dir}"/*.F90 "${script_dir}/build.sh"
} > oracle-sha256sums.txt
echo "urban oracle built in ${build_dir}; fixtures in ${build_dir}/fixtures"
