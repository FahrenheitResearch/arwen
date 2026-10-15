#!/usr/bin/env bash
# Build and run the BEP (sf_urban_physics = 2) column oracles against WRF v4.6.1.
#
#   bash tools/urban_bep_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT is a clone of wrf-model/WRF tag v4.6.1 (commit
# d66e442fccc04111067e29274c9f9eaccc3cef28).  Every file read from it must
# match SOURCES.sha256 here; a mismatch stops the build before anything
# compiles.
#
# What it compiles, byte-unmodified, at -O0 with WRF's phys/ defines:
# module_model_constants, module_wrf_error, module_sf_urban,
# module_bep_bem_helper, module_sf_bep.  The coupling blocks the LSM drivers
# run after BEP are cut out of the v4.6.1 files by line range (they sit at
# different line numbers than in v4.7.1) and must be byte-identical to the
# v4.7.1 cuts tools/urban_wrf471_oracle INCLUDEs; the script checks that.
#
# What it runs: run_bep461_{nlcd,lcz} (SUBROUTINE BEP, 4 steps) and
# run_bep461_couple_{nlcd,lcz} (zeroing + BEP + Noah and Noah-MP coupling
# blocks + the surface-diagnostic override, 2 steps).  Fixtures land in
# BUILD_DIR/fixtures/<name>/; gpuwm.verify.urban_oracle.load_case reads them
# and tools/urban_bep_wrf461_oracle/compare.py grades the GPU port.
#
# ORACLE_LINK_EXTRA (optional) names extra objects linked into every driver
# ahead of the C library; build_crlibm_arm.sh uses it to link correctly
# rounded acosf/asinf/atanf/tanf (see README.md).  compiler.txt records which
# math symbols each driver defines (T) or takes from the C library (U).
#
# libmvec: the reference is -O0; the script fails if any -O0 object pulls in a
# glibc vector-math (_ZGV*) symbol, and a positive control proves the grep can
# see one on this toolchain.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
shared="${here}/../urban_wrf471_oracle"

# --- byte identity to the pinned v4.6.1 sources ---------------------------
while read -r sum rel; do
    [[ -z "${sum}" || "${sum}" == \#* ]] && continue
    have=$(sha256sum "${source_root}/${rel}" | cut -d' ' -f1)
    if [[ "${have}" != "${sum}" ]]; then
        echo "${rel}: sha256 ${have}, pinned ${sum}" >&2
        exit 3
    fi
done < "${here}/SOURCES.sha256"

mkdir -p "${build_dir}"
cd "${build_dir}"

# --- the coupling blocks, cut from v4.6.1 ---------------------------------
sed -n 1679,1776p "${source_root}/phys/module_sf_noahdrv.F" > couple_noahdrv_461.inc
{ sed -n 3364,3373p "${source_root}/phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F"
  sed -n 3690,3777p "${source_root}/phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F"; } \
    > couple_noahmpdrv_461.inc
sed -n 3027,3031p "${source_root}/phys/module_surface_driver.F" > sfcdiag_surface_driver_461.inc
for pair in couple_noahdrv_461.inc:couple_noahdrv_1679_1776.inc \
            couple_noahmpdrv_461.inc:couple_noahmpdrv_3363_3372_3689_3776.inc \
            sfcdiag_surface_driver_461.inc:sfcdiag_surface_driver_3028_3032.inc; do
    if ! cmp -s "${pair%%:*}" "${shared}/${pair##*:}"; then
        echo "${pair%%:*} differs from the v4.7.1 cut ${pair##*:}" >&2
        exit 7
    fi
done
sha256sum couple_noahdrv_461.inc couple_noahmpdrv_461.inc \
    sfcdiag_surface_driver_461.inc > include-sha256.txt

defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4"
defines="${defines} -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"
fc="gfortran -c -O0 -cpp ${defines} ${free} -fallow-argument-mismatch -I ${build_dir}"

gfortran -c -O0 -cpp ${free} -fallow-argument-mismatch "${shared}/stub_wrf.F90"
objects=(stub_wrf.o)
for rel in share/module_model_constants.F frame/module_wrf_error.F \
           phys/module_sf_urban.F phys/module_bep_bem_helper.F phys/module_sf_bep.F; do
    base=$(basename "${rel}" .F)
    ${fc} -o "${base}.o" "${source_root}/${rel}"
    objects+=("${base}.o")
    nm -u "${base}.o" | sed 's/^ *//' | sort > "undefined-O0-${base}.txt"
    if grep -q '_ZGV' "undefined-O0-${base}.txt"; then
        echo "-O0 object ${base}.o pulled in a libmvec vector symbol" >&2
        exit 4
    fi
done
nm module_sf_bep.o | grep -q '__module_sf_bep_MOD_bep$' \
    || { echo "module_sf_bep.o does not export bep; is -Dwrfmodel set?" >&2; exit 6; }

gfortran -c -O0 ${free} -I "${build_dir}" "${shared}/oracle_io.F90"
objects+=(oracle_io.o)

gfortran -c -Ofast -ftree-vectorize -o libmvec_positive_control.o \
    "${shared}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
grep -q '_ZGV' undefined-control.txt \
    || { echo "libmvec positive control produced no _ZGV symbol" >&2; exit 5; }

cp "${source_root}/run/URBPARM.TBL" "${source_root}/run/URBPARM_LCZ.TBL" .

mkdir -p fixtures
for driver in "${here}"/run_bep461_*.F90; do
    name=$(basename "${driver}" .F90)
    gfortran -c -O0 -cpp ${defines} ${free} -I "${here}" -I "${build_dir}" "${driver}"
    gfortran -o "${name}" "${objects[@]}" "${name}.o" ${ORACLE_LINK_EXTRA:-}
    rm -rf "fixtures/${name#run_}"
    mkdir -p "fixtures/${name#run_}"
    "./${name}" "fixtures/${name#run_}" > "${name}-stdout.txt"
    cat "${name}-stdout.txt"
done

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 libm symbols per object (THE REFERENCE):"
    for f in undefined-O0-*.txt; do
        echo "## ${f#undefined-O0-}"
        grep -E 'expf|powf|logf|log10f|atanf|sqrtf|cbrtf|sinf|cosf|_ZGV' "${f}" || true
    done
    echo "# positive control (plain expf loop at -Ofast):"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt
{
    gfortran --version | head -1
    ldd --version | head -1
    echo "extra link objects: ${ORACLE_LINK_EXTRA:-none}"
    for driver in "${here}"/run_bep461_*.F90; do
        name=$(basename "${driver}" .F90)
        echo "${name}: $(nm "${name}" | grep -E ' [TU] (acosf|asinf|atanf|tanf|sinf|cosf|expf|logf|powf)(@|$)' | tr -s ' ' | tr '
' ';')"
    done
} > compiler.txt
{
    cat "${here}/SOURCES.sha256"
    sha256sum "${here}"/*.F90 "${here}"/*.inc "${here}/build.sh" \
        "${shared}/oracle_io.F90" "${shared}/stub_wrf.F90"
    cat include-sha256.txt
} > oracle-sha256sums.txt
(cd fixtures && find . -type f -name '*.bin' | sort | xargs sha256sum) > fixtures-sha256.txt
echo "urban BEP v4.6.1 oracle built in ${build_dir}; fixtures in ${build_dir}/fixtures"
