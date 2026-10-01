#!/usr/bin/env bash
# Build and run the WRF v4.7.1 UW PBL (bl_pbl_physics=9) column oracle.
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR CASES_DIR
#
# WRF_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh
# writes such a tree); every one is checked against its pin before anything
# compiles, so a fixture cannot come from an edited file.  CASES_DIR holds
# the case files make_cases.py wrote.  Four builds, each in its own
# directory because Fortran module files would collide:
#
#   pristine  the reference: byte-unmodified WRF sources, gfortran -O0, the
#             WRF preprocessor defines for an EM, RWORDSIZE=4 build.  Its
#             output is the fixture of record.
#   o2        the same sources at WRF's own gfortran optimisation
#             (configure.defaults: -O2 -ftree-vectorize -funroll-loops).
#             Evidence only: the script reports whether its outputs equal
#             the reference's byte for byte on every case.
#   snan      the reference with every local real initialised to a
#             signalling NaN and every local integer to -999999.  If the
#             scheme read an uninitialised local on the path the cases
#             take, the outputs would change; the script requires they do
#             not.
#   spy       the reference with make_spy.py's read-only stage recorders.
#             The script requires its step outputs to equal the reference's
#             byte for byte, which is what makes its stage records usable as
#             fixtures for the private routines.
#
# libmvec guard (the pattern of the other oracles in tools/): the -O0
# objects must not reference any _ZGV* vector symbol; a positive control
# proves the grep can fire on this toolchain.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR CASES_DIR" >&2
    exit 2
fi
src=$(realpath "$1")
out=$(realpath -m "$2")
cases=$(realpath "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

( cd "${src}" && grep -v '^#' "${here}/SOURCES.sha256" | sha256sum -c --quiet - )
echo "sources: all $(grep -vc '^#' "${here}/SOURCES.sha256") pins verified"

WRFDEFS="-DEM_CORE=1 -DNMM_CORE=0 -DCOAMPS_CORE=0 -DDA_CORE=0 -DEXP_CORE=0 \
-DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4 -Dwrfmodel -DGRIB1 \
-DINTIO -DLIMIT_ARGS -DCONFIG_BUF_LEN=65536 -DMAX_DOMAINS_F=21 -DMAX_HISTORY=25 \
-DNMM_NEST=0"
BASE="-cpp -ffree-form -ffree-line-length-none -fno-range-check -g ${WRFDEFS}"

# compile order = USE order
CAM=(
  phys/module_cam_shr_kind_mod.F
  phys/module_cam_shr_const_mod.F
  phys/module_cam_physconst.F
  phys/module_cam_support.F
  phys/module_cam_constituents.F
  phys/module_cam_gffgch.F
  phys/module_cam_wv_saturation.F
  phys/module_cam_esinti.F
  phys/module_cam_upper_bc.F
  phys/module_cam_bl_diffusion_solver.F
  phys/module_cam_molec_diff.F
  phys/module_cam_trb_mtn_stress.F
  phys/module_cam_bl_eddy_diff.F
  share/module_model_constants.F
  phys/module_bl_camuwpbl_driver.F
)

build_variant() {
    local name=$1 opt=$2 spydir=$3 extra=$4
    local dir="${out}/${name}"
    rm -rf "${dir}"
    mkdir -p "${dir}"
    (
        cd "${dir}"
        gfortran -c ${opt} ${BASE} ${extra} "${here}/stub_wrf.F90"
        objs=(stub_wrf.o)
        if [[ -n "${spydir}" ]]; then
            gfortran -c ${opt} ${BASE} ${extra} "${here}/uwspy.F90"
            objs+=(uwspy.o)
        fi
        for f in "${CAM[@]}"; do
            local b
            b=$(basename "${f}" .F)
            local s="${src}/${f}"
            if [[ -n "${spydir}" && -f "${spydir}/$(basename "${f}")" ]]; then
                s="${spydir}/$(basename "${f}")"
            fi
            gfortran -c ${opt} ${BASE} ${extra} -o "${b}.o" "${s}"
            objs+=("${b}.o")
        done
        gfortran -c ${opt} ${BASE} ${extra} "${here}/uwio.F90"
        local d=""
        [[ -n "${spydir}" ]] && d="-DUWSPY"
        gfortran -c ${opt} ${BASE} ${extra} ${d} "${here}/run_camuwpbl.F90"
        gfortran -o run_camuwpbl "${objs[@]}" uwio.o run_camuwpbl.o
        for o in "${objs[@]}"; do
            nm -u "${o}" | sed 's/^ *U *//' | sed "s|^|${o}: |"
        done | sort > undefined.txt
    )
}

build_variant pristine "-O0" "" ""
build_variant o2 "-O2 -ftree-vectorize -funroll-loops" "" ""
build_variant snan "-O0" "" "-finit-real=snan -finit-integer=-999999 -finit-derived"
python3 "${here}/make_spy.py" "${src}" "${out}/spy-src"
build_variant spy "-O0" "${out}/spy-src" ""

# --- libmvec guard ------------------------------------------------------------
if grep -q '_ZGV' "${out}/pristine/undefined.txt"; then
    echo "-O0 objects pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' "${out}/pristine/undefined.txt" >&2
    exit 4
fi
cat > "${out}/libmvec_control.f90" <<'EOF'
subroutine control(x, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  x = exp(x)
end subroutine control
EOF
( cd "${out}" && gfortran -c -Ofast -ftree-vectorize -mavx2 -o libmvec_control.o libmvec_control.f90 )
if ! nm -u "${out}/libmvec_control.o" | grep -q '_ZGV'; then
    echo "libmvec positive control produced no _ZGV symbol; the guard is vacuous" >&2
    exit 5
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# libm symbols the -O0 reference objects call:"
    grep -E ': (exp|log|pow|cos|acos|sqrt|log10|_ZGV)' "${out}/pristine/undefined.txt" || true
    echo "# the same, -O2 -ftree-vectorize -funroll-loops:"
    grep -E ': (exp|log|pow|cos|acos|sqrt|log10|_ZGV)' "${out}/o2/undefined.txt" || true
} > "${out}/libm-report.txt"
cat "${out}/libm-report.txt"

# --- run every case file through every variant ----------------------------------
status=0
mkdir -p "${out}/fixtures"
for c in "${cases}"/cases-*.bin; do
    stem=$(basename "${c}" .bin)
    for v in pristine o2 snan spy; do
        ( cd "${out}/${v}" && ./run_camuwpbl "${c}" "${out}/${v}/${stem}" )
    done
    for v in pristine o2 snan spy; do
        if [[ ! -s "${out}/${v}/${stem}.bin" || ! -s "${out}/${v}/${stem}.manifest" ]]; then
            echo "FAIL: ${v} wrote an empty fixture for ${stem}" >&2
            exit 7
        fi
    done
    if [[ ! -s "${out}/spy/${stem}-spy.bin" ]]; then
        echo "FAIL: the spy build recorded no stages for ${stem}" >&2
        exit 7
    fi
    cp "${out}/pristine/${stem}.bin" "${out}/pristine/${stem}.manifest" "${out}/fixtures/"
    for v in snan spy; do
        if ! cmp -s "${out}/pristine/${stem}.bin" "${out}/${v}/${stem}.bin"; then
            echo "FAIL: ${v} build differs from the reference on ${stem}" >&2
            status=6
        fi
    done
    if cmp -s "${out}/pristine/${stem}.bin" "${out}/o2/${stem}.bin"; then
        echo "${stem}: -O2 build byte-identical to the -O0 reference"
    else
        echo "${stem}: -O2 build DIFFERS from the -O0 reference (evidence, not a failure)"
    fi
    cp "${out}/spy/${stem}-spy.bin" "${out}/spy/${stem}-spy.manifest" "${out}/fixtures/"
done
cp "${cases}/cases-index.json" "${out}/fixtures/" 2>/dev/null || true

# --- stage programs: every run_stage_*.F90 beside this script is compiled
# against the pristine objects and run once with its output stem ---------------
shopt -s nullglob
for prog in "${here}"/run_stage_*.F90; do
    name=$(basename "${prog}" .F90)
    (
        cd "${out}/pristine"
        gfortran -c -O0 ${BASE} -o "${name}.o" "${prog}"
        objs=$(ls ./*.o | grep -v -e run_camuwpbl.o -e 'run_stage_')
        gfortran -o "${name}" ${objs} "${name}.o"
        ./"${name}" "${out}/fixtures/${name}"
    )
    if [[ ! -s "${out}/fixtures/${name}.bin" ]]; then
        echo "FAIL: ${name} wrote no fixture" >&2
        status=7
    fi
done
shopt -u nullglob
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
} > "${out}/fixtures/toolchain.txt"
( cd "${out}/fixtures" && sha256sum ./*.bin ./*.manifest > SHA256SUMS )
echo "oracle fixtures in ${out}/fixtures"
exit ${status}
