#!/usr/bin/env bash
# Build and run the WRF v4.7.1 ndown oracle (rebalance and blend_terrain).
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR CASES_DIR
#
# WRF_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh writes such a tree); every one is checked
# against its pin before anything compiles.  The two routines are cut out of their files by line range, and the cut
# is checked to begin with the routine's SUBROUTINE line and end with its END SUBROUTINE line, so the oracle compiles
# WRF's own text, byte for byte.  CASES_DIR holds the case files make_cases.py wrote; every one is run and its
# output written beside it as <case>.out.
#
# Two builds, each in its own directory:
#   pristine  the reference: gfortran -O0, WRF's preprocessor defines for an EM, RWORDSIZE=4 build.  Its outputs are
#             the fixtures of record.
#   snan      the reference with every local real initialised to a signalling NaN and every local integer to
#             -999999.  The script requires its outputs to equal the reference's byte for byte: no uninitialised
#             local reaches an output on the cases run.
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

cut_routine() {   # FILE FIRST LAST NAME OUT
    local file=$1 first=$2 last=$3 name=$4 dest=$5
    sed -n "${first},${last}p" "${file}" > "${dest}"
    head -1 "${dest}" | grep -Eq "^ *SUBROUTINE ${name} *\(" \
        || { echo "line ${first} of ${file} is not SUBROUTINE ${name}" >&2; exit 3; }
    tail -1 "${dest}" | grep -Eq "^ *END SUBROUTINE ${name} *$" \
        || { echo "line ${last} of ${file} is not END SUBROUTINE ${name}" >&2; exit 3; }
}

WRFDEFS="-DEM_CORE=1 -DNMM_CORE=0 -DCOAMPS_CORE=0 -DDA_CORE=0 -DEXP_CORE=0 \
-DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4 -Dwrfmodel -DGRIB1 \
-DINTIO -DLIMIT_ARGS -DCONFIG_BUF_LEN=65536 -DMAX_DOMAINS_F=21 -DMAX_HISTORY=25 \
-DNMM_NEST=0"
BASE="-cpp -ffree-form -ffree-line-length-none -fno-range-check -g ${WRFDEFS}"

build_variant() {
    local name=$1 extra=$2
    local dir="${out}/${name}"
    rm -rf "${dir}"
    mkdir -p "${dir}"
    (
        cd "${dir}"
        cut_routine "${src}/dyn_em/module_initialize_real.F" 4982 5266 rebalance rebalance_wrf471.inc
        cut_routine "${src}/dyn_em/nest_init_utils.F" 712 785 blend_terrain blend_terrain_wrf471.inc
        cp "${here}/dummy_new_args.inc" "${here}/dummy_new_decl.inc" .
        gfortran -c -O0 ${BASE} ${extra} "${here}/stub_wrf.F90"
        gfortran -c -O0 ${BASE} ${extra} "${src}/share/module_model_constants.F"
        gfortran -c -O0 ${BASE} ${extra} -I. "${here}/rebalance_host.F90"
        gfortran -c -O0 ${BASE} ${extra} -I. "${here}/blend_host.F90"
        gfortran -c -O0 ${BASE} ${extra} "${here}/run_ndown.F90"
        gfortran -O0 -o run_ndown run_ndown.o rebalance_host.o blend_host.o module_model_constants.o stub_wrf.o
        # libmvec guard: an -O0 object must call glibc's scalar functions, never a _ZGV vector variant.
        if nm ./*.o | grep -q '_ZGV'; then
            echo "${name}: an object references a libmvec vector symbol" >&2
            exit 4
        fi
    )
}

build_variant pristine ""
build_variant snan "-finit-real=snan -finit-integer=-999999"

n=0
for case_file in "${cases}"/*.case; do
    stem=$(basename "${case_file}" .case)
    mode=${stem%%-*}
    "${out}/pristine/run_ndown" "${mode}" "${case_file}" "${cases}/${stem}.out"
    "${out}/snan/run_ndown" "${mode}" "${case_file}" "${out}/snan/${stem}.out"
    cmp -s "${cases}/${stem}.out" "${out}/snan/${stem}.out" \
        || { echo "${stem}: the snan build differs from the reference" >&2; exit 5; }
    n=$((n + 1))
done
echo "oracle: ${n} cases run; the snan build is byte-identical to the reference on every one"
gfortran --version | head -1
ldd --version | head -1
