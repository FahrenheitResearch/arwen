#!/usr/bin/env bash
# Build and run the RUC soil-ingest oracle against byte-unmodified WRF v4.6.1
# share/module_soil_pre.F.
#
# What is real and what is stubbed
# --------------------------------
# REAL, from the pinned tree, unmodified:
#   share/module_soil_pre.F          init_soil_depth_3, init_soil_3_real and
#                                    the external skip_middle_points_t, all of
#                                    which live in this one file
#   frame/module_state_description.F the registry-generated scheme integers.
#                                    NOT stubbed: process_soil_real branches on
#                                    RUCLSMSCHEME/LSMSCHEME/... and a stub would
#                                    be free to get one wrong.
# STUBBED, in stub_wrf.F90, to exactly the seven symbols `nm -u` reports on a
# full build's share/module_soil_pre.o:
#   module_date_time's current_date/start_date (two CHARACTERs, read only by a
#   diagnostic write at :1547 that the soil-depth path never reaches), plus
#   wrf_message / wrf_debug / wrf_error_fatal3 / nl_get_mminlu /
#   nl_get_aggregate_lu.
#
# libmvec
# -------
# A full WRF build's share/module_soil_pre.o DOES carry _ZGVbN4v_expf: WRF
# compiles share/ at -O2 -ftree-vectorize and gfortran swaps the scalar expf in
# adjust_soil_temp_new for glibc's 4-ULP vector form.  init_soil_3_real itself
# contains no transcendental at all -- only +, -, *, / and MAX -- so the swap
# cannot move THIS reference.  The oracle is still built at -O0 and this script
# still fails on a _ZGV* symbol, and it still builds the positive control,
# because a guard that has never been observed to fire proves nothing.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
soil_source="${source_root}/share/module_soil_pre.F"
msd_source="${source_root}/frame/module_state_description.F"
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"

test -f "${soil_source}"
test -f "${msd_source}"

# --- byte identity to the pinned commit -------------------------------------
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- share/module_soil_pre.F; then
    echo "share/module_soil_pre.F differs from the pinned commit" >&2
    exit 3
fi
# module_state_description.F is registry-generated and therefore untracked;
# checking it against git would always fail.  Its checksum is recorded below.

mkdir -p "${build_dir}"
cd "${build_dir}"

# --- WRF's own .F -> .f90 pipeline ------------------------------------------
# configure.wrf:365-370.  The step that matters is tools/standard.exe
# (arch/standard.sed:8), which rewrites `CALL wrf_error_fatal (` into
# `CALL wrf_error_fatal3 ( __FILE__ , __LINE__ ,` -- which is why a full
# build's object references wrf_error_fatal3_ and a naive `gfortran -cpp`
# does not.  Running WRF's pipeline rather than hand-patching the source is
# what keeps share/module_soil_pre.F byte-unmodified.
preprocess_wrf_F () {
    local src="$1" out="$2"
    shift 2
    sed -e "s/^\!.*'.*//" -e "s/^ *\!.*'.*//" "${src}" > "${out}.G"
    /lib/cpp -P -nostdinc -traditional-cpp "$@" "${out}.G" > "${out}.bb"
    "${source_root}/tools/standard.exe" "${out}.bb" \
        | /lib/cpp -P -nostdinc -traditional-cpp > "${out}"
    rm -f "${out}.G" "${out}.bb"
}

preprocess_wrf_F "${soil_source}" module_soil_pre.f90 -DWRF_CHEM=0
preprocess_wrf_F "${msd_source}" module_state_description.f90

# Receipt that the pipeline above is WRF's: the full build left its own
# generated .f90 beside the source, and ours must be identical to it.
if [[ -f "${source_root}/share/module_soil_pre.f90" ]]; then
    if ! diff -q "${source_root}/share/module_soil_pre.f90" \
            module_soil_pre.f90 > /dev/null; then
        echo "generated module_soil_pre.f90 differs from the one WRF's own" \
             "build produced; the preprocessing pipeline is not WRF's" >&2
        diff "${source_root}/share/module_soil_pre.f90" module_soil_pre.f90 \
            | head -40 >&2
        exit 6
    fi
    echo "receipt: generated module_soil_pre.f90 is identical to WRF's own"
else
    echo "note: no WRF-built module_soil_pre.f90 to compare the pipeline to" >&2
fi

# --- the reference build: -O0, no vectoriser --------------------------------
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none \
    "${script_dir}/stub_wrf.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" module_state_description.f90
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" -o module_soil_pre_O0.o module_soil_pre.f90
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_init_soil_3.F90"
gfortran -o run_init_soil_3 stub_wrf.o module_state_description.o \
    module_soil_pre_O0.o run_init_soil_3.o

# --- receipts ---------------------------------------------------------------
nm -u module_soil_pre_O0.o | sed 's/^ *//' | sort > undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-O0.txt >&2
    exit 4
fi

# WRF's own share/ setting, built only as evidence and never linked.
gfortran -c -O2 -ftree-vectorize -ffree-form \
    -ffree-line-length-none -I "${build_dir}" \
    -o module_soil_pre_O2vec.o module_soil_pre.f90
nm -u module_soil_pre_O2vec.o | sed 's/^ *//' | sort > undefined-O2vec.txt

# Positive control: the grep must be able to find _ZGV* on this toolchain, or
# its silence on the -O0 object means nothing.
gfortran -c -Ofast -ftree-vectorize \
    -o libmvec_positive_control.o "${script_dir}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol; the guard on" \
         "module_soil_pre_O0.o is vacuous on this toolchain" >&2
    cat undefined-control.txt >&2
    exit 5
fi

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 (THE REFERENCE), transcendental/vector symbols:"
    grep -E 'expf|powf|logf|sqrtf|_ZGV' undefined-O0.txt || echo "  (none)"
    echo "# -O2 -ftree-vectorize (WRF's own share/ setting):"
    grep -E 'expf|powf|logf|sqrtf|_ZGV' undefined-O2vec.txt || echo "  (none)"
    echo "# positive control (plain expf loop at -Ofast), proves the grep works:"
    grep -E 'expf|_ZGV' undefined-control.txt || echo "  (none)"
} > libmvec-report.txt
cat libmvec-report.txt

# --- run --------------------------------------------------------------------
./run_init_soil_3 ruc-soil-ingest.csv 2> run.log

sha256sum "${soil_source}" "${msd_source}" \
    "${script_dir}/run_init_soil_3.F90" "${script_dir}/stub_wrf.F90" \
    ruc-soil-ingest.csv > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built in ${build_dir}"
