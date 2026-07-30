#!/usr/bin/env bash
# Build and run the YSU PBL oracle against the byte-unmodified WRF v4.6.1
# phys/physics_mmm/bl_ysu.F90.
#
# Why there is no stub_wrf.F90 here.  The RUC and Noah-MP harnesses need one
# because their modules USE module_wrf_error and module_model_constants and
# CALL wrf_message / wrf_dm_bcast_*.  bl_ysu.F90 does none of that: its only
# USE is ccpp_kind_types and every CALL it makes (tridin_ysu, tridi2n,
# get_pblh) is to a procedure in the same module.  `nm -u module_bl_ysu.o`
# below is the receipt for that claim -- it must list nothing but libm and
# libgfortran.  ccpp_kind_types.F is therefore compiled from the pinned tree
# too, with WRF's own default -DRWORDSIZE=4, rather than replaced by a stub:
# a stub would be free to get kind_phys wrong, and kind_phys IS the precision
# the whole reference is measured in.
#
# libmvec.  WRF's own configure.wrf compiles phys/ at -O2 with
# -ftree-vectorize, and at that setting gfortran's auto-vectoriser replaces
# scalar expf/powf in the k-loops with the glibc vector forms
# (_ZGVbN4v_expf, _ZGVbN4vv_powf, ...), whose accuracy contract is 4 ULP
# rather than the scalar routines' ~0.5.  That silently changes the reference
# a bit-parity gate is measured against.  The oracle is therefore built at
# -O0 and this script FAILS if any _ZGV* symbol appears in the -O0 object.
# It also builds a second, throw-away -O2 -ftree-vectorize object purely to
# demonstrate that the swap is real on this toolchain and record which
# symbols appear.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
kind_source="${source_root}/phys/ccpp_kind_types.F"
ysu_source="${source_root}/phys/physics_mmm/bl_ysu.F90"
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"

test -f "${kind_source}"
test -f "${ysu_source}"

# --- byte identity to the pinned commit -------------------------------------
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- \
        phys/ccpp_kind_types.F phys/physics_mmm/bl_ysu.F90; then
    echo "bl_ysu.F90 or ccpp_kind_types.F differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

# --- the reference build: -O0, no vectoriser --------------------------------
gfortran -c -O0 -cpp -DRWORDSIZE=4 -ffree-form -ffree-line-length-none \
    "${kind_source}"
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none \
    -I "${build_dir}" -o bl_ysu_O0.o "${ysu_source}"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_bl_ysu.F90"
gfortran -o run_bl_ysu ccpp_kind_types.o bl_ysu_O0.o run_bl_ysu.o

# --- receipts ---------------------------------------------------------------
nm -u bl_ysu_O0.o | sed 's/^ *//' | sort > undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-O0.txt >&2
    exit 4
fi
if grep -Eqv '^(U )?(_gfortran_|__gfortran_|GCC_|expf|powf|sqrtf|__powf_finite|__expf_finite|logf|cbrtf|atanf|_ZGV|__stack_chk_fail|memset|memcpy)' \
        <(sed 's/^U *//' undefined-O0.txt); then
    echo "note: undefined symbols beyond libm/libgfortran are listed in" \
         "undefined-O0.txt; review before trusting the no-stub claim" >&2
fi

# The vectorised builds exist only as evidence; they are never linked.
gfortran -c -O2 -ftree-vectorize -cpp -ffree-form -ffree-line-length-none \
    -I "${build_dir}" -o bl_ysu_O2vec.o "${ysu_source}"
nm -u bl_ysu_O2vec.o | sed 's/^ *//' | sort > undefined-O2vec.txt
gfortran -c -Ofast -ftree-vectorize -cpp -ffree-form -ffree-line-length-none \
    -I "${build_dir}" -o bl_ysu_Ofast.o "${ysu_source}"
nm -u bl_ysu_Ofast.o | sed 's/^ *//' | sort > undefined-Ofast.txt

# Positive control: the grep must be able to find _ZGV* on this toolchain at
# all, or its silence on bl_ysu.o means nothing.
gfortran -c -Ofast -ftree-vectorize \
    -o libmvec_positive_control.o "${script_dir}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol; the guard on" \
         "bl_ysu.o is vacuous on this toolchain" >&2
    cat undefined-control.txt >&2
    exit 5
fi

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 libm symbols (THE REFERENCE):"
    grep -E 'expf|powf|logf|sqrtf|cbrtf|_ZGV' undefined-O0.txt || true
    echo "# -O2 -ftree-vectorize (WRF's own phys/ setting):"
    grep -E 'expf|powf|logf|sqrtf|cbrtf|_ZGV' undefined-O2vec.txt || true
    echo "# -Ofast -ftree-vectorize:"
    grep -E 'expf|powf|logf|sqrtf|cbrtf|_ZGV' undefined-Ofast.txt || true
    echo "# positive control (plain expf loop at -Ofast), proves the grep works:"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt
cat libmvec-report.txt

# --- run --------------------------------------------------------------------
./run_bl_ysu ysu-levels.csv ysu-surface.csv

sha256sum "${kind_source}" "${ysu_source}" \
    "${script_dir}/run_bl_ysu.F90" \
    ysu-levels.csv ysu-surface.csv > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built in ${build_dir}"
