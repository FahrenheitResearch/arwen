#!/usr/bin/env bash
# Build the WSM6 (mp_physics = 6) column oracle against the byte-unmodified
# WRF v4.6.1 sources and run it over the column set make_columns.py writes.
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# FILE SET.  WRF's own entry point phys/module_mp_wsm6.F (the wrapper
# module_microphysics_driver.F calls), the scheme phys/physics_mmm/
# mp_wsm6.F90 and mp_wsm6_effectRad.F90, their dependencies
# phys/physics_mmm/module_libmassv.F90 and mp_radar.F90, the kind module
# phys/ccpp_kind_types.F and the physical constants
# share/module_model_constants.F.  No stub: `nm -u` on the objects is the
# receipt (libm and libgfortran only).
#
# -DRWORDSIZE=4 is not optional: ccpp_kind_types.F takes the DOUBLE branch
# when the macro is undefined (cpp evaluates an undefined identifier as 0).
# run_wsm6_columns refuses to run if kind_phys is not real(4).
#
# -O0 and libmvec.  WRF compiles phys/ at -O2 -ftree-vectorize, where the
# vectoriser may swap scalar expf/logf/powf for glibc's 4-ULP vector forms.
# The reference is built at -O0 and this script FAILS if any _ZGV* symbol
# appears in an -O0 object.  A positive control proves the grep can see the
# symbol on this toolchain.  The stock -O2 -ftree-vectorize -funroll-loops
# build is ALSO run, as evidence of what WRF's own flags do to these words.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
files=(phys/ccpp_kind_types.F share/module_model_constants.F
       phys/physics_mmm/module_libmassv.F90 phys/physics_mmm/mp_radar.F90
       phys/physics_mmm/mp_wsm6.F90 phys/physics_mmm/mp_wsm6_effectRad.F90
       phys/module_mp_wsm6.F)

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${files[@]}"; then
    echo "a WSM6 oracle source differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

fc_flags=(-cpp -DRWORDSIZE=4 -ffree-form -ffree-line-length-none)
compile_set() {   # $1 = object dir, rest = optimisation flags
    local dir=$1; shift
    mkdir -p "${dir}"
    for f in "${files[@]}"; do
        gfortran -c "$@" "${fc_flags[@]}" -J "${dir}" -I "${dir}" \
            -o "${dir}/$(basename "${f%.*}").o" "${source_root}/${f}"
    done
    gfortran -c "$@" "${fc_flags[@]}" -J "${dir}" -I "${dir}" \
        -o "${dir}/run_wsm6_columns.o" "${script_dir}/run_wsm6_columns.F90"
    gfortran -o "${dir}/run_wsm6_columns" \
        "${dir}"/ccpp_kind_types.o "${dir}"/module_model_constants.o \
        "${dir}"/module_libmassv.o "${dir}"/mp_radar.o "${dir}"/mp_wsm6.o \
        "${dir}"/mp_wsm6_effectRad.o "${dir}"/module_mp_wsm6.o \
        "${dir}"/run_wsm6_columns.o
}

# --- the reference: -O0 ------------------------------------------------------
compile_set O0 -O0
for o in O0/mp_wsm6.o O0/mp_wsm6_effectRad.o O0/module_mp_wsm6.o O0/mp_radar.o; do
    nm -u "$o" | sed 's/^ *//'
done | sort -u > undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-O0.txt >&2
    exit 4
fi

# --- WRF's own phys/ flags, evidence only ---------------------------------------
compile_set O2vec -O2 -ftree-vectorize -funroll-loops
for o in O2vec/mp_wsm6.o O2vec/mp_wsm6_effectRad.o O2vec/module_mp_wsm6.o O2vec/mp_radar.o; do
    nm -u "$o" | sed 's/^ *//'
done | sort -u > undefined-O2vec.txt

gfortran -c -Ofast -ftree-vectorize -o libmvec_positive_control.o \
    "${script_dir}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol; the -O0 guard is vacuous" >&2
    exit 5
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 (THE REFERENCE) libm/libmvec symbols:"
    grep -E 'exp|log|pow|sqrt|_ZGV' undefined-O0.txt || true
    echo "# -O2 -ftree-vectorize -funroll-loops (WRF's phys/ flags):"
    grep -E 'exp|log|pow|sqrt|_ZGV' undefined-O2vec.txt || true
    echo "# positive control:"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt
cat libmvec-report.txt

# --- the init-constant dump (private -> public copy; one line differs) -------
mkdir -p dump
sed 's/^ private$/ public/' "${source_root}/phys/physics_mmm/mp_wsm6.F90" > dump/mp_wsm6.F90
ndiff=$(diff "${source_root}/phys/physics_mmm/mp_wsm6.F90" dump/mp_wsm6.F90 | grep -c '^[<>]' || true)
if [[ "${ndiff}" != "2" ]]; then
    echo "the dump copy of mp_wsm6.F90 differs by more than its accessibility line" >&2
    exit 6
fi
for f in phys/ccpp_kind_types.F share/module_model_constants.F \
         phys/physics_mmm/module_libmassv.F90 phys/physics_mmm/mp_radar.F90; do
    gfortran -c -O0 "${fc_flags[@]}" -J dump -I dump \
        -o "dump/$(basename "${f%.*}").o" "${source_root}/${f}"
done
gfortran -c -O0 "${fc_flags[@]}" -J dump -I dump -o dump/mp_wsm6.o dump/mp_wsm6.F90
gfortran -c -O0 "${fc_flags[@]}" -J dump -I dump -o dump/dump_wsm6_init.o \
    "${script_dir}/dump_wsm6_init.F90"
gfortran -o dump/dump_wsm6_init dump/ccpp_kind_types.o dump/module_model_constants.o \
    dump/module_libmassv.o dump/mp_radar.o dump/mp_wsm6.o dump/dump_wsm6_init.o
./dump/dump_wsm6_init > wsm6-init-f32.txt
cat wsm6-init-f32.txt

# --- run ----------------------------------------------------------------------
python3 "${script_dir}/make_columns.py" "${build_dir}"
# (hail_opt, dt, nsteps): dt 30 is one minor loop; 150 is one loop with
# dtcld > dtcldcr; 300 is nint(2.5) = 3 loops of 100 s.
runs=("0 30 12" "1 30 12" "0 150 3" "1 150 3" "0 300 4" "1 300 4")
for r in "${runs[@]}"; do
    set -- $r
    tag="h$1-dt$2-n$3"
    ./O0/run_wsm6_columns columns.bin "wrf-O0-${tag}.bin" "$1" "$2" "$3"
    ./O2vec/run_wsm6_columns columns.bin "wrf-O2vec-${tag}.bin" "$1" "$2" "$3"
done
sha256sum "${source_root}"/phys/physics_mmm/mp_wsm6.F90 \
    "${source_root}"/phys/physics_mmm/mp_wsm6_effectRad.F90 \
    "${source_root}"/phys/module_mp_wsm6.F "${source_root}"/phys/physics_mmm/mp_radar.F90 \
    "${source_root}"/phys/physics_mmm/module_libmassv.F90 \
    "${source_root}"/share/module_model_constants.F "${source_root}"/phys/ccpp_kind_types.F \
    "${script_dir}"/run_wsm6_columns.F90 "${script_dir}"/make_columns.py \
    columns.bin wrf-O0-*.bin > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built and run in ${build_dir}"
