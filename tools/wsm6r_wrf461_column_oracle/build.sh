#!/usr/bin/env bash
# Build the WSM6R (mp_physics = 106) column oracle against the byte-unmodified
# WRF v4.6.1 source and run it over the column set make_columns.py writes.
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# FILE SET.  phys/module_mp_wsm6r.F, the module the microphysics driver
# USEs for mp_physics = 106 (module_microphysics_driver.F:219, :2356-2376);
# it has no USE statement of its own and takes no constant from the driver.
# Beside it, WSM6's effective-radius routine (physics_mmm/
# mp_wsm6_effectRad.F90 with mp_wsm6.F90 and their dependencies), which
# grades gpuwm's radius diagnosis on the WSM6R state.  No stub: `nm -u` on
# the objects is the receipt (libm and libgfortran only).
#
# -O0 and libmvec.  WRF compiles phys/ at -O2 -ftree-vectorize, where the
# vectoriser may swap scalar expf/logf/powf for glibc's vector forms.  The
# reference is built at -O0 and this script FAILS if any _ZGV* symbol appears
# in an -O0 object.  A positive control proves the grep can see the symbol on
# this toolchain.  The stock -O2 -ftree-vectorize -funroll-loops build is
# ALSO run, as evidence of what WRF's own flags do to these words.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
scheme=phys/module_mp_wsm6r.F
# WSM6R itself, then the WSM6 effective-radius routine gpuwm's radiation
# diagnosis is graded against and its dependencies (the WSM6 oracle's set).
files=(phys/ccpp_kind_types.F share/module_model_constants.F
       phys/physics_mmm/module_libmassv.F90 phys/physics_mmm/mp_radar.F90
       phys/physics_mmm/mp_wsm6.F90 phys/physics_mmm/mp_wsm6_effectRad.F90
       "${scheme}")

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${files[@]}"; then
    echo "an oracle source differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

# -DRWORDSIZE=4 is not optional: ccpp_kind_types.F takes the DOUBLE branch
# when the macro is undefined.  module_mp_wsm6r.F carries no directive and
# uses default REAL, so the flag does not touch it.
fc_flags=(-cpp -DRWORDSIZE=4 -ffree-form -ffree-line-length-none)
objects() { local dir=$1; for f in "${files[@]}"; do
    printf '%s ' "${dir}/$(basename "${f%.*}").o"; done; }
compile_set() {   # $1 = object dir, rest = optimisation flags
    local dir=$1; shift
    mkdir -p "${dir}"
    for f in "${files[@]}"; do
        gfortran -c "$@" "${fc_flags[@]}" -J "${dir}" -I "${dir}" \
            -o "${dir}/$(basename "${f%.*}").o" "${source_root}/${f}"
    done
    gfortran -c "$@" "${fc_flags[@]}" -J "${dir}" -I "${dir}" \
        -o "${dir}/run_wsm6r_columns.o" "${script_dir}/run_wsm6r_columns.F90"
    # shellcheck disable=SC2046
    gfortran -o "${dir}/run_wsm6r_columns" $(objects "${dir}") \
        "${dir}/run_wsm6r_columns.o"
}

# --- the reference: -O0 ------------------------------------------------------
compile_set O0 -O0
for o in O0/module_mp_wsm6r.o O0/mp_wsm6.o O0/mp_wsm6_effectRad.o; do
    nm -u "$o" | sed 's/^ *//'; done | sort -u > undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-O0.txt >&2
    exit 4
fi

# --- WRF's own phys/ flags, evidence only ---------------------------------------
compile_set O2vec -O2 -ftree-vectorize -funroll-loops
for o in O2vec/module_mp_wsm6r.o O2vec/mp_wsm6.o O2vec/mp_wsm6_effectRad.o; do
    nm -u "$o" | sed 's/^ *//'; done | sort -u > undefined-O2vec.txt

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
    grep -E 'exp|log|pow|sqrt|round|_ZGV' undefined-O0.txt || true
    echo "# -O2 -ftree-vectorize -funroll-loops (WRF's phys/ flags):"
    grep -E 'exp|log|pow|sqrt|round|_ZGV' undefined-O2vec.txt || true
    echo "# positive control:"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt
cat libmvec-report.txt

# --- the init-constant dump (the module's SAVE words are public) -------------
mkdir -p dump
for f in "${files[@]}"; do
    gfortran -c -O0 "${fc_flags[@]}" -J dump -I dump \
        -o "dump/$(basename "${f%.*}").o" "${source_root}/${f}"
done
gfortran -c -O0 "${fc_flags[@]}" -J dump -I dump -o dump/dump_wsm6r_init.o \
    "${script_dir}/dump_wsm6r_init.F90"
# shellcheck disable=SC2046
gfortran -o dump/dump_wsm6r_init $(objects dump) dump/dump_wsm6r_init.o
./dump/dump_wsm6r_init > wsm6r-init-f32.txt
cat wsm6r-init-f32.txt

# --- run ----------------------------------------------------------------------
python3 "${script_dir}/make_columns.py" "${build_dir}" wsm6r-init-f32.txt
# (dt, nsteps): loops = max(nint(dt/120), 1) minor loops of dt/loops.
#   30 s  : one loop, short step, 12 free-running steps
#   60 s  : one loop, 6 steps
#   150 s : one loop with dtcld > dtcldcr (nint(1.25) = 1)
#   240 s : two loops of 120 s
#   300 s : three loops of 100 s (nint(2.5) = 3, half away from zero)
runs=("30 12" "60 6" "150 3" "240 3" "300 4")
for r in "${runs[@]}"; do
    set -- $r
    tag="dt$1-n$2"
    ./O0/run_wsm6r_columns columns.bin "wrf-O0-${tag}.bin" "$1" "$2"
    ./O2vec/run_wsm6r_columns columns.bin "wrf-O2vec-${tag}.bin" "$1" "$2"
done
sha256sum "${files[@]/#/${source_root}/}" \
    "${script_dir}"/run_wsm6r_columns.F90 "${script_dir}"/make_columns.py \
    "${script_dir}"/dump_wsm6r_init.F90 \
    columns.bin wrf-O0-*.bin > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built and run in ${build_dir}"
