#!/usr/bin/env bash
# Build the WSM3 (mp_physics = 3) column oracle against the byte-unmodified
# WRF v4.6.1 sources and run it over the column set make_columns.py writes.
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# FILE SET.  WRF's own scheme and entry point phys/module_mp_wsm3.F (the
# module module_microphysics_driver.F calls), its vector helpers
# frame/libmassv.F (vsrec/vssqrt, which the scheme's VREC/VSQRT macros name
# at RWORDSIZE = 4) and the physical constants share/module_model_constants.F.
# No stub: `nm -u` on the objects is the receipt (libm and libgfortran only).
#
# -DRWORDSIZE=4 is not optional: module_mp_wsm3.F picks vsrec/vssqrt (the
# float32 helpers) only when RWORDSIZE == 4; with the macro undefined cpp
# takes the vrec/vsqrt (float64) branch.
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
files=(share/module_model_constants.F frame/libmassv.F phys/module_mp_wsm3.F)

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${files[@]}"; then
    echo "a WSM3 oracle source differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

fc_flags=(-cpp -DRWORDSIZE=4 -DEM_CORE=1 -ffree-form -ffree-line-length-none)
compile_set() {   # $1 = object dir, rest = optimisation flags
    local dir=$1; shift
    mkdir -p "${dir}"
    for f in "${files[@]}"; do
        gfortran -c "$@" "${fc_flags[@]}" -J "${dir}" -I "${dir}" \
            -o "${dir}/$(basename "${f%.*}").o" "${source_root}/${f}"
    done
    for prog in run_wsm3_columns dump_wsm3_init; do
        gfortran -c "$@" "${fc_flags[@]}" -J "${dir}" -I "${dir}" \
            -o "${dir}/${prog}.o" "${script_dir}/${prog}.F90"
        gfortran -o "${dir}/${prog}" \
            "${dir}"/module_model_constants.o "${dir}"/libmassv.o \
            "${dir}"/module_mp_wsm3.o "${dir}/${prog}.o"
    done
}

# --- the reference: -O0 ------------------------------------------------------
compile_set O0 -O0
for o in O0/module_mp_wsm3.o O0/libmassv.o; do
    nm -u "$o" | sed 's/^ *//'
done | sort -u > undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-O0.txt >&2
    exit 4
fi

# --- WRF's own phys/ flags, evidence only ---------------------------------------
compile_set O2vec -O2 -ftree-vectorize -funroll-loops
for o in O2vec/module_mp_wsm3.o O2vec/libmassv.o; do
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

# --- the init-constant dump (unmodified module; its SAVE words are public) ---
./O0/dump_wsm3_init > wsm3-init-f32.txt
cat wsm3-init-f32.txt

# --- the declared-divergence reference: O0 plus the D1 fix, nothing else ----
# WRF adds the cloud-ice fallout fallc(i,1) to the surface only when the
# lowest level is below 0 C (module_mp_wsm3.F:604-607), but the
# semi-Lagrangian CFL limiter lets ice fall out of a warm lowest level, and
# that ice is then counted nowhere.  gpuwm counts it as rain (kernels/
# wsm3.cu D1).  This copy moves the one fallsum line out of the IF; the
# script refuses any other difference.
mkdir -p fix
python3 - "${source_root}/phys/module_mp_wsm3.F" fix/module_mp_wsm3.F <<'PY'
import sys
src = open(sys.argv[1]).read()
old = ("        if((t0c-t(i,1)).gt.0) then\n"
       "        fallsum = fallsum+fallc(i,1)\n"
       "        fallsum_qsi = fall(i,1)+fallc(i,1)\n")
new = ("        fallsum = fallsum+fallc(i,1)\n"
       "        if((t0c-t(i,1)).gt.0) then\n"
       "        fallsum_qsi = fall(i,1)+fallc(i,1)\n")
if src.count(old) != 1:
    sys.exit("D1 site not found exactly once")
open(sys.argv[2], "w").write(src.replace(old, new))
PY
diff "${source_root}/phys/module_mp_wsm3.F" fix/module_mp_wsm3.F > d1-fix.diff || true
ndiff=$(grep -c '^[<>]' d1-fix.diff || true)
if [[ "${ndiff}" != "2" ]]; then
    echo "the D1 copy of module_mp_wsm3.F differs by more than its one moved line" >&2
    exit 6
fi
mkdir -p O0fix
cp O0/module_model_constants.o O0/libmassv.o O0/module_model_constants.mod O0fix/
gfortran -c -O0 "${fc_flags[@]}" -J O0fix -I O0fix -o O0fix/module_mp_wsm3.o \
    fix/module_mp_wsm3.F
gfortran -c -O0 "${fc_flags[@]}" -J O0fix -I O0fix -o O0fix/run_wsm3_columns.o \
    "${script_dir}/run_wsm3_columns.F90"
gfortran -o O0fix/run_wsm3_columns O0fix/module_model_constants.o O0fix/libmassv.o \
    O0fix/module_mp_wsm3.o O0fix/run_wsm3_columns.o

# --- run ----------------------------------------------------------------------
python3 "${script_dir}/make_columns.py" "${build_dir}"
# (dt, nsteps): dt 30 and 72 are one minor loop; 150 is one loop with
# dtcld > dtcldcr; 180 is nint(1.5) = 2 loops of 90 s; 300 is nint(2.5) =
# 3 loops of 100 s.
runs=("30 12" "72 8" "150 3" "180 3" "300 4")
for r in "${runs[@]}"; do
    set -- $r
    tag="dt$1-n$2"
    ./O0/run_wsm3_columns columns.bin "wrf-O0-${tag}.bin" "$1" "$2"
    ./O0fix/run_wsm3_columns columns.bin "wrf-O0fix-${tag}.bin" "$1" "$2"
    ./O2vec/run_wsm3_columns columns.bin "wrf-O2vec-${tag}.bin" "$1" "$2"
done
sha256sum "${source_root}"/phys/module_mp_wsm3.F "${source_root}"/frame/libmassv.F \
    "${source_root}"/share/module_model_constants.F \
    "${script_dir}"/run_wsm3_columns.F90 "${script_dir}"/dump_wsm3_init.F90 \
    "${script_dir}"/make_columns.py \
    fix/module_mp_wsm3.F columns.bin wrf-O0-*.bin wrf-O0fix-*.bin > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built and run in ${build_dir}"
