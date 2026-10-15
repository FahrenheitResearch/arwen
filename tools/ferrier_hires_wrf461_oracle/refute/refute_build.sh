#!/usr/bin/env bash
# Adversarial rerun of the WRF v4.6.1 Ferrier-Aligo column oracle on the
# refute column set (make_columns_refute.py).  CPU only.
#
# usage: refute_build.sh WRF_SOURCE_ROOT BUILD_DIR [LANE_COLUMNS_IN]
#
# Builds the byte-unmodified module four ways and runs each on the refute
# columns:
#   O0        the reference (-O0, scalar libm), same flags as build.sh
#   O2vec     WRF's own GNU flags (-O2 -ftree-vectorize -funroll-loops)
#   O3native  -O3 -march=native (FMA contraction allowed): evidence of which
#             fields are arithmetic-order sensitive, NOT a WRF configuration
#   cov       -O0 --coverage; run on the lane columns (if given) and then on
#             the refute columns, gcov line coverage written for each
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: $0 WRF_SOURCE_ROOT BUILD_DIR [LANE_COLUMNS_IN]" >&2; exit 2; }
src=$(realpath "$1"); out=$(realpath -m "$2"); lane_in=${3:-}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); h=$(dirname "$here")
fer="$src/phys/module_mp_fer_hires.F"; table="$src/run/ETAMPNEW_DATA.expanded_rain"
[[ $(git -C "$src" rev-parse HEAD) == d66e442fccc04111067e29274c9f9eaccc3cef28 ]]
git -C "$src" diff --quiet HEAD -- phys/module_mp_fer_hires.F run/ETAMPNEW_DATA.expanded_rain
mkdir -p "$out"; cd "$out"
python3 "$here/make_columns_refute.py" "$out"
common=(-cpp -ffree-form -ffree-line-length-none -DRWORDSIZE=4 -DIWORDSIZE=4 -fallow-argument-mismatch)
build() {
  local tag=$1; shift
  mkdir -p "$tag"; ( cd "$tag"
    cp "$fer" module_mp_fer_hires.F
    gfortran -c "$@" "${common[@]}" "$h/stub_wrf.F90" 2>/dev/null
    gfortran -c "$@" "${common[@]}" module_mp_fer_hires.F 2>/dev/null
    gfortran -c "$@" "${common[@]}" -fconvert=big-endian "$h/run_ferrier.F90"
    gfortran "$@" -o run_ferrier stub_wrf.o module_mp_fer_hires.o run_ferrier.o
    nm -u module_mp_fer_hires.o | sed 's/^ *//' | sort > undefined-symbols.txt
    ln -sf "$table" ETAMPNEW_DATA.expanded_rain )
}
run() {  # run TAG COLUMNS_IN
  ( cd "$1"; ln -sf "$2" columns.in; rc=0; ./run_ferrier > run.stdout 2> run.stderr || rc=$?; echo "exit $rc $1" ) }
build O0 -O0
grep -q _ZGV O0/undefined-symbols.txt && { echo "O0 pulled libmvec" >&2; exit 4; }
build O2vec -O2 -ftree-vectorize -funroll-loops
build O3native -O3 -march=native -funroll-loops
build cov -O0 --coverage
for t in O0 O2vec O3native; do run "$t" "$out/columns.in"; done
if [[ -n "$lane_in" ]]; then
  run cov "$(realpath "$lane_in")"
  ( cd cov; gcov module_mp_fer_hires.F > /dev/null 2>&1; mv module_mp_fer_hires.F.gcov lane.gcov
    mkdir -p lane-out; mv levels.csv surface.csv lane-out/; rm -f *.gcda )
fi
run cov "$out/columns.in"
( cd cov; gcov module_mp_fer_hires.F > /dev/null 2>&1; mv module_mp_fer_hires.F.gcov refute.gcov )
{ echo "wrf_commit d66e442fccc04111067e29274c9f9eaccc3cef28"
  echo "gfortran $(gfortran -dumpfullversion)"; echo "glibc $(ldd --version | head -1 | awk '{print $NF}')"
  echo "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
  sha256sum columns_inputs.csv */levels.csv */surface.csv; } > SHA256SUMS
cat SHA256SUMS
for t in O2vec O3native cov; do
  echo "== O0 vs $t"; python3 "$h/compare_ferrier.py" O0 "$t" columns_inputs.csv || true
done
