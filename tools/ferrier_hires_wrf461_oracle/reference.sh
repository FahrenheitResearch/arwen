#!/usr/bin/env bash
# The graded WRF v4.6.1 Ferrier-Aligo references for one column set.  CPU only.
#
# usage: reference.sh WRF_SOURCE_ROOT BUILD_DIR COLUMNS_DIR
#
# COLUMNS_DIR holds columns.in and columns_inputs.csv (make_columns*.py).
# Builds the stock O0, stock O2-vectorized and stock coverage references.
# Every comparison uses byte-unmodified WRF physics source.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 WRF_SOURCE_ROOT BUILD_DIR COLUMNS_DIR" >&2; exit 2; }
src=$(realpath "$1"); out=$(realpath -m "$2"); cols=$(realpath "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fer="$src/phys/module_mp_fer_hires.F"; table="$src/run/ETAMPNEW_DATA.expanded_rain"
[[ $(git -C "$src" rev-parse HEAD) == d66e442fccc04111067e29274c9f9eaccc3cef28 ]]
git -C "$src" diff --quiet HEAD -- phys/module_mp_fer_hires.F run/ETAMPNEW_DATA.expanded_rain
mkdir -p "$out"; cd "$out"
cp "$cols/columns.in" "$cols/columns_inputs.csv" .
common=(-cpp -ffree-form -ffree-line-length-none -DRWORDSIZE=4 -DIWORDSIZE=4 -fallow-argument-mismatch)
build() {  # build TAG PATCH(0/1) FLAGS...
  local tag=$1 patched=$2; shift 2
  mkdir -p "$tag"; ( cd "$tag"
    mkdir -p phys; cp "$fer" phys/module_mp_fer_hires.F
    if [[ $patched == 1 ]]; then patch -s -p1 < "$here/rcw_fix.patch"; fi
    cp phys/module_mp_fer_hires.F module_mp_fer_hires.F
    gfortran -c "$@" "${common[@]}" "$here/stub_wrf.F90"
    gfortran -c "$@" "${common[@]}" module_mp_fer_hires.F 2>/dev/null
    gfortran -c "$@" "${common[@]}" -fconvert=big-endian "$here/run_ferrier.F90"
    gfortran "$@" -o run_ferrier stub_wrf.o module_mp_fer_hires.o run_ferrier.o
    nm -u module_mp_fer_hires.o | sed 's/^ *//' | sort > undefined-symbols.txt
    ln -sf "$table" ETAMPNEW_DATA.expanded_rain
    ln -sf ../columns.in columns.in
    ./run_ferrier > run.stdout 2> run.stderr )
}
build O0 0 -O0
build O2vec 0 -O2 -ftree-vectorize -funroll-loops
build cov 0 -O0 --coverage
for t in O0 O2vec; do
  if grep -q _ZGV "$t/undefined-symbols.txt"; then echo "$t pulled libmvec" >&2; exit 4; fi
done
( cd cov && gcov module_mp_fer_hires.F > /dev/null 2>&1 )
python3 "$here/compare_ferrier.py" cov O0 columns_inputs.csv > cov-vs-O0.txt || true
{ echo "wrf_commit d66e442fccc04111067e29274c9f9eaccc3cef28"
  echo "gfortran $(gfortran -dumpfullversion)"
  echo "glibc $(ldd --version | head -1 | awk '{print $NF}')"
  sha256sum columns_inputs.csv O0/levels.csv O0/surface.csv O2vec/levels.csv O2vec/surface.csv
} > SHA256SUMS
cat SHA256SUMS
echo "== stock O0 vs stock O2vec"
python3 "$here/compare_ferrier.py" O0 O2vec columns_inputs.csv || true
