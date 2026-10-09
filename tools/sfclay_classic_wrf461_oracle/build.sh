#!/usr/bin/env bash
# Build and run the classic MM5 surface-layer oracle (sf_sfclay_physics=91)
# against the byte-unmodified WRF v4.6.1 phys/module_sf_sfclay.F.
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_TREE]
#
# WRF_SOURCE_ROOT  pristine v4.6.1 checkout (commit pinned below).
# BUILD_DIR        where objects, the driver and the outputs go.
# WRF_BUILD_TREE   optional: a compiled WRF tree (configure + compile em_real
#                  done).  Its own phys/module_sf_sfclay.o -- the object
#                  inside that wrf.exe -- is linked into a third driver.
#
# Three references come out:
#
#   wrf-O0.txt     the oracle.  module_sf_sfclay.F at -O0: every ALOG, EXP,
#                  ATAN and REAL**REAL is one scalar libm call.  build.sh
#                  fails if the object pulls in any libmvec (_ZGV*) symbol.
#   wrf-stock.txt  the same source at WRF's own configure.wrf flags
#                  (-O2 -ftree-vectorize -funroll-loops).  At these flags
#                  gfortran vectorises SFCLAYINIT's table loop and SFCLAY1D's
#                  simple loops through libmvec.  Measured, not graded.
#   wrf-built.txt  (only with WRF_BUILD_TREE) the object a real wrf.exe links.
#
# Why the -O0 build is the oracle and not the stock object: libmvec's SSE
# routines carry a 4 ULP contract instead of the scalar ~0.5 ULP, and the
# vectoriser applies them only to the columns that fall in a full vector
# body; the scalar remainder of the same loop calls scalar powf.  So a stock
# wrf.exe gives a column different bits depending on its position in the
# tile (and the table on the host's libmvec build).  That is a property of
# the build, not of the scheme; the -O0 object is the scheme's arithmetic.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_TREE]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
built_tree=${3:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sfclay_source="${source_root}/phys/module_sf_sfclay.F"
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"

test -f "${sfclay_source}"

# --- byte identity to the pinned commit -------------------------------------
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- phys/module_sf_sfclay.F; then
    echo "module_sf_sfclay.F differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"/{O0,stock,built}
cd "${build_dir}"
sha256sum "${sfclay_source}" | sed "s|${source_root}/||" > source-sha256.txt

python3 "${script_dir}/make_columns.py" "${build_dir}/columns.txt"

# --- the reference build: -O0, no vectoriser --------------------------------
# WRF preprocesses .F with cpp -P -traditional and -DEM_CORE=1 on an em_real
# build; that #if is the only one in the file and selects the optional-
# argument call that passes ISFTCFLX/IZ0TLND/USTM/CK/CKA/CD/CDA.
(cd O0
 gfortran -c -O0 -cpp -DEM_CORE=1 -ffree-form -ffree-line-length-none \
     -o module_sf_sfclay.o "${sfclay_source}"
 gfortran -c -O0 -ffree-form -ffree-line-length-none -I . \
     "${script_dir}/run_sfclay_classic.F90"
 gfortran -o run_sfclay_classic module_sf_sfclay.o run_sfclay_classic.o
 nm -u module_sf_sfclay.o | sed 's/^ *//' | sort > undefined.txt)
if grep -q '_ZGV' O0/undefined.txt; then
    echo "-O0 object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' O0/undefined.txt >&2
    exit 4
fi
O0/run_sfclay_classic columns.txt wrf-O0.txt

# WOOF corrects WRF :767: PSIQ10 must use PSIH10. Grade every CK word
# against a labelled referee with only that line changed, rather than
# accepting any value at the old CK holdout. Keep the shared WRF pristine.
mkdir -p corrected
cp "${sfclay_source}" corrected/module_sf_sfclay.F
(cd corrected
 patch --batch -p2 < "${script_dir}/wrf-ck-correction.patch"
 gfortran -c -O0 -cpp -DEM_CORE=1 -ffree-form -ffree-line-length-none \
     -o module_sf_sfclay.o module_sf_sfclay.F
 gfortran -c -O0 -ffree-form -ffree-line-length-none -I . \
     "${script_dir}/run_sfclay_classic.F90"
 gfortran -o run_sfclay_classic module_sf_sfclay.o run_sfclay_classic.o
 sha256sum module_sf_sfclay.F > source-sha256.txt)
corrected/run_sfclay_classic columns.txt wrf-ck-corrected.txt

# --- WRF's stock flags (configure.wrf GNU entry, unchanged) ------------------
(cd stock
 gfortran -c -O2 -ftree-vectorize -funroll-loops -w -cpp -DEM_CORE=1 \
     -ffree-form -ffree-line-length-none -fconvert=big-endian \
     -frecord-marker=4 -fallow-argument-mismatch -fallow-invalid-boz \
     -o module_sf_sfclay.o "${sfclay_source}"
 gfortran -c -O0 -ffree-form -ffree-line-length-none -I . \
     "${script_dir}/run_sfclay_classic.F90"
 gfortran -o run_sfclay_classic module_sf_sfclay.o run_sfclay_classic.o
 nm -u module_sf_sfclay.o | sed 's/^ *//' | sort > undefined.txt)
if ! grep -q '_ZGV' stock/undefined.txt; then
    echo "note: the stock-flag object has no libmvec symbol on this toolchain" >&2
fi
stock/run_sfclay_classic columns.txt wrf-stock.txt

# --- the object inside a real wrf.exe ----------------------------------------
if [[ -n "${built_tree}" ]]; then
    built_obj="${built_tree}/phys/module_sf_sfclay.o"
    test -f "${built_obj}"
    mod_dir=$(dirname "$(find "${built_tree}" -name module_sf_sfclay.mod | head -1)")
    (cd built
     cp "${built_obj}" module_sf_sfclay.o
     gfortran -c -O0 -ffree-form -ffree-line-length-none -I "${mod_dir}" \
         "${script_dir}/run_sfclay_classic.F90"
     gfortran -o run_sfclay_classic module_sf_sfclay.o run_sfclay_classic.o
     nm -u module_sf_sfclay.o | sed 's/^ *//' | sort > undefined.txt
     sha256sum module_sf_sfclay.o > object-sha256.txt)
    built/run_sfclay_classic columns.txt wrf-built.txt
fi

{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "libmvec symbols, -O0 object:  $(grep -c '_ZGV' O0/undefined.txt || true)"
    echo "libmvec symbols, stock object: $(grep '_ZGV' stock/undefined.txt | tr '\n' ' ')"
    if [[ -f built/undefined.txt ]]; then
        echo "libmvec symbols, wrf.exe object: $(grep '_ZGV' built/undefined.txt | tr '\n' ' ')"
    fi
} > toolchain.txt
sha256sum columns.txt wrf-*.txt > oracle-sha256sums.txt
cat toolchain.txt
cat oracle-sha256sums.txt
