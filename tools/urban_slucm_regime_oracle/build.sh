#!/usr/bin/env bash
# Build and run the SLUCM regime column oracle (sf_urban_physics = 1).
#
#   bash tools/urban_slucm_regime_oracle/build.sh WRF_SRC_DIR BUILD_DIR OUT_DIR
#
# WRF_SRC_DIR is what tools/urban_wrf461_oracle/fetch_sources.sh writes (the
# five pinned v4.6.1 files plus SOURCES.sha256); a v4.7.1 directory with its
# own SOURCES.sha256 works the same way.  Every file compiled or read is
# checked against SOURCES.sha256 before anything is built.
#
# Optional environment:
#   UCM_LIBM_SHIM=1   link tools/urban_slucm_regime_oracle/cr_atan_log10.c
#                     ahead of libm: atanf and log10f become the double
#                     routines rounded once to float (correctly rounded on
#                     every input the kernel's own sweep measured).  Used
#                     only to attribute a difference to those two calls on a
#                     C library whose atanf/log10f are not correctly rounded.
#
# Builds, all gfortran -O0 (the WRF physics compiled with no contraction):
#   ref   THE FIXTURE for every variant without the green roof
#   snan  -finit-real=snan: any row that differs from ref read an
#         uninitialised local; must be none outside the green-roof dew arm
#   zero  -finit-real=zero: the green-roof fixture.  WRF reads ETR undefined
#         on the green-roof dew arm (module_sf_urban.F, SMFLX/SRT before
#         TRANSP); the engine defines ETR = 0 there, which is this build.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: build.sh WRF_SRC_DIR BUILD_DIR OUT_DIR" >&2
    exit 2
fi
src=$(realpath "$1")
build=$(realpath -m "$2")
out=$(realpath -m "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stub=$here/../urban_wrf471_oracle/stub_ucm.F90

for f in module_sf_urban.F module_model_constants.F module_wrf_error.F \
         URBPARM.TBL URBPARM_LCZ.TBL; do
    want=$(awk -v n="$f" '$2 == n || $2 ~ ("/" n "$") {print $1}' "$src/SOURCES.sha256" | sed -n 1p)
    got=$(sha256sum "$src/$f" | cut -d' ' -f1)
    if [[ -z "$want" || "$want" != "$got" ]]; then
        echo "$f does not match SOURCES.sha256 (want '$want', got '$got')" >&2
        exit 3
    fi
done

mkdir -p "$build" "$out"
cd "$build"
defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"
shim_obj=()
if [[ "${UCM_LIBM_SHIM:-0}" == 1 ]]; then
    gcc -O2 -fno-builtin -c "$here/cr_atan_log10.c" -o cr_atan_log10.o
    shim_obj=("$build/cr_atan_log10.o")
fi

build_one() {   # $1 = tag, rest = extra flags
    local tag=$1; shift
    mkdir -p "$tag"
    ( cd "$tag"
      gfortran -c -O0 "$@" ${free} "$stub"
      gfortran -c -O0 "$@" -cpp ${defines} ${free} "$src/module_model_constants.F"
      gfortran -c -O0 "$@" -cpp ${defines} ${free} -fallow-argument-mismatch "$src/module_wrf_error.F"
      gfortran -c -O0 "$@" -cpp ${defines} ${free} -fallow-argument-mismatch -I. "$src/module_sf_urban.F"
      gfortran -c -O0 "$@" ${free} -I. "$here/slucm_regime_oracle.F90"
      gfortran -o run_ucm "${shim_obj[@]}" stub_ucm.o module_model_constants.o \
          module_wrf_error.o module_sf_urban.o slucm_regime_oracle.o
      cp "$src/URBPARM.TBL" "$src/URBPARM_LCZ.TBL" . )
}
build_one ref
build_one snan -finit-real=snan -finit-integer=-2147483647 -finit-logical=false
build_one zero -finit-real=zero

nm -u ref/module_sf_urban.o | sed 's/^ *//' | sort > undefined-O0-urban.txt
if grep -q '_ZGV' undefined-O0-urban.txt; then
    echo "-O0 module_sf_urban.o pulled in a libmvec vector symbol" >&2
    exit 4
fi
{
    echo "# gfortran: $(gfortran --version | sed -n 1p)"
    echo "# glibc:    $(ldd --version | sed -n 1p)"
    echo "# libm shim (atanf/log10f correctly rounded): ${UCM_LIBM_SHIM:-0}"
    echo "# module_sf_urban.F sha256: $(sha256sum "$src/module_sf_urban.F" | cut -d' ' -f1)"
    echo "# -O0 module_sf_urban.o libm symbols:"
    grep -E 'expf|powf|logf|log10f|atanf|atan2f|sqrtf|sinf|cosf|tanf|asinf|acosf|__powisf2|_ZGV' \
        undefined-O0-urban.txt || true
} > "$out/libm-report.txt"

# TABLE VARIANT CH TS AH ALH IMP IRI GR FGR BOUND   (-9 = table value)
variants=(
    "nlcd default     -9 -9 -9 -9 -9 -9 -9 -9 -9"
    "lcz  default     -9 -9 -9 -9 -9 -9 -9 -9 -9"
    "nlcd ch1         1  -9 -9 -9 -9 -9 -9 -9 -9"
    "nlcd ts2         -9 2  -9 -9 -9 -9 -9 -9 -9"
    "nlcd ch1ts2      1  2  -9 -9 -9 -9 -9 -9 -9"
    "nlcd ahalh       -9 -9 1  1  -9 -9 -9 -9 -9"
    "nlcd imp2        -9 -9 -9 -9 2  -9 -9 -9 -9"
    "nlcd bound2      -9 -9 -9 -9 -9 -9 -9 -9 2"
    "lcz  imp2ahalh   -9 -9 1  1  2  -9 -9 -9 -9"
    "lcz  ch1ahalh    1  -9 1  1  -9 -9 -9 -9 -9"
)
gr_variants=(
    "nlcd gr          -9 -9 -9 -9 -9 -9 1  0.5 -9"
    "nlcd griri       -9 -9 1  -9 2  1  1  0.7 -9"
    "lcz  gr          -9 -9 -9 -9 -9 -9 1  0.4 -9"
    "lcz  grch1       1  -9 -9 -9 -9 1  1  0.6 -9"
)
run_set() {   # $1 = build tag, $2.. = variant lines
    local tag=$1; shift
    for line in "$@"; do
        set -- $line
        ( cd "$build/$tag" && ./run_ucm "$1" "$2" "$build/$tag/slucm-$1-$2" \
              "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" )
    done
}
run_set ref "${variants[@]}" "${gr_variants[@]}"
run_set snan "${variants[@]}" "${gr_variants[@]}"
run_set zero "${gr_variants[@]}"

for line in "${variants[@]}"; do
    set -- $line
    name="slucm-$1-$2"
    if ! cmp -s "ref/$name.csv" "snan/$name.csv"; then
        echo "$name: -finit-real=snan changes the output; an uninitialised read reaches a row" >&2
        exit 6
    fi
    cp "ref/$name.csv" "ref/$name-table.csv" "ref/$name-switches.csv" "$out/"
    gzip -n -9 -f "$out/$name.csv"
done
for line in "${gr_variants[@]}"; do
    set -- $line
    name="slucm-$1-$2"
    cp "zero/$name.csv" "zero/$name-table.csv" "zero/$name-switches.csv" "$out/"
    gzip -n -9 -f "$out/$name.csv"
    ndiff=$(paste -d'\n' "ref/$name.csv" "zero/$name.csv" | awk 'NR%2{a=$0;next}{if(a!=$0)n++}END{print n+0}')
    nnan=$(grep -c -i 'nan' "snan/$name.csv" || true)
    echo "$name: rows differing ref vs zero-init: $ndiff; snan rows with NaN: $nnan" >> "$out/libm-report.txt"
done
( cd "$out" && sha256sum *.csv *.csv.gz > oracle-sha256sums.txt )
gfortran --version | sed -n 1p > "$out/compiler.txt"
ldd --version | sed -n 1p >> "$out/compiler.txt"
echo "slucm regime oracle written to $out"
