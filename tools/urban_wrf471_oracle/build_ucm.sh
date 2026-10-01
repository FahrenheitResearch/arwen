#!/usr/bin/env bash
# Build and run the single-layer UCM column oracle (sf_urban_physics = 1)
# against the byte-unmodified WRF v4.7.1 phys/module_sf_urban.F.
#
#   bash tools/urban_wrf471_oracle/build_ucm.sh WRF_SRC_DIR BUILD_DIR [OUT_DIR]
#
# WRF_SRC_DIR holds the pinned files flat (the design stage's wrf-src/ copy:
# module_sf_urban.F, module_model_constants.F, module_wrf_error.F,
# URBPARM.TBL, URBPARM_LCZ.TBL, SOURCES.sha256).  Every file compiled or read
# is checked against SOURCES.sha256 before anything is built, in place of the
# git-commit check the v4.6.1 harnesses use.
#
# Three builds of the same sources:
#   ref   -O0                          THE FIXTURE
#   snan  -O0 -finit-real=snan ...     negative control: any row whose value
#                                      differs from ref read an uninitialised
#                                      local.  The one known reader is the
#                                      green-roof dew arm (below).
#   zero  -O0 -finit-real=zero         the green-roof dew fixture: WRF's
#                                      undefined ETR read made defined as 0.
#
# WRF DEFECT, recorded not reproduced.  module_sf_urban.F:1161-1168: when the
# green-roof potential evaporation EPGR <= 0 on the FIRST Newton iteration,
# SMFLX is called with ETR (a local of `urban`, :571) that no statement has
# written in this call -- TRANSP, which zeroes it, runs only when EPGR > 0.
# SRT then reads ET(1..3) (:3590, :3612).  The port defines ETR = 0 at entry
# (no transpiration under dew), which is what the `zero` build computes; the
# `snan` build proves the read is real (its dew rows go NaN).
#
# libmvec: same discipline as tools/noah_wrf461_oracle/build.sh.  The -O0
# object must carry no _ZGV* symbol and the positive control must.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build_ucm.sh WRF_SRC_DIR BUILD_DIR [OUT_DIR]" >&2
    exit 2
fi
src=$(realpath "$1")
build=$(realpath -m "$2")
out=$(realpath -m "${3:-$2/fixture}")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

files=(module_sf_urban.F module_model_constants.F module_wrf_error.F
       URBPARM.TBL URBPARM_LCZ.TBL)
for f in "${files[@]}"; do
    want=$(awk -v n="$f" '$2 == n {print $1}' "$src/SOURCES.sha256")
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

build_one() {   # $1 = tag, rest = extra flags
    local tag=$1; shift
    mkdir -p "$tag"
    ( cd "$tag"
      gfortran -c -O0 "$@" ${free} "$here/stub_ucm.F90"
      gfortran -c -O0 "$@" -cpp ${defines} ${free} "$src/module_model_constants.F"
      gfortran -c -O0 "$@" -cpp ${defines} ${free} -fallow-argument-mismatch "$src/module_wrf_error.F"
      gfortran -c -O0 "$@" -cpp ${defines} ${free} -fallow-argument-mismatch -I. "$src/module_sf_urban.F"
      gfortran -c -O0 "$@" ${free} -I. "$here/ucm_column_oracle.F90"
      gfortran -o run_ucm stub_ucm.o module_model_constants.o module_wrf_error.o \
          module_sf_urban.o ucm_column_oracle.o
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
cat > libmvec_control.f90 <<'EOF'
subroutine ctl(x, n)
  integer n, i
  real x(n)
  do i = 1, n
    x(i) = exp(x(i))
  end do
end subroutine
EOF
gfortran -c -Ofast -ftree-vectorize -o libmvec_control.o libmvec_control.f90
if ! nm -u libmvec_control.o | grep -q '_ZGV'; then
    echo "libmvec positive control emitted no _ZGV symbol; the guard is vacuous" >&2
    exit 5
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 module_sf_urban.o libm symbols (THE REFERENCE):"
    grep -E 'expf|powf|logf|log10f|atanf|atan2f|sqrtf|sinf|cosf|tanf|asinf|acosf|__powisf2|_ZGV' \
        undefined-O0-urban.txt || true
    echo "# positive control (-Ofast expf loop):"
    nm -u libmvec_control.o | grep -E 'expf|_ZGV' || true
} > "$out/libm-report.txt"

# variant table: TABLE VARIANT CH TS AH ALH IMP IRI GR FGR BOUND   (-9 = table value)
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
)
# The green roof (GROPTION 1) reads ETR undefined on dew; its variants are
# produced by the `zero` build (see the header) and checked against `ref` on
# every row that does not take the dew arm.
gr_variants=(
    "nlcd gr          -9 -9 -9 -9 -9 -9 1  0.5 -9"
    "nlcd griri       -9 -9 1  -9 2  1  1  0.7 -9"
    "lcz  gr          -9 -9 -9 -9 -9 -9 1  0.4 -9"
)
run_set() {   # $1 = build tag, $2.. = variant lines
    local tag=$1; shift
    for line in "$@"; do
        set -- $line
        ( cd "$build/$tag" && ./run_ucm "$1" "$2" "$build/$tag/ucm-$1-$2" \
              "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" )
    done
}
run_set ref "${variants[@]}" "${gr_variants[@]}"
run_set snan "${variants[@]}" "${gr_variants[@]}"
run_set zero "${gr_variants[@]}"

# Non-green-roof variants: ref must equal snan byte for byte (no uninitialised
# read reaches any row), and ref is the fixture.
for line in "${variants[@]}"; do
    set -- $line
    name="ucm-$1-$2"
    if ! cmp -s "ref/$name.csv" "snan/$name.csv"; then
        echo "$name: -finit-real=snan changes the output; an uninitialised read reaches a row" >&2
        exit 6
    fi
    cp "ref/$name.csv" "ref/$name-table.csv" "ref/$name-switches.csv" "$out/"
    gzip -n -9 -f "$out/$name.csv"
done
# Green-roof variants: the fixture is the `zero` build.  Rows whose ref and
# zero values agree did not take the undefined read; the count that differ is
# recorded in the report.
for line in "${gr_variants[@]}"; do
    set -- $line
    name="ucm-$1-$2"
    cp "zero/$name.csv" "zero/$name-table.csv" "zero/$name-switches.csv" "$out/"
    gzip -n -9 -f "$out/$name.csv"
    ndiff=$(paste -d'\n' "ref/$name.csv" "zero/$name.csv" | awk 'NR%2{a=$0;next}{if(a!=$0)n++}END{print n+0}')
    nnan=$(grep -c -i 'nan' "snan/$name.csv" || true)
    echo "$name: rows differing ref vs zero-init: $ndiff; snan rows with NaN: $nnan" >> "$out/libm-report.txt"
done
( cd "$out" && sha256sum *.csv *.csv.gz > oracle-sha256sums.txt )
gfortran --version | head -1 > "$out/compiler.txt"
ldd --version | head -1 >> "$out/compiler.txt"
echo "ucm oracle written to $out"
