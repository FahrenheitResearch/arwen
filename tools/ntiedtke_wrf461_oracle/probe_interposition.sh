#!/usr/bin/env bash
# Can we capture a private routine's arguments AT THE REAL CALL SITE?
#
# WHY THIS MATTERS.  Every harness so far SYNTHESISES the inputs of the
# routine it grades -- reconstructing what cumastrn would have passed --
# and that has been wrong three times in five slices:
#
#   cutypen : passed FRESH arrays where cumastrn passes live ones
#   4a      : skipped cumastrn:500-541, so pmfub was 0
#   4b      : never captured paph at the surface interface
#
# Each time the oracle and the mirror agreed, because they agreed with each
# other about a state WRF never visits.  max_ulp == 0 is structurally blind
# to it: there is nothing for the reference to disagree with.
#
# The fix is to stop reconstructing and start CAPTURING: run the real chain
# from the public entry point and record the arguments the real cumastrn
# actually passes.  cumastrn's calls to its private routines are
# same-translation-unit, so `ld --wrap` cannot reach them -- those
# references are bound at compile time.  A SHARED LIBRARY can: calls to
# global symbols go through the PLT, and LD_PRELOAD interposes there.
#
# Two things have to hold, and this script tests both rather than assuming:
#
#   1. -fPIC must not move a single bit.  The oracle's arithmetic has to be
#      the same arithmetic.  Checked by running the pinned driver harness
#      against both builds and diffing the CSVs byte for byte.
#   2. Interposition must actually reach an INTERNAL call -- cumastrn ->
#      cuentrn -- not merely an external one.
#
# If either fails, say so and fall back to replicating cumastrn's body.
set -euo pipefail

src=${1:-/tmp/wrf461src}
work=${2:-/tmp/ntinterpose}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

rm -rf "$work"; mkdir -p "$work"; cd "$work"

# --- 1. the PIC shared library ---------------------------------------------
gfortran -c -O0 -cpp -DRWORDSIZE=4 -ffree-form -ffree-line-length-none \
    -fPIC -o k.o "$src/phys/ccpp_kind_types.F"
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none -fPIC -I . \
    -o s.o "$src/phys/physics_mmm/cu_ntiedtke.F90"
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none -fPIC -I . \
    -o m.o "$src/phys/module_cu_ntiedtke.F"

# the private routines must be global before anything can interpose them
objcopy \
    --globalize-symbol=__cu_ntiedtke_MOD_cumastrn \
    --globalize-symbol=__cu_ntiedtke_MOD_cuinin \
    --globalize-symbol=__cu_ntiedtke_MOD_cutypen \
    --globalize-symbol=__cu_ntiedtke_MOD_cuadjtqn \
    --globalize-symbol=__cu_ntiedtke_MOD_cubasmcn \
    --globalize-symbol=__cu_ntiedtke_MOD_cuentrn \
    --globalize-symbol=__cu_ntiedtke_MOD_cudlfsn \
    --globalize-symbol=__cu_ntiedtke_MOD_cuddrafn \
    --globalize-symbol=__cu_ntiedtke_MOD_cuflxn \
    --globalize-symbol=__cu_ntiedtke_MOD_cudtdqn \
    --globalize-symbol=__cu_ntiedtke_MOD_cududvn \
    s.o s_glob.o
gfortran -shared -o libnt.so k.o s_glob.o m.o

# --- 2. does -fPIC move any bits? ------------------------------------------
gfortran -c -O0 -ffree-form -ffree-line-length-none -fPIC -I . \
    "$here/nt_cases.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none -fPIC -I . \
    "$here/run_cu_ntiedtke.F90"
gfortran -o run_pic run_cu_ntiedtke.o nt_cases.o -L. -lnt -Wl,-rpath,"$work"
mkdir -p picrun && (cd picrun && LD_LIBRARY_PATH="$work" ../run_pic >/dev/null)

echo "=== -fPIC vs the pinned static build, byte for byte ==="
pic_ok=1
for f in nt-levels.csv nt-surface.csv; do
    if cmp -s "picrun/$f" "/tmp/ntbuild/$f"; then
        echo "  $f  IDENTICAL"
    else
        echo "  $f  *** DIFFERS -- PIC moved bits, this route is unusable ***"
        pic_ok=0
    fi
done

# --- 3. does interposition reach an INTERNAL call? -------------------------
# cumastrn -> cuentrn is entirely inside cu_ntiedtke.F90.  If the shim runs,
# a same-translation-unit call went through the PLT and is interposable.
cat > shim.c <<'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
static long n_calls = 0;
typedef void (*fn_t)(int*, int*, int*, int*, int*, int*, float*, float*,
                     float*, float*);
void __cu_ntiedtke_MOD_cuentrn(int *klon, int *klev, int *kk, int *kcbot,
                               int *ldcum, int *ldwork, float *pgeoh,
                               float *pmfu, float *pdmfen, float *pdmfde) {
    static fn_t real = 0;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "__cu_ntiedtke_MOD_cuentrn");
    n_calls++;
    real(klon, klev, kk, kcbot, ldcum, ldwork, pgeoh, pmfu, pdmfen, pdmfde);
}
__attribute__((destructor)) static void report(void) {
    FILE *f = fopen("interpose-count.txt", "w");
    if (f) { fprintf(f, "%ld\n", n_calls); fclose(f); }
}
CEOF
gcc -shared -fPIC -o shim.so shim.c -ldl

mkdir -p shimrun
(cd shimrun && LD_PRELOAD="$work/shim.so" LD_LIBRARY_PATH="$work" \
    ../run_pic >/dev/null)

echo
echo "=== interposition on an INTERNAL call (cumastrn -> cuentrn) ==="
if [[ -s shimrun/interpose-count.txt ]]; then
    calls=$(cat shimrun/interpose-count.txt)
    if [[ "${calls}" -gt 0 ]]; then
        echo "  shim saw ${calls} calls -- INTERPOSITION WORKS"
    else
        echo "  shim loaded but saw 0 calls -- internal calls bypass the PLT"
    fi
else
    echo "  shim never ran"
fi

echo
echo "=== and the interposed run must still produce the pinned answer ==="
for f in nt-levels.csv nt-surface.csv; do
    if cmp -s "shimrun/$f" "/tmp/ntbuild/$f"; then
        echo "  $f  IDENTICAL"
    else
        echo "  $f  *** DIFFERS -- the shim perturbed the answer ***"
    fi
done

[[ "${pic_ok}" -eq 1 ]] || exit 9
