#!/usr/bin/env bash
# Build and run the wide Shin-Hong column oracle against the byte-unmodified
# WRF v4.6.1 phys/module_bl_shinhong.F, then write the column inputs and the
# WRF outputs into OUT_DIR.
#
#   bash tools/shinhong_column_oracle/build.sh WRF_SOURCE_ROOT OUT_DIR [PYTHON]
#
# The module is pinned by its sha256 (the v4.6.1 release file; the same pin
# tools/shinhong_wrf461_oracle/build.sh uses).  Three objects are built from
# it and each one runs every column:
#
#   wrf-out.bin        THE REFERENCE.  WRF's own phys/ flags from
#                      arch/configure.defaults (GNU: -O2 -funroll-loops)
#                      with the vectoriser off (-fno-tree-vectorize), so
#                      every libm call is the scalar one.  This is the
#                      arithmetic a stock WRF build performs, minus the
#                      libmvec SIMD calls.  At -O2 GCC folds x**2. to x*x
#                      (and x**0. to 1), as the shipping build does.
#   wrf-out-O0.bin     -O0.  gfortran 13.3 at -O0 keeps all 17 x**2. sites
#                      as calls to the C library's powf(x, 2.0), and on this
#                      host that call differs from x*x for hundreds of
#                      thousands of normal float32 inputs (measured
#                      exhaustively below, receipt build/pow2-receipt.txt).
#                      No WRF build compiles that way; kept as evidence.
#   wrf-out-O2vec.bin  WRF's flags exactly (-O2 -ftree-vectorize
#                      -funroll-loops): the shipping build, whose vectorised
#                      loops call libmvec's _ZGVbN4vv_powf.  Context only.
#
# The reference build FAILS if its object pulls in any _ZGV* (libmvec)
# symbol or contains an FMA instruction.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT OUT_DIR [PYTHON]" >&2
    exit 2
fi
source_root=$(realpath "$1")
out_dir=$(realpath -m "$2")
python=${3:-python3}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
shinhong_source="${source_root}/phys/module_bl_shinhong.F"
pinned_sha256="99f44dbeb5e586b96be14424b8ab27c9986ffbd81f007f41fb8528d8ea466d56"

actual_sha256=$(sha256sum "${shinhong_source}" | awk '{print $1}')
if [[ "${actual_sha256}" != "${pinned_sha256}" ]]; then
    echo "module_bl_shinhong.F is ${actual_sha256}, not the pinned ${pinned_sha256}" >&2
    exit 3
fi

mkdir -p "${out_dir}/build"
cd "${out_dir}/build"
fflags="-cpp -ffree-form -ffree-line-length-none"

build_arm() {   # name, compiler flags...
    local name=$1
    shift
    mkdir -p "mod-${name}"
    gfortran -c "$@" ${fflags} -J "mod-${name}" \
        -o "module_bl_shinhong_${name}.o" "${shinhong_source}"
    gfortran -c -O0 -ffree-form -ffree-line-length-none -I "mod-${name}" \
        -o "run_shinhong_columns_${name}.o" "${script_dir}/run_shinhong_columns.F90"
    gfortran -o "run_shinhong_columns_${name}" \
        "module_bl_shinhong_${name}.o" "run_shinhong_columns_${name}.o"
    nm -u "module_bl_shinhong_${name}.o" | sed 's/^ *//' | sort \
        > "undefined-${name}.txt"
    objdump -dr "module_bl_shinhong_${name}.o" \
        | awk '/R_X86_64_PLT32/ {print $3}' | sed 's/-0x4$//' \
        | grep -E 'pow|exp|tanh|sqrt|_ZGV' | sort | uniq -c \
        > "libm-calls-${name}.txt" || true
}

build_arm O2s -O2 -funroll-loops -fno-tree-vectorize
build_arm O0 -O0
build_arm O2vec -O2 -ftree-vectorize -funroll-loops

if grep -q '_ZGV' undefined-O2s.txt; then
    echo "reference object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-O2s.txt >&2
    exit 4
fi
if objdump -d module_bl_shinhong_O2s.o | grep -Eq 'vfmadd|vfmsub|vfnmadd|vfnmsub'; then
    echo "reference object contains FMA instructions" >&2
    exit 5
fi

# Receipt: how often the C library's powf(x, 2.0) is not x*x on this host,
# over every positive finite float32 (the -O0 artefact described above).
cat > pow2.c <<'CEOF'
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
int main(void) {
    long normal = 0, other = 0;
    #pragma omp parallel for reduction(+:normal,other) schedule(static)
    for (long b = 1; b < 0x7f800000L; ++b) {
        uint32_t u = (uint32_t)b; float x; memcpy(&x, &u, 4);
        volatile float p = powf(x, 2.0f); volatile float m = x * x;
        if (memcmp((const void *)&p, (const void *)&m, 4)) {
            if (isfinite(m) && fabsf(m) >= 1.17549435e-38f) ++normal; else ++other;
        }
    }
    printf("powf(x,2.0f) != x*x on %ld positive float32 x with a normal finite"
           " square, and %ld more in the subnormal/overflow range\n",
           normal, other);
    return 0;
}
CEOF
gcc -O1 -fopenmp -fno-builtin -o pow2 pow2.c -lm
OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} ./pow2 > pow2-receipt.txt

cd "${out_dir}"
"${python}" "${script_dir}/make_columns.py" columns-in.bin columns.json
./build/run_shinhong_columns_O2s columns-in.bin wrf-out.bin
./build/run_shinhong_columns_O0 columns-in.bin wrf-out-O0.bin
./build/run_shinhong_columns_O2vec columns-in.bin wrf-out-O2vec.bin

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    for arm in O2s O0 O2vec; do
        echo "# ${arm}: libm call sites in the module object"
        cat "build/libm-calls-${arm}.txt"
    done
    echo "# $(cat build/pow2-receipt.txt)"
} > toolchain.txt
sha256sum "${shinhong_source}" "${script_dir}/run_shinhong_columns.F90" \
    "${script_dir}/make_columns.py" columns-in.bin wrf-out.bin \
    wrf-out-O0.bin wrf-out-O2vec.bin \
    | sed "s#${source_root}#<wrf-4.6.1-checkout>#; s#${script_dir}#<oracle-dir>#" \
    > oracle-sha256sums.txt
cat toolchain.txt
echo "oracle written to ${out_dir}"
