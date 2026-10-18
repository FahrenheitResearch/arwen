#!/usr/bin/env bash
# Build and run the swint_opt = 1 column oracle from byte-unmodified WRF
# v4.6.1 (CPU only).
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# Writes into BUILD_DIR:
#   columns.npz, <scenario>.in.bin   the columns (make_columns.py)
#   swint_wrf461.F90                 the four routines, cut unchanged
#                                    (extract.py)
#   <scenario>.O0.out.bin            reference of record: -O0, scalar libm,
#                                    no FMA
#   <scenario>.O0cr.out.bin          the same -O0 binary with a correctly
#                                    rounded asinf preloaded (asinf_cr.so)
#   <scenario>.O2vec.out.bin         evidence only: WRF's stock GNU physics
#                                    flags, as wrf.exe compiles the driver
#   receipt.txt                      commit, sha256s, compiler, symbols
# Then compare_gpu.py (through the GPU mutex) grades WOOF against them.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- \
        share/module_model_constants.F phys/module_radiation_driver.F; then
    echo "a compiled WRF file differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"
{
    echo "wrf commit ${head_commit}"
    gfortran --version | head -1
    ldd --version | head -1
    "${python}" "${script_dir}/extract.py" "${source_root}" swint_wrf461.F90
    sha256sum "${source_root}/share/module_model_constants.F" \
        "${source_root}/phys/module_radiation_driver.F" swint_wrf461.F90 \
        "${script_dir}/run_swint.F90" "${script_dir}/make_columns.py"
} > receipt.txt

# The clamp scenario's points: where calc_coszen's float32 sum rounds
# past one on this scenario's own calendar (clamp_search.c).
"${python}" "${script_dir}/make_columns.py" "${build_dir}" --clamp-calendar
gcc -O0 -ffp-contract=off -fno-builtin -o clamp_search \
    "${script_dir}/clamp_search.c" -lm
./clamp_search clamp_cal.txt > clamp_hits.txt
echo "clamp_search hits: $(wc -l < clamp_hits.txt)" >> receipt.txt

"${python}" "${script_dir}/make_columns.py" "${build_dir}" | tee -a receipt.txt

# A correctly rounded asinf for the second reference build (see below).
cat > asinf_cr.c <<'CEOF'
#include <math.h>
/* asin rounded once from x86 80-bit long double: the correctly rounded
   float32 asin over the declination range (asinf_sweep.c checks it) */
float asinf(float x) { return (float)asinl((long double)x); }
CEOF
gcc -O2 -shared -fPIC -o asinf_cr.so asinf_cr.c -lm

# Reference of record: every libm call scalar, nothing contracted
# (baseline x86-64 has no FMA instruction).
f0=(-O0 -ffp-contract=off -cpp -ffree-form -ffree-line-length-none)
gfortran -c "${f0[@]}" -DEM_CORE=1 \
    "${source_root}/share/module_model_constants.F"
gfortran -c "${f0[@]}" -o swint_O0.o swint_wrf461.F90
gfortran -c "${f0[@]}" -o run_swint.o "${script_dir}/run_swint.F90"
gfortran -o run_swint_O0 module_model_constants.o swint_O0.o run_swint.o
if nm -u swint_O0.o | grep -q '_ZGV'; then
    echo "-O0 object pulled in a libmvec vector symbol" >&2; exit 4
fi
if objdump -d swint_O0.o | grep -Eq 'vfn?m(add|sub)'; then
    echo "-O0 object contains an FMA instruction" >&2; exit 4
fi

# Evidence only: WRF's stock GNU flags (arch/configure.defaults).  The
# driver program stays -O0 and native-endian: only the routines change.
gfortran -c -O2 -ftree-vectorize -funroll-loops -w -ffree-form \
    -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 \
    -o swint_O2vec.o swint_wrf461.F90
gfortran -o run_swint_O2vec module_model_constants.o swint_O2vec.o run_swint.o
{
    echo "undefined symbols, -O0 routines:"
    nm -u swint_O0.o | sed 's/^ *//' | sort | tr '\n' ' '; echo
    echo "undefined symbols, stock-flag routines:"
    nm -u swint_O2vec.o | sed 's/^ *//' | sort | tr '\n' ' '; echo
    echo "FMA instructions in stock-flag routines:" \
        "$(objdump -d swint_O2vec.o | grep -Ec 'vfn?m(add|sub)' || true)"
} >> receipt.txt

# Three runs per scenario:
#   O0     reference of record: this box's C library (glibc 2.39) as is
#   O0cr   the same binary with a correctly rounded asinf preloaded: the
#          C library's float32 asinf misrounds by one ULP on 0.11% of the
#          declination range (asinf_sweep.c), and C libraries from glibc
#          2.41 on return the correctly rounded word
#   O2vec  evidence only: WRF's stock GNU flags (libmvec sinf/cosf)
for scen in $("${python}" -c "import numpy as n; print(' '.join(n.load('columns.npz')['scenarios']))"); do
    ./run_swint_O0 "${scen}.in.bin" "${scen}.O0.out.bin"
    LD_PRELOAD="${build_dir}/asinf_cr.so" ./run_swint_O0 \
        "${scen}.in.bin" "${scen}.O0cr.out.bin"
    ./run_swint_O2vec "${scen}.in.bin" "${scen}.O2vec.out.bin"
done
sha256sum ./*.out.bin columns.npz >> receipt.txt
cat receipt.txt
