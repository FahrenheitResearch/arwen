#!/usr/bin/env bash
# Build and run the Dudhia shortwave (ra_sw_physics=1) column oracle from
# byte-unmodified WRF v4.6.1.
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR [make_columns.py options]
#
# Options after BUILD_DIR go to make_columns.py, e.g. ``--stress 20000`` for
# the randomised stress set (written to its own BUILD_DIR).
#
# Writes into BUILD_DIR:
#   dudhia-inputs.bin / .npz   the columns (make_columns.py)
#   dudhia-wrf-O0.bin          reference of record: -O0, scalar libm, no FMA
#   dudhia-wrf-O2vec.bin       evidence only: WRF's stock physics flags
#   libm-report.txt, oracle-sha256sums.txt, compiler.txt, wrf-commit.txt
#   libm_sweep.so              the C library's own log10f/powf, for
#                              sweep_libm.py's exhaustive comparison
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [make_columns options]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
shift 2
column_args=("$@")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
constants_source="${source_root}/share/module_model_constants.F"
sw_source="${source_root}/phys/module_ra_sw.F"
test -f "${constants_source}"
test -f "${sw_source}"

# The reference is the pinned source, unmodified.
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- \
        share/module_model_constants.F phys/module_ra_sw.F; then
    echo "a compiled WRF Dudhia dependency differs from pinned HEAD" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

"${python}" "${script_dir}/make_columns.py" "${build_dir}" "${column_args[@]}"

# Reference of record.  -O0 keeps every libm call scalar (log10f/powf), and
# baseline x86-64 has no FMA instruction, so nothing is contracted.
fflags=(-O0 -ffp-contract=off -cpp -ffree-form -ffree-line-length-none)
gfortran -c "${fflags[@]}" -DEM_CORE=1 "${constants_source}"
gfortran -c "${fflags[@]}" -o module_ra_sw_O0.o "${sw_source}"
gfortran -c "${fflags[@]}" -I "${build_dir}" "${script_dir}/run_swrad.F90"
gfortran -o run_swrad_O0 module_model_constants.o module_ra_sw_O0.o \
    run_swrad.o

nm -u module_ra_sw_O0.o | sed 's/^ *//' | sort > undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 Dudhia object pulled in a libmvec vector symbol" >&2
    exit 4
fi
if objdump -d module_ra_sw_O0.o | grep -Eq 'vfn?m(add|sub)'; then
    echo "-O0 Dudhia object contains an FMA instruction" >&2
    exit 4
fi

# Evidence only: the stock WRF GNU physics flags (configure.defaults), which
# the shipped wrf.exe uses.  Never the reference; reported beside it.
gfortran -c -O2 -ftree-vectorize -funroll-loops -w -ffree-form \
    -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 \
    -o module_ra_sw_O2vec.o "${sw_source}"
gfortran -o run_swrad_O2vec module_model_constants.o \
    module_ra_sw_O2vec.o run_swrad.o
nm -u module_ra_sw_O2vec.o | sed 's/^ *//' | sort > undefined-O2vec.txt

# The C library's own float32 log10f and powf, for the exhaustive sweep.
gcc -O2 -fPIC -shared -fopenmp -ffp-contract=off \
    -o libm_sweep.so "${script_dir}/libm_sweep.c" -lm

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 Dudhia object (reference of record), libm imports:"
    grep -E 'log|pow|exp|sqrt|_ZGV' undefined-O0.txt || true
    echo "# -O2 -ftree-vectorize -funroll-loops (stock WRF physics flags):"
    grep -E 'log|pow|exp|sqrt|_ZGV' undefined-O2vec.txt || true
    echo "# FMA instructions in the -O2vec object: $(objdump -d \
        module_ra_sw_O2vec.o | grep -Ec 'vfn?m(add|sub)' || true)"
} > libm-report.txt
cat libm-report.txt

./run_swrad_O0 dudhia-inputs.bin dudhia-wrf-O0.bin
./run_swrad_O2vec dudhia-inputs.bin dudhia-wrf-O2vec.bin
if cmp -s dudhia-wrf-O0.bin dudhia-wrf-O2vec.bin; then
    echo "stock-flags build: byte-identical to the -O0 reference" \
        | tee -a libm-report.txt
else
    echo "stock-flags build: DIFFERS from the -O0 reference" \
        | tee -a libm-report.txt
fi

sha256sum "${constants_source}" "${sw_source}" \
    "${script_dir}/make_columns.py" "${script_dir}/run_swrad.F90" \
    dudhia-inputs.bin dudhia-inputs.npz dudhia-wrf-O0.bin \
    dudhia-wrf-O2vec.bin > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "${pinned_commit}" > wrf-commit.txt
echo "Dudhia SW oracle built in ${build_dir}"
