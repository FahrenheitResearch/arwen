#!/usr/bin/env bash
# Build and run the WRF v4.6.1 Ferrier-Aligo (mp_physics=5) column oracle
# from byte-unmodified WRF source.  CPU only.
#
# usage: build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# Writes into BUILD_DIR:
#   columns_inputs.csv, columns.in   (make_columns.py)
#   O0/levels.csv, O0/surface.csv     the reference (-O0, scalar libm)
#   O2vec/levels.csv, O2vec/surface.csv  the same source at WRF's own GNU
#                                        flags (-O2 -ftree-vectorize
#                                        -funroll-loops), evidence only
#   SHA256SUMS
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
fer_source="${source_root}/phys/module_mp_fer_hires.F"
table="${source_root}/run/ETAMPNEW_DATA.expanded_rain"
test -f "${fer_source}"
test -f "${table}"

# The reference is the pinned source, not merely a file with the expected
# basename: refuse another revision and any local edit to what is compiled
# or read.
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- \
        phys/module_mp_fer_hires.F run/ETAMPNEW_DATA.expanded_rain; then
    echo "module_mp_fer_hires.F or its table differs from pinned HEAD" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

python3 "${script_dir}/make_columns.py" "${build_dir}"

build_one() {
    local tag=$1; shift
    local flags=("$@")
    mkdir -p "${tag}"
    (
        cd "${tag}"
        gfortran -c "${flags[@]}" -cpp -ffree-form -ffree-line-length-none \
            -DRWORDSIZE=4 -DIWORDSIZE=4 -fallow-argument-mismatch \
            "${script_dir}/stub_wrf.F90"
        gfortran -c "${flags[@]}" -cpp -ffree-form -ffree-line-length-none \
            -DRWORDSIZE=4 -DIWORDSIZE=4 -fallow-argument-mismatch \
            -o module_mp_fer_hires.o "${fer_source}"
        # -fconvert only takes effect in the main program's unit; WRF reads
        # ETAMPNEW_DATA.expanded_rain big-endian.
        gfortran -c "${flags[@]}" -cpp -ffree-form -ffree-line-length-none \
            -fconvert=big-endian "${script_dir}/run_ferrier.F90"
        gfortran -o run_ferrier stub_wrf.o module_mp_fer_hires.o \
            run_ferrier.o
        nm -u module_mp_fer_hires.o | sed 's/^ *//' | sort \
            > undefined-symbols.txt
        ln -sf "${table}" ETAMPNEW_DATA.expanded_rain
        ln -sf ../columns.in columns.in
        ./run_ferrier > run.stdout 2> run.stderr
    )
}

build_one O0 -O0
# The reference object must stay on scalar libm.
if grep -q '_ZGV' O0/undefined-symbols.txt; then
    echo "-O0 Ferrier object pulled in a libmvec vector symbol" >&2
    exit 4
fi
build_one O2vec -O2 -ftree-vectorize -funroll-loops

{
    echo "wrf_commit ${pinned_commit}"
    echo "gfortran $(gfortran -dumpfullversion)"
    echo "glibc $(ldd --version | head -1 | awk '{print $NF}')"
    sha256sum "${fer_source}" "${table}" | sed "s#${source_root}/##"
    sha256sum columns_inputs.csv O0/levels.csv O0/surface.csv \
        O2vec/levels.csv O2vec/surface.csv
} > SHA256SUMS
cat SHA256SUMS
python3 "${script_dir}/compare_ferrier.py" O0 O2vec || true
