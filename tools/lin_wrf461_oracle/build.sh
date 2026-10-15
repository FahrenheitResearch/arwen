#!/usr/bin/env bash
# Build and run the Purdue Lin (mp_physics = 2) column oracle from
# byte-unmodified WRF v4.6.1, then pack its fixture.
#
#   tools/lin_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR [FIXTURE]
#
# FIXTURE defaults to tests/data/oracles/lin/lin-wrf461-columns.npz in this
# tree.  The reference is compiled at -O0 (scalar libm, no vector variants,
# no contraction); the build refuses a wrong WRF revision or a locally edited
# compiled WRF file, and refuses a -O0 object that pulled in a libmvec symbol.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [FIXTURE]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
fixture=${3:-${repo_root}/tests/data/oracles/lin/lin-wrf461-columns.npz}

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
constants_source="${source_root}/share/module_model_constants.F"
radar_source="${source_root}/phys/module_mp_radar.F"
lin_source="${source_root}/phys/module_mp_lin.F"
for source in "${constants_source}" "${radar_source}" "${lin_source}"; do
    test -f "${source}"
done

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- \
        share/module_model_constants.F phys/module_mp_radar.F \
        phys/module_mp_lin.F; then
    echo "a compiled WRF Lin dependency differs from pinned HEAD" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

fflags=(-O0 -cpp -ffree-form -ffree-line-length-none)
gfortran -c "${fflags[@]}" "${script_dir}/stub_wrf.F90"
gfortran -c "${fflags[@]}" "${constants_source}"
gfortran -c "${fflags[@]}" -I "${build_dir}" "${radar_source}"
gfortran -c "${fflags[@]}" -I "${build_dir}" -o module_mp_lin_O0.o \
    "${lin_source}" 2> lin-compile-warnings.txt
gfortran -c "${fflags[@]}" -I "${build_dir}" "${script_dir}/run_lin.F90"
gfortran -o run_lin stub_wrf.o module_model_constants.o module_mp_radar.o \
    module_mp_lin_O0.o run_lin.o

nm -u module_mp_lin_O0.o | sed 's/^ *//' | sort > undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 Lin object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-O0.txt >&2
    exit 4
fi
{
    echo "# scalar math symbols the -O0 module_mp_lin object calls"
    grep -E ' (expf|powf|sqrtf|logf|exp|pow|log10|__powisf2)$' \
        undefined-O0.txt || true
} > libm-report.txt

"${PYTHON:-python3}" "${script_dir}/make_lin_inputs.py" "${build_dir}"
./run_lin lin-inputs.bin lin-outputs.bin

gfortran --version | sed -n '1p' > compiler.txt
ldd --version | sed -n '1p' > glibc.txt
echo "${pinned_commit}" > wrf-commit.txt
sha256sum "${constants_source}" "${radar_source}" "${lin_source}" \
    "${script_dir}/stub_wrf.F90" "${script_dir}/run_lin.F90" \
    "${script_dir}/make_lin_inputs.py" lin-inputs.bin lin-outputs.bin \
    > oracle-sha256sums.txt
"${PYTHON:-python3}" "${script_dir}/pack_lin_oracle.py" "${build_dir}" "${fixture}"
echo "Lin oracle built in ${build_dir}"
