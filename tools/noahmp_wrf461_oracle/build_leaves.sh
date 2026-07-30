#!/usr/bin/env bash
# Build the per-leaf Noah-MP oracle against a visibility-patched SCRATCH COPY
# of the pinned WRF v4.6.1 tree.
#
# The pinned tree is never written to.  Only phys/module_sf_noahmplsm.F is
# copied and patched; every other source is read straight out of the pinned
# tree, byte for byte.
#
# Order of operations, all fail-closed:
#   1. verify the pristine module SHA-256 (the digest already pinned in
#      gpuwm/core/noahmp_mynn_contract.py);
#   2. verify the patch file SHA-256;
#   3. apply the patch to the scratch copy and verify the patched SHA-256;
#   4. run check_visibility_patch.py, which re-derives the substitution and
#      refuses anything that is not `private ::` -> `public ::`;
#   5. compile and run;
#   6. validate the fixture (structure, live/dead flags, discrimination).
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build_leaves.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)

patch_file="${repo_root}/patches/noahmp-lsm-leaf-visibility.patch"
constants_source="${source_root}/share/module_model_constants.F"
gecros_source="${source_root}/phys/module_sf_gecros.F"
lsm_source="${source_root}/phys/module_sf_noahmplsm.F"

PRISTINE_SHA=bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282
PATCHED_SHA=3cd3690d6455cfb8549cb41979b7e101e7436464c478f7f7973ab226488ac206
PATCH_SHA=164885a9ce956a51be2ab189165da8b60217b636a720c994e04dfb862a05aad5

for f in "${constants_source}" "${gecros_source}" "${lsm_source}" \
         "${patch_file}"; do
    test -f "${f}"
done

check_sha() {
    local expect="$1" path="$2" label="$3" got
    got=$(sha256sum "${path}" | cut -d' ' -f1)
    if [[ "${got}" != "${expect}" ]]; then
        echo "build_leaves.sh: ${label} sha256 mismatch" >&2
        echo "  path     ${path}" >&2
        echo "  expected ${expect}" >&2
        echo "  actual   ${got}" >&2
        exit 3
    fi
}

check_sha "${PRISTINE_SHA}" "${lsm_source}" "pinned pristine module"
check_sha "${PATCH_SHA}" "${patch_file}" "visibility patch"

rm -rf "${build_dir}"
mkdir -p "${build_dir}/patched/phys"
cp "${lsm_source}" "${build_dir}/patched/phys/module_sf_noahmplsm.F"
( cd "${build_dir}/patched" \
  && patch -p1 --no-backup-if-mismatch < "${patch_file}" )
patched_source="${build_dir}/patched/phys/module_sf_noahmplsm.F"
check_sha "${PATCHED_SHA}" "${patched_source}" "visibility-patched module"

python3 "${script_dir}/check_visibility_patch.py" \
    "${lsm_source}" "${patched_source}" "${patch_file}"

cd "${build_dir}"

fflags=(-O0 -cpp -ffree-form -ffree-line-length-none)

gfortran -c "${fflags[@]}" "${script_dir}/stub_wrf.F90"
gfortran -c "${fflags[@]}" "${constants_source}"
gfortran -c "${fflags[@]}" "${gecros_source}"
gfortran -c "${fflags[@]}" -I "${build_dir}" "${patched_source}"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_leaves.F90"

gfortran -o run_leaves stub_wrf.o module_model_constants.o \
    module_sf_gecros.o module_sf_noahmplsm.o run_leaves.o

./run_leaves noahmp-leaves.csv noahmp-leaves-discrimination.csv

python3 "${script_dir}/validate_leaf_oracle.py" \
    noahmp-leaves.csv noahmp-leaves-discrimination.csv

sha256sum "${constants_source}" "${gecros_source}" "${lsm_source}" \
    "${patch_file}" "${patched_source}" \
    "${script_dir}/stub_wrf.F90" "${script_dir}/run_leaves.F90" \
    noahmp-leaves.csv noahmp-leaves-discrimination.csv \
    > leaf-oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
cat leaf-oracle-sha256sums.txt compiler.txt
