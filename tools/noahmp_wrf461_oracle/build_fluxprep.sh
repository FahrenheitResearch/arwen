#!/usr/bin/env bash
# Build the SFCDIF1 / RAGRB / STOMATA oracle against a visibility-patched
# SCRATCH COPY of the pinned WRF v4.6.1 tree.
#
# This is a sibling of build_leaves.sh with the same fail-closed order and the
# same digests; it exists as a separate file only so that parallel lanes never
# edit one script.  The pinned tree is never written to.  Only
# phys/module_sf_noahmplsm.F is copied and patched; every other source is read
# straight out of the pinned tree, byte for byte.
#
# Order of operations, all fail-closed:
#   1. verify the pristine module SHA-256;
#   2. verify the patch file SHA-256;
#   3. apply the patch to the scratch copy and verify the patched SHA-256;
#   4. run check_visibility_patch.py, which re-derives the substitution and
#      refuses anything that is not `private ::` -> `public ::`;
#   5. compile and run;
#   6. validate the fixture (structure, live/dead flags, discrimination,
#      branch coverage).
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build_fluxprep.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
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
        echo "build_fluxprep.sh: ${label} sha256 mismatch" >&2
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
    -I "${build_dir}" "${script_dir}/run_fluxprep.F90"

gfortran -o run_fluxprep stub_wrf.o module_model_constants.o \
    module_sf_gecros.o module_sf_noahmplsm.o run_fluxprep.o

./run_fluxprep noahmp-fluxprep.csv noahmp-fluxprep-discrimination.csv

python3 "${script_dir}/validate_fluxprep_oracle.py" \
    noahmp-fluxprep.csv noahmp-fluxprep-discrimination.csv

sha256sum "${constants_source}" "${gecros_source}" "${lsm_source}" \
    "${patch_file}" "${patched_source}" \
    "${script_dir}/stub_wrf.F90" "${script_dir}/run_fluxprep.F90" \
    noahmp-fluxprep.csv noahmp-fluxprep-discrimination.csv \
    > fluxprep-oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
cat fluxprep-oracle-sha256sums.txt compiler.txt
