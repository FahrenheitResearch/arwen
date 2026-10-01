#!/usr/bin/env bash
# Build the WPS terrain-smoother oracle against the byte-unmodified WPS
# v4.6.0 geogrid/src/smooth_module.F and geogrid/src/parallel_module.F.
#
# WPS v4.6.0 is the WPS release paired with WRF v4.7.x (there is no WPS 4.7).
# Tag v4.6.0 is an annotated tag; its commit is pinned below and checked, and
# the two sources must not differ from it.
#
# Flags are WPS's own for GNU/Linux (configure.defaults, the gfortran
# stanzas): FFLAGS = -ffree-form -O -fconvert=big-endian -frecord-marker=4,
# CPP = cpp -P -traditional with -D_UNDERSCORE -DBYTESWAP -DLINUX -DIO_NETCDF
# -DBIT32.  -D_MPI is left out: one process, where exchange_halo_r is a
# no-op in the MPI build as well.  No -r8 (geogrid computes in default REAL,
# single precision) and no -march (so no fused multiply-add).
#
# The script fails if the objects contain an FMA instruction, since a
# contracted multiply-add would move the reference a bit-parity test is
# measured against.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WPS_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="335c76a111f84503e8b963abaf273ea8053645bb"

sources=(
    geogrid/src/parallel_module.F
    geogrid/src/smooth_module.F
)
for rel in "${sources[@]}"; do
    test -f "${source_root}/${rel}"
done

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WPS tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${sources[@]}"; then
    echo "a WPS smoother source differs from the pinned commit" >&2
    git -C "${source_root}" diff --name-only HEAD -- "${sources[@]}" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

cppflags=(-P -traditional -D_UNDERSCORE -DBYTESWAP -DLINUX -DIO_NETCDF -DBIT32)
fflags=(-ffree-form -O -fconvert=big-endian -frecord-marker=4)

for rel in "${sources[@]}"; do
    name=$(basename "${rel}" .F)
    cpp "${cppflags[@]}" "${source_root}/${rel}" > "${name}.f90"
    gfortran "${fflags[@]}" -c "${name}.f90" -o "${name}.o"
done
gfortran "${fflags[@]}" -c "${script_dir}/driver.F90" -o driver.o
gfortran -o wps_smooth_driver driver.o smooth_module.o parallel_module.o

if objdump -d smooth_module.o | grep -Eq '\bvfn?m(add|sub)'; then
    echo "smooth_module.o contains fused multiply-add instructions" >&2
    exit 4
fi

{
    echo "wps_commit=${head_commit}"
    for rel in "${sources[@]}"; do
        echo "sha256 $(sha256sum "${source_root}/${rel}" | cut -d' ' -f1) ${rel}"
    done
    echo "cpp=$(cpp --version | head -n 1)"
    echo "gfortran=$(gfortran --version | head -n 1)"
    echo "fflags=${fflags[*]}"
    echo "cppflags=${cppflags[*]}"
} > oracle-provenance.txt
cat oracle-provenance.txt
