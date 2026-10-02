#!/usr/bin/env bash
# Fetch the WRF v4.7.1 files SOURCES.sha256 pins into DEST, laid out as in
# the WRF tree, and refuse any file whose sha256 differs from its pin.  The
# phys/physics_mmm files come from NCAR/MMM-physics at the tag WRF v4.7.1's
# arch/Externals.cfg names; everything else from wrf-model/WRF at v4.7.1.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "usage: fetch_sources.sh DEST" >&2
    exit 2
fi
dest=$(realpath -m "$1")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
wrf="https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1"
mmm="https://raw.githubusercontent.com/NCAR/MMM-physics/20240626-MPASv8.2"
mkdir -p "${dest}"
grep -v '^#' "${script_dir}/SOURCES.sha256" | while read -r sum path; do
    [[ -z "${sum}" ]] && continue
    mkdir -p "${dest}/$(dirname "${path}")"
    [[ -f "${dest}/${path}" ]] && continue
    case "${path}" in
        phys/physics_mmm/*) url="${mmm}/${path#phys/physics_mmm/}" ;;
        *) url="${wrf}/${path}" ;;
    esac
    curl -sfL -o "${dest}/${path}" "${url}"
done
cd "${dest}"
grep -v '^#' "${script_dir}/SOURCES.sha256" | sha256sum -c --quiet -
echo "WRF v4.7.1 terrain-drag sources verified in ${dest}"
