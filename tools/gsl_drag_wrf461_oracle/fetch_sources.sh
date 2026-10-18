#!/usr/bin/env bash
# Fetch the WRF v4.6.1 files SOURCES.sha256 pins into DEST, laid out as in
# the WRF tree, from wrf-model/WRF at the v4.6.1 commit, and refuse any file
# whose sha256 differs from its pin.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "usage: fetch_sources.sh DEST" >&2
    exit 2
fi
dest=$(realpath -m "$1")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
wrf="https://raw.githubusercontent.com/wrf-model/WRF/d66e442fccc04111067e29274c9f9eaccc3cef28"
mkdir -p "${dest}"
grep -v '^#' "${script_dir}/SOURCES.sha256" | while read -r sum path; do
    [[ -z "${sum}" ]] && continue
    mkdir -p "${dest}/$(dirname "${path}")"
    [[ -f "${dest}/${path}" ]] && continue
    curl -sfL -o "${dest}/${path}" "${wrf}/${path}"
done
cd "${dest}"
grep -v '^#' "${script_dir}/SOURCES.sha256" | sha256sum -c --quiet -
echo "WRF v4.6.1 GSL-drag sources verified in ${dest}"
