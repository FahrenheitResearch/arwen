#!/usr/bin/env bash
# Fetch the WRF v4.7.1 files SOURCES.sha256 pins into DEST, laid out as in the WRF tree, and refuse any file whose
# sha256 differs from its pin, so the fixture of record can never come from an edited file.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "usage: fetch_sources.sh DEST" >&2
    exit 2
fi
dest=$(realpath -m "$1")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
base="https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1"
mkdir -p "${dest}"
grep -v '^#' "${here}/SOURCES.sha256" | while read -r sum path; do
    [[ -z "${sum}" ]] && continue
    mkdir -p "${dest}/$(dirname "${path}")"
    if [[ ! -f "${dest}/${path}" ]]; then
        curl -sfL -o "${dest}/${path}" "${base}/${path}"
    fi
done
cd "${dest}"
grep -v '^#' "${here}/SOURCES.sha256" | sha256sum -c --quiet -
echo "WRF v4.7.1 sources verified in ${dest}"
