#!/usr/bin/env bash
# Fetch the ccpp-physics files SOURCES.sha256 pins into DEST, laid out as in
# the ccpp-physics tree, and refuse any file whose sha256 differs.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "usage: fetch_sources.sh DEST" >&2
    exit 2
fi
dest=$(realpath -m "$1")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
base="https://raw.githubusercontent.com/ufs-community/ccpp-physics/edd85a55502179611fdb7b902f0639c305cd1c3b"
mkdir -p "${dest}"
grep -v '^#' "${script_dir}/SOURCES.sha256" | while read -r sum path; do
    [[ -z "${sum}" ]] && continue
    mkdir -p "${dest}/$(dirname "${path}")"
    [[ -f "${dest}/${path}" ]] || curl -sfL -o "${dest}/${path}" "${base}/${path}"
done
cd "${dest}"
grep -v '^#' "${script_dir}/SOURCES.sha256" | sha256sum -c --quiet -
echo "ccpp-physics edd85a55 sources verified in ${dest}"
