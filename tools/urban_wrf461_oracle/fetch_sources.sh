#!/usr/bin/env bash
# Fetch byte-unmodified WRF v4.6.1 sources at the peeled release commit.
set -euo pipefail
if [[ $# != 1 ]]; then
    echo "usage: fetch_sources.sh SOURCE_DIR" >&2
    exit 2
fi
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src=$(realpath -m "$1")
commit=d66e442fccc04111067e29274c9f9eaccc3cef28
mkdir -p "$src"
for file in phys/module_sf_urban.F share/module_model_constants.F \
    frame/module_wrf_error.F run/URBPARM.TBL run/URBPARM_LCZ.TBL; do
    curl --fail --location --retry 2 --silent --show-error \
        "https://raw.githubusercontent.com/wrf-model/WRF/$commit/$file" \
        -o "$src/${file##*/}"
done
cp "$here/SOURCES.sha256" "$src/SOURCES.sha256"
(cd "$src" && sha256sum -c SOURCES.sha256)
printf '%s\n' "$commit" > "$src/COMMIT.txt"
