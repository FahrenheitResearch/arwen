#!/usr/bin/env bash
# Fetch the byte-unmodified WRF v4.6.1 sources the BEP+BEM regime oracle
# compiles (release tag v4.6.1, peeled commit d66e442f...), plus v4.7.1's
# module_sf_bep_bem.F (tag v4.7.1, commit f52c197e...) for the control build.
#
#   bash tools/urban_bem_regime_oracle/fetch_sources.sh SOURCE_DIR
#
# Every file is checked against SOURCES.sha256 / SOURCES-v471-control.sha256.
set -euo pipefail
if [[ $# != 1 ]]; then
    echo "usage: fetch_sources.sh SOURCE_DIR" >&2
    exit 2
fi
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src=$(realpath -m "$1")
v461=d66e442fccc04111067e29274c9f9eaccc3cef28
v471=f52c197ed39d12e087d02c50f412d90d418f6186
get() {  # commit rel dest
    mkdir -p "$(dirname "$3")"
    curl --fail --location --retry 3 --silent --show-error \
        "https://raw.githubusercontent.com/wrf-model/WRF/$1/$2" -o "$3"
}
while read -r _ rel; do
    get "$v461" "$rel" "$src/$rel"
done < "$here/SOURCES.sha256"
get "$v471" phys/module_sf_bep_bem.F "$src/v471/phys/module_sf_bep_bem.F"
(cd "$src" && sha256sum -c "$here/SOURCES.sha256" "$here/SOURCES-v471-control.sha256")
printf 'v4.6.1 %s\nv4.7.1 %s (module_sf_bep_bem.F only)\n' "$v461" "$v471" > "$src/COMMIT.txt"
