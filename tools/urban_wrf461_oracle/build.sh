#!/usr/bin/env bash
# The same Fortran argument driver calls WRF v4.6.1's original urban routine.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh SOURCE_DIR BUILD_DIR [OUT_DIR]" >&2
    exit 2
fi
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src=$(realpath "$1")
(cd "$src" && sha256sum -c "$here/SOURCES.sha256")
exec bash "$here/../urban_wrf471_oracle/build_ucm.sh" "$@"
