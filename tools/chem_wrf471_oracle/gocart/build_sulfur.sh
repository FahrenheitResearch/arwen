#!/usr/bin/env bash
# The shared build also runs aging. No shared source or stub is edited.
set -euo pipefail
src=$(realpath "$1")
build=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pin=f8c37bd236d2fcaeb9361e8bde5865066dd3487e4105ca4e0aaaea69dd8147b4
test "$(sha256sum "${src}/chem/chemics_init.F" | cut -d' ' -f1)" = "$pin"
mkdir -p "$build"
# Exact range, including its enclosing ENDIF and the inner ENDDO.
sed -n '1765,1791p' "${src}/chem/chemics_init.F" > "${build}/chemics_init_solar.inc"
cmp "${build}/chemics_init_solar.inc" "${here}/chemics_init_solar.inc"
exec bash "${here}/build.sh" "$@"
