#!/usr/bin/env bash
# Build the regime oracle on the platform the urban fixtures use: Ubuntu
# 26.04, GNU Fortran 15.2.0, glibc 2.43 (tests/data/oracles/urban/bem,
# ucm461, slucm_regime461).  Needs docker; the compiler comes from the
# public Ubuntu archive inside the container.
#
#   bash tools/urban_bem_regime_oracle/build_u2604.sh WORK_DIR
#
# WORK_DIR must hold wrf461-src/ (fetch_sources.sh).  Writes
# WORK_DIR/build-u2604 and WORK_DIR/out-u2604 (the fixtures).
set -euo pipefail
work=$(realpath "$1")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$(realpath "$here/..")" != "$work/tools" ]]; then
    mkdir -p "$work/tools/urban_wrf471_oracle"
    cp -r "$here" "$work/tools/"
    cp "$here/../urban_wrf471_oracle/stub_wrf.F90" "$work/tools/urban_wrf471_oracle/"
fi
docker run --rm -v "$work:/o" -w /o ubuntu:26.04 bash -c '
set -e
apt-get update -qq >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gfortran gcc binutils >/dev/null 2>&1
bash tools/urban_bem_regime_oracle/build.sh wrf461-src build-u2604 out-u2604 > build-u2604.log 2>&1 || { tail -30 build-u2604.log; exit 1; }
chown -R '"$(id -u):$(id -g)"' build-u2604 out-u2604 build-u2604.log'
tail -20 "$work/build-u2604.log"
