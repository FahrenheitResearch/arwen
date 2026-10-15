#!/usr/bin/env bash
# Build the regime oracle on the reference platform the SLUCM kernel targets:
# Ubuntu 26.04, GNU Fortran 15.2.0, glibc 2.43 (the platform of the existing
# tests/data/oracles/urban/ucm and ucm461 fixtures).  Needs docker; the
# compiler comes from the public Ubuntu archive inside the container.
#
#   bash tools/urban_slucm_regime_oracle/build_u2604.sh WORK_DIR
#
# WORK_DIR must hold wrf461-src/ (tools/urban_wrf461_oracle/fetch_sources.sh)
# and wrf471-src/ (the same five files with phys/module_sf_urban.F of tag
# v4.7.1, sha256 623868c7..., and tools/urban_wrf471_oracle/SOURCES.sha256).
# Writes WORK_DIR/out-<v>-u2604 (the fixtures) and out-<v>-u2604-tlocctl
# (the tloc attribution control, never a fixture) for v = 461 and 471.
set -euo pipefail
work=$(realpath "$1")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$work/tools"
if [[ "$(realpath "$here/..")" != "$work/tools" ]]; then
    cp -r "$here" "$work/tools/"
    cp -r "$here/../urban_wrf471_oracle" "$work/tools/"
fi
docker run --rm -v "$work:/o" -w /o ubuntu:26.04 bash -c '
set -e
apt-get update -qq >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gfortran gcc binutils >/dev/null 2>&1
gfortran --version | sed -n 1p
ldd --version | sed -n 1p
for v in 461 471; do
  bash tools/urban_slucm_regime_oracle/build.sh wrf$v-src build-$v-u2604 out-$v-u2604 > build-$v-u2604.log 2>&1
  bash tools/urban_slucm_regime_oracle/tloc_control.sh wrf$v-src tlocctl-$v-u2604 out-$v-u2604-tlocctl > tlocctl-$v-u2604.log 2>&1
done'
