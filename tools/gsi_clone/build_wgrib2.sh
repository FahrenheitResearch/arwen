#!/usr/bin/env bash
# Build wgrib2 2.0.7, the version the tag's versions/run.ver loads, from
# NOAA CPC's source tarball (it bundles its own jasper, png, zlib and
# g2clib). The makeguess script runs it once per RAP file to repack it to
# simple packing before WPS ungrib reads it. Built with the GNU compilers,
# the tarball's default.
set -euo pipefail
JOBS="${1:-8}"
B=/tmp/build-wgrib2
mkdir -p "$B" /opt/wgrib2/bin
cd "$B"
tar xzf /opt/src/wgrib2.tgz.v2.0.7
cd grib2
# The bundled build takes CC and FC from the environment; the Intel
# variables env.sh sets do not apply here.
env -u CC -u CXX -u FC -u F77 -u F90 CC=gcc FC=gfortran make > /opt/wgrib2/build.log 2>&1 \
  || { tail -60 /opt/wgrib2/build.log; exit 1; }
cp wgrib2/wgrib2 /opt/wgrib2/bin/wgrib2
# -version prints the version and exits with status 8 by design.
/opt/wgrib2/bin/wgrib2 -version > /opt/wgrib2/VERSION || true
grep -q '^v0\.2\.0\.7 ' /opt/wgrib2/VERSION || { cat /opt/wgrib2/VERSION; exit 1; }
gzip -f /opt/wgrib2/build.log
cd / && rm -rf "$B"
