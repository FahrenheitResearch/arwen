#!/usr/bin/env bash
# Build the two NOAA oracles for gpuwm/da/hydrometeor_analysis.py.
#
# NOAA's files are fetched at tag v4.1.21 into <work>/noaa (never committed
# here), checked against the sha256 below, and compiled UNCHANGED.  The
# precipitation block and the clamp are cut out of gsdcloudanalysis.F90 by
# line number into two include files.
#
# Usage: bash build.sh <work-dir>      (gfortran on PATH; CPU only)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=${1:?work directory}
mkdir -p "$work/noaa"
base=https://raw.githubusercontent.com/NOAA-EMC/HRRR/v4.1.21
fetch() {  # path sha256
  local name
  name=$(basename "$1")
  [ -s "$work/noaa/$name" ] || curl -sS -f -o "$work/noaa/$name" "$base/$1"
  echo "$2  $work/noaa/$name" | sha256sum -c - > /dev/null \
    || { echo "sha256 mismatch for $name" >&2; exit 2; }
}
gsd=sorc/hrrr_gsi.fd/libsrc/GSD/gsdcloud
fetch $gsd/hydro_mxr_thompson.f90 d282cbcb7e197d4809a9425615f9b7e5f8769a61ff965aa753ae29cf3879ad74
fetch $gsd/kinds.f90 f7a6d354d40092a2be6e039bd293fea981abe4e4da26931bf656b8391cec3b74
fetch $gsd/constants.f90 63117995601bfb6aec192d99a06dc5eec665fa3cb46610988f77dc162291b97a
fetch $gsd/PrecipMxr_radar.f90 b3eee7b4bedabdf28732152229091fb709420330726b2daa4ef6728b87da381e
fetch sorc/hrrr_gsi.fd/src/gsi/gsdcloudanalysis.F90 88d3fdbc41ed4a6df4825dd372a3ec6e75dde310bfa20e90621b4bf894556ead
cd "$work"
sed -n '872,1049p' noaa/gsdcloudanalysis.F90 > noaa_gsdcloudanalysis_0872_1049.inc
sed -n '1053,1066p' noaa/gsdcloudanalysis.F90 > noaa_gsdcloudanalysis_1053_1066.inc
# -fno-tree-vectorize: at -O2 gfortran 15 turns the theta-to-temperature loop
# of PrecipMxr_radar.f90 into glibc's two-wide vector pow (_ZGVbN2vv_pow),
# which is not the scalar pow; the oracle is the scalar IEEE evaluation.
FFLAGS=${FFLAGS:--O2 -ffp-contract=off -fno-tree-vectorize}
{
  gfortran --version | head -1
  echo "FFLAGS: $FFLAGS"
  ldd --version | head -1
  sha256sum noaa/*.f90 noaa/*.F90 noaa_gsdcloudanalysis_*.inc
} > build-info.txt
gfortran $FFLAGS -c noaa/kinds.f90 -o kinds.o
gfortran $FFLAGS -c noaa/constants.f90 -o constants.o
gfortran $FFLAGS -c noaa/hydro_mxr_thompson.f90 -o hydro_mxr_thompson.o
gfortran $FFLAGS -c noaa/PrecipMxr_radar.f90 -o PrecipMxr_radar.o
gfortran $FFLAGS -c "$here/oracle_a.f90" -o oracle_a.o
gfortran $FFLAGS -I. -c "$here/oracle_b.f90" -o oracle_b.o
gfortran $FFLAGS oracle_a.o hydro_mxr_thompson.o kinds.o -o oracle_a
gfortran $FFLAGS oracle_b.o PrecipMxr_radar.o hydro_mxr_thompson.o constants.o kinds.o -o oracle_b
cat build-info.txt
