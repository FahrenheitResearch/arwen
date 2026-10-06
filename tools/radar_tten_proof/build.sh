#!/usr/bin/env bash
# Fetch NOAA's files at the pinned tag, check them against the recorded
# digests, and compile them UNCHANGED with the harnesses in this folder.
#
#   build.sh <workdir> [extra gfortran flags]
#
# Output: <workdir>/oracle_ref2tten, oracle_smooth, oracle_vinterp, and
# <workdir>/BUILD.txt (compiler version, flags, digests).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="${1:?workdir}"; shift || true
extra=("$@")
tag=v4.1.21
base="https://raw.githubusercontent.com/NOAA-EMC/HRRR/${tag}/sorc/hrrr_ref2tten.fd"
files=(kinds.f90 constants.f90 pbl_height.f90 build_missing_REFcone.f90 radar_ref2tten.f90 smooth.f90 vinterp_radar_ref.f90)
mkdir -p "$work/noaa"
for f in "${files[@]}"; do
  [ -s "$work/noaa/$f" ] || curl -sfL -o "$work/noaa/$f" "$base/$f"
done
( cd "$work/noaa" && sha256sum -c "$here/NOAA-SHA256SUMS" )
flags=(-O2 -ffp-contract=off -fno-fast-math "${extra[@]}")
cd "$work"
src=()
for f in kinds.f90 constants.f90 pbl_height.f90 build_missing_REFcone.f90 radar_ref2tten.f90 smooth.f90; do src+=("noaa/$f"); done
gfortran "${flags[@]}" -J "$work" -o oracle_ref2tten "${src[@]}" "$here/harness.f90"
# vinterp_radar_ref.f90:103 spells "stop(114)", legacy syntax NOAA's Intel
# compiler accepts and every gfortran here (12 to 15) rejects ("Blank
# required in STOP statement").  The line is the routine's refusal of an
# unknown mosaic level count, which no case reaches.  It is the ONE token
# changed in a build copy, "stop(114)" -> "stop 114"; the copy is checked to
# differ from NOAA's file at that line only, and both digests are recorded.
mkdir -p "$work/noaa-build"
sed '103s/stop(114)/stop 114/' noaa/vinterp_radar_ref.f90 > noaa-build/vinterp_radar_ref.f90
changed=$(diff noaa/vinterp_radar_ref.f90 noaa-build/vinterp_radar_ref.f90 | grep -c '^[<>]' || true)
[ "$changed" = "2" ] || { echo "vinterp build copy differs in more than one line" >&2; exit 3; }
gfortran "${flags[@]}" -J "$work" -o oracle_vinterp noaa/kinds.f90 noaa-build/vinterp_radar_ref.f90 "$here/harness_vinterp.f90"
gfortran "${flags[@]}" -J "$work" -o oracle_smooth noaa/kinds.f90 noaa/smooth.f90 "$here/harness_smooth.f90"
{
  echo "tag ${tag}"
  echo "compiler $(gfortran --version | head -1)"
  echo "vinterp_radar_ref.f90 built from a copy with :103 stop(114) -> stop 114:"
  diff noaa/vinterp_radar_ref.f90 noaa-build/vinterp_radar_ref.f90 || true
  ( sha256sum noaa-build/vinterp_radar_ref.f90 )
  echo "flags ${flags[*]}"
  echo "glibc $(ldd --version | head -1)"
  echo "cpu $(lscpu | sed -n 's/^Model name: *//p')"
  ( cd noaa && sha256sum "${files[@]}" )
} > BUILD.txt
cat BUILD.txt
