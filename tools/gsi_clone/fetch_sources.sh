#!/usr/bin/env bash
# Fetch every pinned source the GSI clone container builds from into ./src,
# then write SOURCES.sha256 beside it. The Dockerfile verifies these hashes
# before it builds anything, so a moved or re-rolled upstream tarball stops
# the build instead of silently changing the libraries.
#
# Versions come from NOAA-EMC/HRRR tag v4.1.21 versions/build.ver (the
# WCOSS2 module set the operational executables were built against).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dest="${1:-$here/src}"
mkdir -p "$dest"
cd "$dest"

fetch() {  # fetch <output name> <url>
  local out="$1" url="$2"
  if [[ -s "$out" ]]; then echo "have $out"; return 0; fi
  echo "get  $out <- $url"
  curl -fsSL --retry 4 --retry-delay 5 -o "$out.part" "$url"
  mv "$out.part" "$out"
}

gh() {  # gh <org/repo> <tag> -> <repo>-<tag>.tar.gz
  local repo="$1" tag="$2" name
  name="$(basename "$repo")-${tag}.tar.gz"
  fetch "$name" "https://github.com/${repo}/archive/refs/tags/${tag}.tar.gz"
}

# Third-party libraries (build.ver: zlib 1.2.11, libpng 1.6.37, libjpeg 9c,
# jasper 2.0.25, hdf5 1.10.6, netcdf 4.7.4, pnetcdf 1.12.2). netcdf-fortran 4.5.3 is the
# Fortran layer WCOSS2 ships inside its netcdf/4.7.4 module. netcdf-c 4.7.4
# has no release tarball left on Unidata's server, so it comes from the tag
# archive and builds with CMake (the tag carries no generated configure).
fetch zlib-1.2.11.tar.gz        https://zlib.net/fossils/zlib-1.2.11.tar.gz
fetch libpng-1.6.37.tar.gz      https://download.sourceforge.net/libpng/libpng-1.6.37.tar.gz
fetch jpegsrc.v9c.tar.gz        https://www.ijg.org/files/jpegsrc.v9c.tar.gz
gh    jasper-software/jasper    version-2.0.25
fetch hdf5-1.10.6.tar.gz        https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.gz
gh    Unidata/netcdf-c          v4.7.4
fetch netcdf-fortran-4.5.3.tar.gz https://downloads.unidata.ucar.edu/netcdf-fortran/4.5.3/netcdf-fortran-4.5.3.tar.gz
fetch pnetcdf-1.12.2.tar.gz     https://parallel-netcdf.github.io/Release/pnetcdf-1.12.2.tar.gz
fetch cmake-3.18.4-Linux-x86_64.tar.gz https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4-Linux-x86_64.tar.gz

# NCEPLIBS at the build.ver versions, only the ones the stand-alone tools
# (process_mosaic, process_cloud, process_lightning, ref2tten) link through
# the module variables their makefiles read. GSI and EnKF build their own
# bacio, bufr, crtm, ip, nemsio, sfcio, sigio, sp, w3emc and w3nco from
# hrrr_gsi.fd/libsrc: HRRR's cmake takes the GENERIC host branch on any host
# name it does not list (WCOSS2's included), and that branch turns
# BUILD_CORELIBS on.
gh    NOAA-EMC/NCEPLIBS-bacio   v2.4.1
gh    NOAA-EMC/NCEPLIBS-w3nco   v2.4.1
gh    NOAA-EMC/NCEPLIBS-bufr    bufr_v11.4.0
gh    NOAA-EMC/NCEPLIBS-g2      v3.4.5
gh    NOAA-EMC/NCEPLIBS-g2tmpl  v1.10.0
gh    NOAA-EMC/NCEPLIBS-wrf_io  v1.1.1

sha256sum ./*.tar.gz | sed 's# \./# #' | sort -k2 > "$here/SOURCES.sha256"
echo "wrote $here/SOURCES.sha256"

# Run-time tools the analysis scripts call, kept apart from ./src so adding
# one does not rebuild the libraries. wgrib2 2.0.7 is the tag's
# versions/run.ver wgrib2_ver; the makeguess script repacks the RAP file with
# it before ungrib.
mkdir -p "$here/src-run"
cd "$here/src-run"
fetch wgrib2.tgz.v2.0.7 https://ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/wgrib2.tgz.v2.0.7
sha256sum ./wgrib2.tgz.v2.0.7 | sed 's# \./# #' > "$here/SOURCES-run.sha256"
echo "wrote $here/SOURCES-run.sha256"
