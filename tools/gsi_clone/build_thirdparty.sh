#!/usr/bin/env bash
# Build the pinned third-party libraries the HRRR v4.1.21 tools link
# (versions from the tag's versions/build.ver). Runs inside the toolchain
# stage after env.sh.
set -euo pipefail
JOBS="${1:-8}"
S=/opt/src
B=/tmp/build-thirdparty
mkdir -p "$B" "$PREFIX"
cd "$B"

# Plain Intel compilers (no MPI), like the WCOSS2 serial hdf5/netcdf modules;
# pnetcdf is the one MPI library.
export CC=icc CXX=icpc FC=ifort F77=ifort F90=ifort
export CPPFLAGS="-I$PREFIX/include" LDFLAGS="-L$PREFIX/lib"

tar xzf "$S/zlib-1.2.11.tar.gz"
( cd zlib-1.2.11 && ./configure --prefix="$PREFIX" && make -j"$JOBS" && make install )

tar xzf "$S/libpng-1.6.37.tar.gz"
( cd libpng-1.6.37 && ./configure --prefix="$PREFIX" && make -j"$JOBS" && make install )

# libjpeg 9c (build.ver libjpeg_ver). Nothing here links it; CMake's
# FindJasper, which NCEPLIBS-g2 calls, refuses a jasper without a libjpeg
# beside it.
tar xzf "$S/jpegsrc.v9c.tar.gz"
( cd jpeg-9c && ./configure --prefix="$PREFIX" && make -j"$JOBS" && make install )

# jasper without its optional libjpeg codec: the tools' makefiles link
# JASPER_LIB, PNG_LIB and Z_LIB only, so a jasper that needed -ljpeg would
# not link.
tar xzf "$S/jasper-version-2.0.25.tar.gz"
cmake -S jasper-version-2.0.25 -B jasper-build -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_BUILD_TYPE=Release -DJAS_ENABLE_SHARED=OFF -DJAS_ENABLE_DOC=OFF \
  -DJAS_ENABLE_PROGRAMS=OFF -DJAS_ENABLE_LIBJPEG=OFF -DJAS_ENABLE_OPENGL=OFF \
  -DJAS_ENABLE_AUTOMATIC_DEPENDENCIES=OFF -DCMAKE_INSTALL_LIBDIR=lib
cmake --build jasper-build -j"$JOBS" && cmake --install jasper-build

# hdf5 with its Fortran layer: the tag's WRF configure file links
# -lhdf5_fortran and -lhdf5hl_fortran.
tar xzf "$S/hdf5-1.10.6.tar.gz"
( cd hdf5-1.10.6 && ./configure --prefix="$PREFIX" --with-zlib="$PREFIX" --enable-hl \
    --enable-fortran --disable-tests && make -j"$JOBS" && make install )

tar xzf "$S/netcdf-c-v4.7.4.tar.gz"
cmake -S netcdf-c-4.7.4 -B netcdf-c-build -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release -DENABLE_NETCDF_4=ON \
  -DENABLE_DAP=OFF -DENABLE_TESTS=OFF -DBUILD_UTILITIES=ON -DCMAKE_INSTALL_LIBDIR=lib \
  -DHDF5_C_LIBRARY="$PREFIX/lib/libhdf5.so" -DHDF5_HL_LIBRARY="$PREFIX/lib/libhdf5_hl.so" \
  -DHDF5_INCLUDE_DIR="$PREFIX/include"
cmake --build netcdf-c-build -j"$JOBS" && cmake --install netcdf-c-build

tar xzf "$S/netcdf-fortran-4.5.3.tar.gz"
( cd netcdf-fortran-4.5.3 && LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}" \
    ./configure --prefix="$PREFIX" && make -j"$JOBS" && make install )

tar xzf "$S/pnetcdf-1.12.2.tar.gz"
( cd pnetcdf-1.12.2 && env -u CC -u CXX -u FC -u F77 -u F90 ./configure --prefix="$PREFIX" \
    MPICC=mpiicc MPICXX=mpiicpc MPIF77=mpiifort MPIF90=mpiifort && make -j"$JOBS" && make install )

ls -l "$PREFIX/lib"
cd / && rm -rf "$B"
