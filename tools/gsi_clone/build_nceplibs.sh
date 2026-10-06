#!/usr/bin/env bash
# Build the pinned NCEPLIBS the stand-alone HRRR tools link, and lay down the
# WCOSS2 names the tag's build files call by absolute path or by driver name.
# Runs inside the toolchain stage after build_thirdparty.sh.
set -euo pipefail
JOBS="${1:-8}"
S=/opt/src
B=/tmp/build-nceplibs
mkdir -p "$B" "$NCEPLIBS"
cd "$B"

export CC=icc CXX=icpc FC=ifort F77=ifort F90=ifort

# NCEPLIBS (CMake). bufr builds its static-allocation kinds 4/8/d only under
# the Intel C compiler, which is what the tools' BUFR_LIB4/BUFR_LIBd name.
for pkg in NCEPLIBS-bacio-v2.4.1:NCEPLIBS-bacio-2.4.1 \
           NCEPLIBS-w3nco-v2.4.1:NCEPLIBS-w3nco-2.4.1 \
           NCEPLIBS-bufr-bufr_v11.4.0:NCEPLIBS-bufr-bufr_v11.4.0 \
           NCEPLIBS-g2-v3.4.5:NCEPLIBS-g2-3.4.5 \
           NCEPLIBS-g2tmpl-v1.10.0:NCEPLIBS-g2tmpl-1.10.0; do
  tarball="${pkg%%:*}"; dir="${pkg##*:}"
  tar xzf "$S/$tarball.tar.gz"
  cmake -S "$dir" -B "$dir-build" -DCMAKE_INSTALL_PREFIX="$NCEPLIBS" \
    -DCMAKE_PREFIX_PATH="$PREFIX;$NCEPLIBS" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_LIBDIR=lib
  cmake --build "$dir-build" -j"$JOBS" && cmake --install "$dir-build"
done

# wrf_io 1.1.1 ships a makefile and (stale) prebuilt objects; build it clean.
tar xzf "$S/NCEPLIBS-wrf_io-v1.1.1.tar.gz"
mkdir -p "$NCEPLIBS/wrf_io"
cp -r NCEPLIBS-wrf_io-1.1.1/wrf.post.lib/v1.1.1/lib/wrf_io/. "$NCEPLIBS/wrf_io/"
( cd "$NCEPLIBS/wrf_io" && make clean && make mySFC=ifort myFC=mpiifort NETCDF="$PREFIX" )

# Cray driver names the tag's makefiles and build scripts call (ftn, cc, CC),
# over the Intel MPI drivers. Like the Cray driver, each supplies the OpenMP
# runtime when it links: the tag's WPS links WRF's I/O library, which is
# compiled with -qopenmp, through a link line that names no OpenMP flag, and
# without the runtime that link fails on __kmpc_* symbols (MEASURED
# 2026-10-03). A compile-only call is passed through untouched, so no source
# is compiled with OpenMP that the tag does not compile with it.
mkdir -p /opt/craywrap
for pair in ftn:mpiifort cc:mpiicc CC:mpiicpc; do
  cat > "/opt/craywrap/${pair%%:*}" <<EOF
#!/bin/sh
link=1
[ \$# -eq 0 ] && link=0
for arg in "\$@"; do
  case "\$arg" in -c|-E|-S|-V|--version|-syntax-only|-fsyntax-only) link=0 ;; esac
done
if [ \$link -eq 1 ]; then exec ${pair##*:} "\$@" -liomp5 -lpthread; fi
exec ${pair##*:} "\$@"
EOF
  chmod 755 "/opt/craywrap/${pair%%:*}"
done

# The tag's WRF configure file (configure.wrf.useme) names the WCOSS2 netcdf,
# hdf5 and pnetcdf installs by absolute path. The same paths here point at
# the libraries built above, so that file is used as shipped.
hpc=/apps/prod/hpc-stack/intel-19.1.3.304/cray-mpich-8.1.4
mkdir -p "$hpc/netcdf" "$hpc/hdf5" /apps/prod/pnetcdf/1.12.2/intel/19.1.3.304/cray-mpich
ln -sfn "$PREFIX" "$hpc/netcdf/4.7.4"
ln -sfn "$PREFIX" "$hpc/hdf5/1.10.6"
ln -sfn "$PREFIX" /apps/prod/pnetcdf/1.12.2/intel/19.1.3.304/cray-mpich/8.1.4

ls -l "$NCEPLIBS/lib" "$NCEPLIBS/wrf_io/libwrfio_nf.a"
cd / && rm -rf "$B"
