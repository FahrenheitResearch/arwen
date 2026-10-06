#!/usr/bin/env bash
# Build the libraries a GSI pin names, at the pin's versions, into
# /opt/gsi-pin/libs. Runs inside Dockerfile.pin after env-pin.sh. The recipe
# for each library is keyed by its name; the versions and tarballs come only
# from the pin file.
set -euo pipefail
pinfile="$1"; JOBS="${2:-8}"
# shellcheck disable=SC1090
. "$pinfile"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be positive to avoid an unlimited make build" >&2; exit 1; }
limit=$(nproc)
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r quota period < /sys/fs/cgroup/cpu.max
  if [[ "$quota" != max ]]; then limit=$((quota / period)); ((limit > 0)) || limit=1; fi
fi
((JOBS <= limit)) || JOBS=$limit
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
S=/opt/src-pin
P=$PIN_LIBS
: "${TMPDIR:?set TMPDIR inside the build workspace}"
B="$TMPDIR/build-pin-libs"
mkdir -p "$B" "$P"
cd "$B"

ver() { awk -F'|' -v n="$1" '$1==n{print $2}' <<< "$PIN_SOURCES"; }
unpack() {  # unpack <name> -> prints the top directory
  local name="$1" tarball="$S/$1-$(ver "$1").tar.gz" top
  # Some archives list their entries as ./<top>/..., so a leading ./ is
  # dropped before the first component is taken.
  top="$(tar tzf "$tarball" | sed 's#^\./##' | grep -v '^$' | head -1 | cut -d/ -f1)"
  [[ -n "$top" && "$top" != . ]] || { echo "no top folder in $tarball" >&2; exit 1; }
  rm -rf "$top"; tar xzf "$tarball"; echo "$top"
}
# Each CMake library installs into its own prefix, $P/pkg/<name>, as each is
# its own module on WCOSS2 and its own prefix in spack. One shared prefix does
# not work: bacio, w3emc and ip all install Fortran modules into include_4,
# GSI links bacio_4, and so GSI compiled against ip's 4-byte sp_mod instead of
# the 8-byte one it links (MEASURED 2026-10-03: error #6633 on splegend in
# general_specmod.f90).
prefixes="$P"
cmake_lib() {  # cmake_lib <name> <source dir> [cmake args...]
  local name="$1" src="$2"; shift 2
  cmake -S "$src" -B "$src-build" -DCMAKE_INSTALL_PREFIX="$P/pkg/$name" -DCMAKE_PREFIX_PATH="$prefixes" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_TESTING=OFF "$@"
  cmake --build "$src-build" -j"$JOBS"
  cmake --install "$src-build"
  prefixes="$prefixes;$P/pkg/$name"
}
export CPPFLAGS="-I$P/include" LDFLAGS="-L$P/lib"

# CMake, the Kitware binary (build.ver cmake_ver).
mkdir -p "$PIN_ROOT/cmake"
tar xzf "$S/cmake-$(ver cmake).tar.gz" -C "$PIN_ROOT/cmake" --strip-components=1
cmake --version | head -1

d=$(unpack zlib);  ( cd "$d" && CC=icc ./configure --prefix="$P" && make -j"$JOBS" && make install )

# hdf5 with MPI and the Fortran layer: WCOSS2's hdf5-D module is the
# MPI-dependent build.
d=$(unpack hdf5)
( cd "$d" && ./configure --prefix="$P" --enable-parallel --enable-fortran --enable-hl \
    --with-zlib="$P" --disable-tests --disable-tools && make -j"$JOBS" && make install )

# netcdf-c on that hdf5 (parallel netCDF-4 comes with it). No remote-data,
# byte-range or Zarr layers: nothing in GSI or EnKF opens a URL.
d=$(unpack netcdf-c)
( cd "$d" && ./configure --prefix="$P" --disable-dap --disable-byterange --disable-nczarr \
    --disable-libxml2 --disable-testsets && make -j"$JOBS" && make install )
d=$(unpack netcdf-fortran)
( cd "$d" && ./configure --prefix="$P" && make -j"$JOBS" && make install )

d=$(unpack bacio);  cmake_lib bacio "$d"
d=$(unpack bufr);   cmake_lib bufr "$d" -DBUILD_UTILS=OFF -DENABLE_PYTHON=OFF
d=$(unpack w3emc);  cmake_lib w3emc "$d"
# ip needs LAPACK; MKL is the LAPACK GSI itself links.
d=$(unpack ip);     cmake_lib ip "$d" -DBLA_VENDOR=Intel10_64lp_seq
d=$(unpack sigio);  cmake_lib sigio "$d"
d=$(unpack sfcio);  cmake_lib sfcio "$d"
d=$(unpack nemsio); cmake_lib nemsio "$d"
d=$(unpack wrf_io); cmake_lib wrf_io "$d"
d=$(unpack ncio);   cmake_lib ncio "$d"
d=$(unpack ncdiag); cmake_lib ncdiag "$d"
# CRTM's top-level CMakeLists adds its test folder unconditionally, and the
# tests need coefficient data the tarball does not carry. Spack's recipe
# comments that line out when tests are not run; so does this one. Nothing
# under libsrc/ is touched.
d=$(unpack crtm)
sed -i 's/^add_subdirectory(test)/# add_subdirectory(test)  (tests not built; see build_pin_libs.sh)/' "$d/CMakeLists.txt"
cmake_lib crtm "$d"

echo "$prefixes" > "$P/cmake-prefixes.txt"
ls "$P/lib" "$P/pkg"
cd / && rm -rf "$B"
