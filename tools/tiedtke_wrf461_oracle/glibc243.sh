#!/bin/sh
# glibc243.sh SYSROOT CMD... : run CMD under Ubuntu glibc 2.43 (libc6 2.43-2ubuntu2.4).
#
# The project's WRF oracles link the libm of weather-node-1 (Ubuntu glibc
# 2.43, libm.so.6 sha256 1c488db6...).  gfortran binds WRF's REAL32 tan() to
# that libm's tanf, which since glibc 2.41 is CORE-MATH's correctly rounded
# tanf, the function gpuwm/core/kernels/glibc_trig_flt32.cuh transcribes.
# glibc 2.40 and older (Ubuntu 24.04 ships 2.39) carry a tanf that is not
# correctly rounded: over every argument cu6 can reach it differs from 2.43
# by 1 ULP on 1,079,526 of them (tan_domain_check.py), and that one word
# reaches the cu6 tendencies through organized detrainment (F:2939).  An
# oracle run on such a host reports that libm, not the port.
#
# On an older host this fetches the pinned 2.43 libc6 package once into
# SYSROOT and starts CMD with the 2.43 loader and libc/libm ahead of the
# host's own libraries.  Nothing is installed system-wide.
set -eu
root=${1:?pass a lane-owned sysroot directory}; shift
deb=libc6_2.43-2ubuntu2.4_amd64.deb
lib=$root/usr/lib/x86_64-linux-gnu
if [ ! -f "$lib/libm.so.6" ]; then
  mkdir -p "$root"
  curl --fail --location --silent --show-error -o "$root/$deb" "http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/$deb"
  echo "a613b457f3ff9c84ebd28af8a05afdec22605ad9aed87c71e3a6ae5170aeb9ca  $root/$deb" | sha256sum -c - >&2
  dpkg-deb -x "$root/$deb" "$root"
fi
echo "1c488db6f4fe8cecb72456bc018de000b7a9b539c2994864b0416b5e8371b3d8  $lib/libm.so.6" | sha256sum -c --quiet -
exec "$lib/ld-linux-x86-64.so.2" --library-path "$lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$@"
