#!/bin/bash
# probe.sh SNIPPET.f90 : read back WRF's own initialised NSSL module values.
# Appends "subroutine n2m_probe" (body = SNIPPET, see scalars.f90, arrays.f90,
# consts.f90, tables.f90) to the scalar-build WRF module copy that build.sh
# leaves in $ORACLE_BUILD/scalar, compiles it with the oracle's scalar flags,
# and runs it.  scalars.out and arrays.out are the readbacks the kernels cite.
set -e
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
b=${ORACLE_BUILD:?set ORACLE_BUILD to the build.sh output}/scalar
snip=$(realpath "$1")
d=$(mktemp -d)
python3 "$here/insert.py" "$snip" "$b/module_mp_nssl_2mom.f90" "$d/m.f90"
cd "$d"
gfortran -O2 -ftree-vectorize -funroll-loops -w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch -fallow-invalid-boz -fno-tree-vectorize -c m.f90
printf 'program main\nuse module_mp_nssl_2mom\ncall n2m_probe()\nend program\n' > main.f90
gfortran -O2 -c main.f90 && gfortran -o probe main.o m.o "$b/stub_wrf.o" "$b/stub_wrf3.o"
: > namelist.input
./probe
cd /; rm -rf "$d"
