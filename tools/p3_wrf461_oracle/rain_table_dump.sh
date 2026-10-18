#!/usr/bin/env bash
# Dump WRF's own rain tables (vn_table, vm_table, revap_table) without
# editing module_mp_p3.F: the table loop (:607-670) and the constants it
# reads (:178-183, :219) are cut out of the pinned source by line number
# and wrapped in a program, compiled with arm A's flags, linked with and
# without crlibm_shim.o.
#   bash rain_table_dump.sh <WRF_SOURCE_ROOT> <OUT_DIR>
# Writes OUT_DIR/rain_tables_{glibc,CR}.bin: vn, vm, revap, each (300,10)
# float32, Fortran order.
set -euo pipefail
WRF=$1; OUT=$2; HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$WRF/phys/module_mp_p3.F
[ "$(sha256sum "$SRC" | cut -d' ' -f1)" = 716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6 ] || { echo "FATAL: source hash" >&2; exit 1; }
mkdir -p "$OUT"; F=$OUT/rain_table_dump.f90
{
  echo "program rain_table_dump"
  echo " implicit none"
  echo " real :: pi, thrd, sxth, piov3, piov6, mu_r_constant, mu_r, dm, lamr"
  echo " real :: dum1, dum2, dum3, dum4, dum5, dd, dia, amg, vt"
  echo " integer :: ii, jj, kk, procnum, u"
  echo " real, dimension(300,10) :: vn_table, vm_table, revap_table"
  echo " procnum = 1"
  sed -n '178,183p' "$SRC"
  sed -n '219p' "$SRC"
  sed -n '607,670p' "$SRC"
  echo " open(newunit=u, file='rain_tables.bin', access='stream', form='unformatted', status='replace')"
  echo " write(u) vn_table, vm_table, revap_table"
  echo " close(u)"
  echo "end program rain_table_dump"
} > "$F"
cd "$OUT"
gfortran -cpp -ffree-form -ffree-line-length-none -O0 -ffp-contract=off -fno-unsafe-math-optimizations "$F" -o rtd_glibc
gcc -O2 -c "$HERE/crlibm_shim.c" -o crlibm_shim.o
gfortran -cpp -ffree-form -ffree-line-length-none -O0 -ffp-contract=off -fno-unsafe-math-optimizations "$F" crlibm_shim.o -lquadmath -o rtd_cr
./rtd_glibc && mv rain_tables.bin rain_tables_glibc.bin
./rtd_cr && mv rain_tables.bin rain_tables_CR.bin
sha256sum rain_tables_glibc.bin rain_tables_CR.bin
