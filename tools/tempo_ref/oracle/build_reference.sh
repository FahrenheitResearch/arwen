#!/bin/bash
# TEMPO column oracle, referee side: build unmodified TEMPO 9adf9ef with its own
# Makefile.gfortran flags, run tempo_init on the pinned tables, write the
# Fortran init dump and run the column set.  CPU only.
#
#   build_reference.sh <work dir> <TEMPO table dir>
#
# <work dir> receives: TEMPO/ (clone at the pin), build/, tempo_blob.bin and
# tempo_ref_columns.bin (tools/tempo_ref/tempo_ref_driver.F90, the six
# historical scenarios), columns_in.bin + columns_catalog.json
# (make_columns.py), ref_out.bin (tempo_oracle_driver.F90) and SHA256SUMS.
# The table dir must hold TEMPO's four tables (gpuwm.tempo_assets pins them).
set -euo pipefail
WORK=${1:?usage: build_reference.sh WORK_DIR TABLE_DIR}
TABLES=${2:?usage: build_reference.sh WORK_DIR TABLE_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
PIN=9adf9ef2827104640125f0c322e1218fce74e94c
mkdir -p "$WORK"
cd "$WORK"
if [ ! -d TEMPO ]; then
  git clone -q https://github.com/NCAR/TEMPO.git TEMPO
fi
git -C TEMPO checkout -q $PIN
test "$(git -C TEMPO rev-parse HEAD)" = $PIN
test -z "$(git -C TEMPO status --porcelain --untracked-files=no)"   # unmodified

# TEMPO's Makefile.gfortran fcflags, verbatim
FCFLAGS="-c -g -O0 -fbacktrace -fcheck=all -ffree-line-length-0 -fdefault-real-8 -fdefault-double-8"
rm -rf build && mkdir -p build
for f in machine module_mp_tempo_cfgs module_mp_tempo_params module_mp_tempo_ml \
         module_mp_tempo_utils module_mp_tempo_aerosols module_mp_tempo_diags \
         module_mp_tempo_tables module_mp_tempo_main module_mp_tempo_driver; do
  gfortran $FCFLAGS -J build TEMPO/src/$f.F90 -o build/$f.o
done
gfortran $FCFLAGS -J build -I build "$HERE/../tempo_ref_driver.F90" -o build/tempo_ref_driver.o
gfortran $FCFLAGS -J build -I build "$HERE/tempo_oracle_driver.F90" -o build/tempo_oracle_driver.o
LIBOBJ=$(ls build/*.o | grep -v -e tempo_ref_driver.o -e tempo_oracle_driver.o)
gfortran -o build/tempo_ref_driver build/tempo_ref_driver.o $LIBOBJ
gfortran -o build/tempo_oracle_driver build/tempo_oracle_driver.o $LIBOBJ
gfortran --version | sed -n 1p > build/COMPILER.txt
ldd --version | sed -n 1p >> build/COMPILER.txt

# tempo_init reads the tables from the working directory
mkdir -p TEMPO/run_ref run_oracle
for t in ccn_activate.bin freeze_water_data_tempo_v3 qr_acr_qg_data_tempo_v3 qr_acr_qs_data_tempo_v3; do
  ln -sf "$TABLES/$t" TEMPO/run_ref/$t
  ln -sf "$TABLES/$t" run_oracle/$t
done
ulimit -s unlimited || true
(cd TEMPO/run_ref && "$WORK/build/tempo_ref_driver")
mv TEMPO/run_ref/tempo_blob.bin TEMPO/run_ref/tempo_ref_columns.bin .

"${PYTHON:-python3}" "$HERE/make_columns.py" "$WORK"
(cd run_oracle && "$WORK/build/tempo_oracle_driver" "$WORK/columns_in.bin" "$WORK/ref_out.bin")
sha256sum tempo_blob.bin tempo_ref_columns.bin columns_in.bin columns_catalog.json ref_out.bin \
  "$TABLES"/ccn_activate.bin "$TABLES"/*_tempo_v3 > SHA256SUMS
cat SHA256SUMS
