#!/bin/bash
# CPU only. Build byte-unmodified public TEMPO; no source instrumentation.
set -euo pipefail
WORK=$(realpath -m "${1:?usage: build_adapter_reference.sh WORK}")
HERE=$(cd "$(dirname "$0")" && pwd)
PIN=9adf9ef2827104640125f0c322e1218fce74e94c
mkdir -p "$WORK/build" "$WORK/tables" "$WORK/run"
if [ ! -d "$WORK/TEMPO" ]; then
  git clone -q https://github.com/NCAR/TEMPO.git "$WORK/TEMPO"
  git -C "$WORK/TEMPO" checkout -q "$PIN"
fi
test "$(git -C "$WORK/TEMPO" rev-parse HEAD)" = "$PIN"
test -z "$(git -C "$WORK/TEMPO" status --porcelain --untracked-files=no)"
for t in ccn_activate.bin freeze_water_data_tempo_v3 qr_acr_qg_data_tempo_v3 qr_acr_qs_data_tempo_v3; do
  if [ ! -f "$WORK/tables/$t" ]; then
    curl -fL --retry 3 "https://github.com/NCAR/TEMPO/releases/download/tempo_v3.1.0/$t" -o "$WORK/tables/$t"
  fi
  ln -sf "$WORK/tables/$t" "$WORK/run/$t"
done
python -c 'import sys; from gpuwm.tempo_assets import TEMPO_TABLE_ASSETS; from gpuwm.core.thompson_contract import validate_table_assets; from pathlib import Path; validate_table_assets(Path(sys.argv[1]), TEMPO_TABLE_ASSETS)' "$WORK/tables"
FCFLAGS='-c -g -O0 -fbacktrace -fcheck=all -ffree-line-length-0 -fdefault-real-8 -fdefault-double-8'
for f in machine module_mp_tempo_cfgs module_mp_tempo_params module_mp_tempo_ml module_mp_tempo_utils module_mp_tempo_aerosols module_mp_tempo_diags module_mp_tempo_tables module_mp_tempo_main module_mp_tempo_driver; do
  gfortran $FCFLAGS -J "$WORK/build" -I "$WORK/build" "$WORK/TEMPO/src/$f.F90" -o "$WORK/build/$f.o"
done
# The historical driver also runs scenarios requiring its old scenario file.
# Use its init dump only. This is our driver, not modified TEMPO source.
sed '/tempo_ref_driver: wrote tempo_blob.bin/a\  stop' "$HERE/../../tempo_ref_driver.F90" > "$WORK/build/blob_driver.F90"
gfortran $FCFLAGS -J "$WORK/build" -I "$WORK/build" "$WORK/build/blob_driver.F90" -o "$WORK/build/tempo_ref_driver.o"
gfortran $FCFLAGS -J "$WORK/build" -I "$WORK/build" "$HERE/tempo_adapter_driver.F90" -o "$WORK/build/tempo_adapter_driver.o"
LIBOBJ=()
for f in "$WORK"/build/{machine,module_mp_tempo_*}.o; do LIBOBJ+=("$f"); done
gfortran -o "$WORK/build/tempo_ref_driver" "$WORK/build/tempo_ref_driver.o" "${LIBOBJ[@]}"
gfortran -o "$WORK/build/tempo_adapter_driver" "$WORK/build/tempo_adapter_driver.o" "${LIBOBJ[@]}"
ulimit -s unlimited
(cd "$WORK/run"; "$WORK/build/tempo_ref_driver")
mv "$WORK/run/tempo_blob.bin" "$WORK/tempo_blob.bin"
python "$HERE/make_adapter_columns.py" "$WORK"
(cd "$WORK/run"; "$WORK/build/tempo_adapter_driver" "$WORK/adapter_in.bin" "$WORK/adapter_out.bin")
(cd "$WORK/run"; "$WORK/build/tempo_adapter_driver" "$WORK/adapter_in.bin" "$WORK/adapter_cloud_out.bin" cloud)
gfortran --version | sed -n 1p > "$WORK/COMPILER.txt"
ldd --version | sed -n 1p >> "$WORK/COMPILER.txt"
sha256sum "$WORK"/{adapter_in.bin,adapter_out.bin,adapter_catalog.json,tempo_blob.bin} "$WORK/TEMPO/src/"*.F90 "$WORK/tables/"* > "$WORK/SHA256SUMS"
