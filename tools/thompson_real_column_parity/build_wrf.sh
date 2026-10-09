#!/usr/bin/env bash
# Build the WRF v4.6.1 batch column drivers twice -- pristine and
# rate-instrumented -- and refuse unless both produce byte-identical column
# outputs on the same input.  Two drivers link against each module build:
# run_columns_aero (mp_physics=28, aerosol aware) and run_columns_classic
# (mp_physics=8, is_aerosol_aware false).
#
# usage:
#   ./build_wrf.sh WRF_PHYS_DIR BUILD_DIR TABLE_DIR [PROBE_INPUT.bin] [stock|corrected-racg]
#
# WRF_PHYS_DIR holds module_mp_thompson.F and module_mp_radar.F from tag
#   v4.6.1 (commit d66e442f); their SHA-256s are pinned below, the same pins
#   gpuwm/data/thompson/oracle-aero/PROVENANCE.txt records.
# TABLE_DIR holds qr_acr_qg_V4.dat, qr_acr_qsV2.dat, freezeH2O.dat and
#   CCN_ACTIVATE.BIN (the product's staged table root is exactly that set);
#   all four are SHA-pinned and linked into BUILD_DIR/run.
# PROBE_INPUT.bin, when given, is run through both binaries and the outputs
#   compared byte for byte (the fidelity proof of the instrumentation).
#
# Flags are the oracle's: -O2 -fno-tree-vectorize, and the build refuses a
# binary that links libmvec (tools/thompson_wrf461_oracle/build_aero.sh
# explains why that flag is essential).
set -euo pipefail

fc=${FC:-gfortran}
opt_flags=${OPT_FLAGS:--O2 -fno-tree-vectorize}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
oracle=$(CDPATH= cd -- "$here/../thompson_wrf461_oracle" && pwd)

if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
  echo "usage: $0 WRF_PHYS_DIR BUILD_DIR TABLE_DIR [PROBE_INPUT.bin] [stock|corrected-racg]" >&2
  exit 2
fi
phys=$(realpath "$1")
build=$(realpath -m "$2")
tables=$(realpath "$3")
probe=${4:-}
source_variant=${5:-stock}
case "$source_variant" in
  stock|corrected-racg) ;;
  *) echo "unknown oracle source variant: $source_variant" >&2; exit 2 ;;
esac

THOMPSON_SHA=fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802
RADAR_SHA=aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688
QR_ACR_QG_SHA=89b779855847b2acdca1b40e24c5f1bd89b0c6ed105ca91a5a076d80c2437c3f
QR_ACR_QS_SHA=47350be20bd59c9f31378dd5805ce7d35fd14bebcfafb4ade56626f6eed818d7
FREEZEH2O_SHA=c235d1ce6f8750a671b2273d0e216ed3acf9a869bfd52a14676826f87aab5c02
CCN_SHA=f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd

echo "$THOMPSON_SHA  $phys/module_mp_thompson.F" | sha256sum -c -
echo "$RADAR_SHA  $phys/module_mp_radar.F" | sha256sum -c -
echo "$QR_ACR_QG_SHA  $tables/qr_acr_qg_V4.dat" | sha256sum -c -
echo "$QR_ACR_QS_SHA  $tables/qr_acr_qsV2.dat" | sha256sum -c -
echo "$FREEZEH2O_SHA  $tables/freezeH2O.dat" | sha256sum -c -
echo "$CCN_SHA  $tables/CCN_ACTIVATE.BIN" | sha256sum -c -

mkdir -p "$build/pristine" "$build/rates" "$build/run"
oracle_source="$phys/module_mp_thompson.F"
if [ "$source_variant" = corrected-racg ]; then
  oracle_source="$build/pristine/module_mp_thompson.F"
  python3 "$here/corrected_real_reference.py" source \
    "$phys/module_mp_thompson.F" "$oracle_source" "$build/RACG-SOURCE.json"
fi
python3 "$here/instrument_wrf_rates.py" "$oracle_source" \
  "$build/rates/module_mp_thompson.F" "$build/wrf_rate_schema.json"

for variant in pristine rates; do
  cd "$build/$variant"
  if [ "$variant" = pristine ]; then
    src="$oracle_source"
  else
    src="$build/rates/module_mp_thompson.F"
  fi
  $fc -c $opt_flags -ffree-form -ffree-line-length-none "$oracle/stub_wrf.F90"
  $fc -c $opt_flags -ffree-form -ffree-line-length-none "$phys/module_mp_radar.F"
  $fc -c $opt_flags -cpp -DWRF_CHEM=0 -ffree-form -ffree-line-length-none \
    "$src" -o module_mp_thompson.o
  for driver in run_columns_aero run_columns_classic; do
    $fc -c $opt_flags -ffree-form -ffree-line-length-none \
      "$here/$driver.F90"
    $fc $opt_flags -o $driver stub_wrf.o module_mp_radar.o \
      module_mp_thompson.o $driver.o
    if nm -D $driver 2>/dev/null | grep -q '_ZGV'; then
      echo "libmvec SIMD math linked into $variant/$driver" >&2
      exit 3
    fi
  done
done

cd "$build/run"
for t in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat CCN_ACTIVATE.BIN; do
  ln -sf "$tables/$t" "$t"
done

{
  echo "wrf_commit = d66e442fccc04111067e29274c9f9eaccc3cef28 (tag v4.6.1)"
  echo "module_mp_thompson.F = $THOMPSON_SHA"
  echo "oracle_source_variant = $source_variant"
  echo "compiled module_mp_thompson.F = $(sha256sum "$oracle_source" | cut -d' ' -f1)"
  if [ "$source_variant" = corrected-racg ]; then
    echo "oracle defect repaired = eight non-hail rain-graupel reads exceed dimNRHG=1; MIN(idx_bg(k),dimNRHG) selects the supplied slab"
    echo "oracle repair receipt = RACG-SOURCE.json"
  fi
  echo "module_mp_radar.F = $RADAR_SHA"
  echo "instrumented module_mp_thompson.F = $(sha256sum "$build/rates/module_mp_thompson.F" | cut -d' ' -f1)"
  echo "fortran = $($fc --version | head -1)"
  echo "libc = $(ldd --version | head -1)"
  echo "opt_flags = $opt_flags"
  echo "pristine binary = $(sha256sum "$build/pristine/run_columns_aero" | cut -d' ' -f1)"
  echo "rates binary = $(sha256sum "$build/rates/run_columns_aero" | cut -d' ' -f1)"
  echo "pristine classic binary = $(sha256sum "$build/pristine/run_columns_classic" | cut -d' ' -f1)"
  echo "rates classic binary = $(sha256sum "$build/rates/run_columns_classic" | cut -d' ' -f1)"
} > "$build/BUILD-RECEIPT.txt"

if [ -n "$probe" ]; then
  export GFORTRAN_CONVERT_UNIT='big_endian:20'
  "$build/pristine/run_columns_aero" "$(realpath "$probe")" "$build/run/probe-pristine.out"
  "$build/rates/run_columns_aero" "$(realpath "$probe")" "$build/run/probe-rates.out"
  unset GFORTRAN_CONVERT_UNIT
  if ! cmp -s "$build/run/probe-pristine.out" "$build/run/probe-rates.out"; then
    echo "instrumented build changed the column outputs; refusing" >&2
    exit 4
  fi
  echo "fidelity: instrumented outputs byte-identical to pristine on $(basename "$probe")" \
    | tee -a "$build/BUILD-RECEIPT.txt"
fi
cat "$build/BUILD-RECEIPT.txt"
