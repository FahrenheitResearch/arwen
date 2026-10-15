#!/usr/bin/env bash
set -euo pipefail
lane=$(realpath "$1")
T="$lane/src"
source /work/env.sh
mkdir -p "$lane/tmp"
export TMPDIR="$lane/tmp" PYTHONDONTWRITEBYTECODE=1
cd "$T"
python -m tools.sfire_wrf471_oracle.regime_receipt \
    --fixtures "$lane/regime-fixtures" --out "$lane/regime-receipt.json"
set +e
python -m tools.sfire_wrf471_oracle.regime_receipt --plant-one-ulp \
    --fixtures "$lane/regime-fixtures" --out "$lane/planted-receipt.json"
status=$?
set -e
[[ $status == 1 ]] || { echo "Planted kernel must exit 1, got $status" >&2; exit 1; }
python -m tools.sfire_wrf471_oracle.validate_planted "$lane/planted-receipt.json"
export SFIRE_REGIME_FIXTURES="$lane/regime-fixtures"
python -m pytest -q tests/test_sfire_wrf471_parity.py tests/test_sfire_atm_wrf471_parity.py \
    tests/test_sfire_driver_wrf471_parity.py tests/test_sfire_regime_receipt.py \
    tests/test_sfire_subnormal_runtime.py > "$lane/focused-default.log" 2>&1
GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_sfire_wrf471_parity.py \
    tests/test_sfire_atm_wrf471_parity.py tests/test_sfire_driver_wrf471_parity.py \
    tests/test_sfire_regime_receipt.py tests/test_sfire_subnormal_runtime.py > "$lane/focused-strict.log" 2>&1
tail -5 "$lane/focused-default.log"
tail -5 "$lane/focused-strict.log"
