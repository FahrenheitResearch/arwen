#!/usr/bin/env bash
# Run on a sprint box, with the lane's source-pinned public WRF clone.
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: build_regimes.sh WRF_ROOT LANE_ROOT' >&2; exit 2; }
wrf=$(realpath "$1")
lane=$(realpath "$2")
tool=$(cd "$(dirname "$0")" && pwd)
cd "$tool/../.."
python -m tools.sfire_wrf471_oracle.correct_regime_sources "$wrf" "$lane/corrected"
SFIRE_CORRECTED_PHYS_SOURCE="$lane/corrected/module_fr_fire_phys.F" \
SFIRE_CORRECTED_CORE_SOURCE="$lane/corrected/module_fr_fire_core.F" \
SFIRE_CORRECTED_MODEL_SOURCE="$lane/corrected/module_fr_fire_model.F" \
SFIRE_CORRECTED_ATM_SOURCE="$lane/corrected/module_fr_fire_atm.F" \
    bash "$tool/build.sh" "$wrf" "$lane/oracle-build"
python -m tools.sfire_wrf471_oracle.pack_fixtures "$lane/oracle-build" "$lane/regime-fixtures"
