#!/usr/bin/env bash
set -euo pipefail
mode=$1
L=/work/pool-sfclay-noftz-mynn-mm5-eta
T=$L/src
source /work/env.sh
export PYTHONPATH=$T:$T/tests CUPY_CACHE_DIR=$L/cupy-cache TMPDIR=$L/tmp
unset GPUWM_NO_LOCAL_GPU GPUWM_WRF_EXACT
if [[ $mode == strict ]]; then export GPUWM_WRF_EXACT=1; fi
cd "$T"
python -m pytest -q -rs tests/test_surface_subnormal_units_gpu.py > "$L/logs/portable-$mode.log" 2>&1
python -m pytest -q -rs tests/test_sfclay_noftz_regression.py > "$L/logs/after-$mode.log" 2>&1
python "$T/tools/sfclay_noftz_check/normal_hashes.py" > "$L/logs/normal-$mode.log" 2>&1
python -m pytest -q -rs tests/test_sfclay_classic_wrf461_parity.py tests/test_sfclayrev_wrf461_parity.py tests/test_myjsfc_wrf461_parity.py tests/test_mynn_sfclay_wrf461_column_oracle.py tests/test_mynn_sfclay_wrf461_column_oracle_gpu.py tests/test_mynn_gsd41_columns_exact_gpu.py tests/test_mynn_gsd41_columns_oracle.py tests/test_mynn_wrf461_exact_gpu.py > "$L/logs/focused-$mode.log" 2>&1
python "$T/tools/sfclay_noftz_check/plant.py" > "$L/logs/plant-driver-$mode.log" 2>&1
mkdir -p "$L/after-$mode"
cp "$L/receipts/"*"-$mode.json" "$L/after-$mode/"
echo "COMPLETE $mode"
tail -3 "$L/logs/portable-$mode.log"
tail -3 "$L/logs/after-$mode.log"
tail -3 "$L/logs/focused-$mode.log"
