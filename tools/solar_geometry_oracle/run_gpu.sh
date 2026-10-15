#!/usr/bin/env bash
set -u
T=$(cd "$(dirname "$0")/../.." && pwd)
L=$(dirname "$T")
source /work/sprint/env.sh
export GPUWM_SWINT_ORACLE=/work/sprint/engine-base/tests/data/swint_oracle/swint_oracle.npz
cd "$T"
status=0
for mode in 0 1; do
    export GPUWM_WRF_EXACT=$mode
    python -m tools.solar_geometry_oracle.compare_versions tests/data/oracles/solar_generations "$L/solar-$mode.json" > "$L/solar-$mode.log" 2>&1
    result=$?
    echo "solar mode $mode exit $result"
    if [ "$result" != 0 ]; then status=1; fi
    cat "$L/solar-$mode.log"
    python -m pytest -q tests/test_solar_generation_gpu.py tests/test_swint_interpolation.py > "$L/consumers-$mode.log" 2>&1
    result=$?
    echo "consumers mode $mode exit $result"
    if [ "$result" != 0 ]; then status=1; fi
    cat "$L/consumers-$mode.log"
done
exit "$status"
