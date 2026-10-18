#!/usr/bin/env bash
# Run every chem parity test on one GPU, logging each oracle comparison.
#
#   bash tools/chem_wrf461_verify/run_parity.sh strict|default OUT_DIR
#
# strict  : GPUWM_WRF_EXACT=1 (no FMA, no FTZ, IEEE div/sqrt; the 0-ULP standard)
# default : GPUWM_WRF_EXACT unset (the arithmetic a bare run uses)
#
# Expects the tree to be importable (PYTHONPATH=$T:$T/tests, as the verify
# base's env.sh sets) and a CUDA card visible.  Writes OUT_DIR/<mode>/:
#   ulp.jsonl   one line per ulp_table call (test, shape, max_ulp, n_nonzero, n)
#   junit.xml   per-test outcome
#   pytest.log  console output
set -euo pipefail
mode=${1:?usage: run_parity.sh strict|default OUT_DIR}
out=$(realpath -m "${2:?usage: run_parity.sh strict|default OUT_DIR}")/$mode
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tree=$(cd "$here/../.." && pwd)
case "$mode" in
    strict) export GPUWM_WRF_EXACT=1 ;;
    default) unset GPUWM_WRF_EXACT ;;
    *) echo "mode must be strict or default" >&2; exit 2 ;;
esac
mkdir -p "$out"
rm -f "$out/ulp.jsonl"
export CHEM_ULP_LOG="$out/ulp.jsonl"
export PYTHONPATH="$here:${PYTHONPATH:-}"
cd "$tree"
tests=(tests/test_chem_*_wrf471_parity.py tests/test_chem_rrtmgp_aerosol_parity.py
       tests/test_chem_plumerise_frp_parity.py tests/test_chem_fire_emissions.py
       tests/test_chem_outputs.py tests/test_chem_mp_coupling.py)
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader > "$out/gpu.txt" || true
set +e
python -m pytest -p ulp_capture -q -rA --junitxml="$out/junit.xml" "${tests[@]}" \
    > "$out/pytest.log" 2>&1
status=$?
set -e
tail -n 5 "$out/pytest.log"
python "$here/summarize.py" "$out" > "$out/summary.json"
exit $status
