#!/bin/bash
set -u
python_bin="${GPUWM_ORACLE_PYTHON:-python}"
receipt_root="${GPUWM_ORACLE_RECEIPTS:?set GPUWM_ORACLE_RECEIPTS}"
trap 'touch "$receipt_root/vertical-focused.DONE"' EXIT
"$python_bin" -m pytest tests/test_smallstep_vertical_wrf471_parity.py tests/test_acoustic.py \
  tests/test_acoustic_nz_tiers.py tests/test_diffusion.py tests/test_steep_terrain_step.py -q \
  > "$receipt_root/vertical-focused-tests.log" 2>&1
echo $? > "$receipt_root/vertical-focused-tests.rc"
