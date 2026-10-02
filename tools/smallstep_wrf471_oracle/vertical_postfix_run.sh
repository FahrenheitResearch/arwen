#!/bin/bash
set -u
python_bin="${GPUWM_ORACLE_PYTHON:-python}"
receipt_root="${GPUWM_ORACLE_RECEIPTS:?set GPUWM_ORACLE_RECEIPTS}"
trap 'touch "$receipt_root/vertical-postfix.DONE"' EXIT
"$python_bin" tools/smallstep_wrf471_oracle/vertical_receipt.py \
  --library "$GPUWM_SMALLSTEP_ORACLE_LIB" \
  --output tests/data/wrf471_smallstep/vertical-native.json \
  --arrays tests/data/wrf471_smallstep/vertical-reference.npz \
  > "$receipt_root/vertical-postfix-native.log" 2>&1
echo $? > "$receipt_root/vertical-postfix-native.rc"
"$python_bin" tools/smallstep_wrf471_oracle/vertical_receipt.py \
  --library "$GPUWM_SMALLSTEP_ORACLE_LIB" --output "$receipt_root/vertical-postfix-causal.json" \
  --no-fma --theta-offset 0 --wrf-phi-order --wrf-top-order --wrf-update-order --preserve-subnormals \
  > "$receipt_root/vertical-postfix-causal.log" 2>&1
echo $? > "$receipt_root/vertical-postfix-causal.rc"
"$python_bin" -m pytest tests/test_smallstep_vertical_wrf471_parity.py tests/test_acoustic.py \
  tests/test_acoustic_nz_tiers.py tests/test_diffusion.py tests/test_steep_terrain_step.py -q \
  > "$receipt_root/vertical-postfix-tests.log" 2>&1
echo $? > "$receipt_root/vertical-postfix-tests.rc"
