#!/bin/bash
# Run from the engine source tree, inside the GPU ownership wrapper.
set -u
python_bin="${GPUWM_ORACLE_PYTHON:-python}"
receipt_root="${GPUWM_ORACLE_RECEIPTS:?set GPUWM_ORACLE_RECEIPTS}"
mkdir -p "$receipt_root"
trap 'touch "$receipt_root/vertical-final.DONE"' EXIT
"$python_bin" tools/smallstep_wrf471_oracle/vertical_receipt.py \
  --library "$GPUWM_SMALLSTEP_ORACLE_LIB" \
  --output tests/data/wrf471_smallstep/vertical-native.json \
  --arrays tests/data/wrf471_smallstep/vertical-reference.npz \
  > "$receipt_root/vertical-final-native.log" 2>&1
echo $? > "$receipt_root/vertical-final-native.rc"
"$python_bin" tools/smallstep_wrf471_oracle/vertical_receipt.py \
  --library "$GPUWM_SMALLSTEP_ORACLE_LIB" \
  --output "$receipt_root/vertical-causal.json" \
  --no-fma --theta-offset 0 --wrf-phi-order --wrf-top-order --preserve-subnormals \
  > "$receipt_root/vertical-causal.log" 2>&1
echo $? > "$receipt_root/vertical-causal.rc"
"$python_bin" -m pytest tests/test_smallstep_vertical_wrf471_parity.py -q \
  > "$receipt_root/vertical-final-tests.log" 2>&1
echo $? > "$receipt_root/vertical-final-tests.rc"
