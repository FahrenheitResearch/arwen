#!/bin/bash
set -u
python_bin="${GPUWM_ORACLE_PYTHON:-python}"
receipt_root="${GPUWM_ORACLE_RECEIPTS:?set GPUWM_ORACLE_RECEIPTS}"
trap 'touch "$receipt_root/vertical-eos.DONE"' EXIT
"$python_bin" tools/smallstep_wrf471_oracle/vertical_receipt.py \
  --library "$GPUWM_SMALLSTEP_ORACLE_LIB" --output "$receipt_root/vertical-eos-causal.json" \
  --no-fma --theta-offset 0 --wrf-phi-order --wrf-top-order --wrf-update-order \
  --wrf-eos-order --preserve-subnormals > "$receipt_root/vertical-eos-causal.log" 2>&1
echo $? > "$receipt_root/vertical-eos-causal.rc"
