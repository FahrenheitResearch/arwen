#!/bin/bash
set -u
python_bin="${GPUWM_ORACLE_PYTHON:-python}"
receipt_root="${GPUWM_ORACLE_RECEIPTS:?set GPUWM_ORACLE_RECEIPTS}"
trap 'touch "$receipt_root/vertical-pi.DONE"' EXIT
"$python_bin" tools/smallstep_wrf471_oracle/vertical_pi_probe.py \
  --library "$GPUWM_SMALLSTEP_ORACLE_LIB" --output "$receipt_root/vertical-pi-red.json" \
  > "$receipt_root/vertical-pi-red.log" 2>&1
echo $? > "$receipt_root/vertical-pi-red.rc"
