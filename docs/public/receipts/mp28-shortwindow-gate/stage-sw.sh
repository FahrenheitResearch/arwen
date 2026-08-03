#!/bin/bash
# ICs per banked stage-runs.sh: ONE ideal.exe (stock vec build) per mp option.
set -e
T=mp28sw
for MP in 8 28; do
  M=$(printf %02d $MP)
  D=$T/runs/ic-mp$M
  rm -rf $D; mkdir -p $D; cd $D
  ln -sf $T/WRF-4.6.1/main/ideal.exe .
  ln -sf $T/input_sounding .
  bash $T/mknml.sh namelist.input $MP 120 40 2000 20000 12 120 10
  ./ideal.exe > ideal.log 2>&1
  echo "ic mp=$MP rc=$? $(stat -c%s wrfinput_d01) bytes"
  sha256sum wrfinput_d01
done
