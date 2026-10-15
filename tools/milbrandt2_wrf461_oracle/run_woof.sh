#!/usr/bin/env bash
# GPU half of the mp=9 oracle: run the shipped launcher on every input set
# under the strict build (GPUWM_WRF_EXACT=1) and under default arithmetic,
# plus one strict run on a (4, ncol/4) layout, then summarise.
#
#   run_woof.sh WORKDIR          (WORKDIR holds in_*.bin and out_*_O2.bin)
#
# Run it through the box's GPU mutex.  The tree imported is whatever
# PYTHONPATH names (on the verification box: source env.sh with T set).
set -euo pipefail
work=$(realpath "$1")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${work}"
for inp in in_*.bin; do
    set_name=${inp#in_}; set_name=${set_name%.bin}
    [[ -f out_${set_name}_O2.bin ]] || continue
    GPUWM_WRF_EXACT=1 python "${here}/woof_run.py" "${inp}" "woof_${set_name}_strict.bin"
    GPUWM_WRF_EXACT=0 python "${here}/woof_run.py" "${inp}" "woof_${set_name}_default.bin"
    ncol=$(python -c "import numpy as n;print(int(n.fromfile('${inp}',dtype='<i4',count=1)[0]))")
    if (( ncol % 4 == 0 )); then
        GPUWM_WRF_EXACT=1 python "${here}/woof_run.py" "${inp}" \
            "woof_${set_name}_strict_layout4.bin" "4x$((ncol / 4))"
    fi
done
python "${here}/summarize.py" "${work}" "${work}/summary.json"
