#!/bin/bash
set -x
T=mp28sw
/venv/mp28/bin/python $T/arwen/tools/mp28_matched/shortwindow_gate.py \
  --runs $T/runs --out $T/out/shortwindow-gate.json
echo "GATE-RC=$?"
echo DONE > $T/gate.done
