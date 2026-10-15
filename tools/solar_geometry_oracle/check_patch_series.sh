#!/usr/bin/env bash
set -euo pipefail
T=$(cd "$(dirname "$0")/../.." && pwd)
L=$(dirname "$T")
source /work/env.sh
A=${1:-$L/applied}
case "$A" in "$L"/*) ;; *) echo 'apply worktree must stay inside the lane'; exit 2;; esac
BASE=4d1079249ad0cc2553a9632ac9ef267553913dd1
SLIM=3836f68c4fa77bd60ecf527bb408ee1832b8e359
git -C "$T" format-patch --output-directory "$L/patches" "$SLIM"..HEAD
git -C /work/engine-base worktree add --detach "$A" "$BASE"
# Reproduce the unrelated inventory defect on the untouched engine base.
set +e
(
    T="$A"
    source /work/env.sh
    cd "$T"
    CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 python -m pytest -q \
        tests/test_kernel_manifest.py::test_every_rawmodule_constructor_in_the_tree_is_named
) > "$L/inherited-inventory.log" 2>&1
echo "inherited inventory test exit $?"
set -e
git -C "$A" am -3 "$L"/patches/*.patch
python - "$T" "$A" "$L/patch-check.json" <<'PY'
from pathlib import Path
import json
import subprocess as sp
import sys
t, applied, receipt = map(Path,sys.argv[1:])
paths = sp.check_output(['git','-C',str(t),'diff','--name-only','3836f68c4fa77bd60ecf527bb408ee1832b8e359','HEAD'],text=True).splitlines()
rows = []
for path in paths:
    expected = sp.check_output(['git','-C',str(t),'rev-parse','HEAD:'+path],text=True).strip()
    actual = sp.check_output(['git','-C',str(applied),'rev-parse','HEAD:'+path],text=True).strip()
    rows.append({'path':path,'expected_blob':expected,'applied_blob':actual})
report = {'engine_base':'4d1079249ad0cc2553a9632ac9ef267553913dd1',
          'lane_commit':sp.check_output(['git','-C',str(t),'rev-parse','HEAD'],text=True).strip(),
          'applied_commit':sp.check_output(['git','-C',str(applied),'rev-parse','HEAD'],text=True).strip(),
          'changed_paths':rows}
receipt.write_text(json.dumps(report,indent=2)+'\n')
assert all(row['expected_blob']==row['applied_blob'] for row in rows), rows
print(len(rows),'changed paths match after git am -3 onto the exact engine base')
PY
