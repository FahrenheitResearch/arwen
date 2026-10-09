#!/usr/bin/env bash
# Re-cut reviewed receipts on the real WRF driver. Run via the GPU queue.
set -euo pipefail
T=$(realpath "$1"); L=$(realpath "$2"); H=$T/tools/thompson_aerosol_column_oracle
R=$L/runs; mkdir -p "$R/stock" "$R/measurement" "$R/melting"
echo "$L/measurement-copy" >> "$L/MANIFEST.txt"
python - "$T" "$R" <<'PY'
import sys
from pathlib import Path
import numpy as np
t, r = map(Path, sys.argv[1:])
z = dict(np.load(t/'tests/data/mp28_column_oracle_wrf461.npz'))
sys.path.insert(0, str(t/'tools/thompson_aerosol_column_oracle'))
from make_oracle_gate import INPUTS
raw = {k: z[k] for k in INPUTS}
raw.update(labels=z['labels'], regime=np.asarray(['review']*157))
np.savez(r/'stock/columns.npz', **raw)
np.savez(r/'measurement/columns.npz', **raw)
m = {k: np.array(v[:3], copy=True) for k,v in raw.items()}
m['labels'] = np.asarray(['exact freezing low RH', 'exact freezing saturated', 'warm neighbor low RH'])
m['regime'] = np.asarray(['melting level']*3)
m['p'][:] = 100000
m['th'][:] = np.float32(273.15)
m['th'][2] = np.nextafter(np.float32(273.15), np.float32(np.inf))
for s in ('qc','qr','qi','qs'):
    m[s][:] = np.float32(1e-4)
m['qg'][:] = 0
m['qv'][:] = np.float32(1e-6)
m['qv'][1] = np.float32(0.0038)
m['nc'][:] = np.float32(1e8)
m['ni'][:] = np.float32(1e5)
m['nr'][:] = np.float32(1e4)
np.savez(r/'melting/columns.npz', **m)
PY
bash "$H/make_racg_read_copy.sh" "$T" "$L/measurement-copy"
for dt in 20 5; do
    GPUWM_WRF_EXACT=1 python "$H/gpu_run.py" "$R/stock/columns.npz" "$R/stock/gpu-strict-dt$dt.npz" --dt "$dt" --serial-fallout
    GPUWM_WRF_EXACT=1 python "$L/measurement-copy/tools/thompson_aerosol_column_oracle/gpu_run.py" "$R/measurement/columns.npz" "$R/measurement/gpu-strict-dt$dt.npz" --dt "$dt"
    python "$H/compare.py" "$R/stock/columns.npz" "$R/stock/gpu-strict-dt$dt.npz" "$L/oracle" "$R/stock/summary-strict-dt$dt.json"
    python "$H/compare.py" "$R/measurement/columns.npz" "$R/measurement/gpu-strict-dt$dt.npz" "$L/oracle" "$R/measurement/summary-strict-dt$dt.json"
    GPUWM_WRF_EXACT=1 python "$H/gpu_run.py" "$R/melting/columns.npz" "$R/melting/gpu-strict-dt$dt.npz" --dt "$dt"
    python "$H/compare.py" "$R/melting/columns.npz" "$R/melting/gpu-strict-dt$dt.npz" "$L/oracle" "$R/melting/summary-strict-dt$dt.json"
    GPUWM_WRF_EXACT=1 python "$H/gpu_run.py" "$R/melting/columns.npz" "$R/melting/gpu-legacy-dt$dt.npz" --dt "$dt" --legacy-warm-mask
    python "$H/compare.py" "$R/melting/columns.npz" "$R/melting/gpu-legacy-dt$dt.npz" "$L/oracle" "$R/melting/summary-legacy-dt$dt.json"
done
python "$H/make_oracle_gate.py" "$R/stock" "$L/receipts/mp28_column_oracle_wrf461.npz" --wrfread "$R/measurement"
python - "$R/melting" "$L/receipts/mp28_melting_level_wrf461.npz" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
r = Path(sys.argv[1]); payload = dict(np.load(r/'columns.npz'))
for dt in (20,5):
    s = json.loads((r/f'summary-strict-dt{dt}.json').read_text())
    assert s['cells_differ'] == 0, s
    legacy = json.loads((r/f'summary-legacy-dt{dt}.json').read_text())
    assert legacy['cells_differ'] > 0, legacy
    z = np.load(r/f'summary-strict-dt{dt}.wrf.npz')
    payload.update({f'wrf_{k}_dt{dt}': z[k] for k in z.files})
np.savez_compressed(sys.argv[2], **payload)
PY
