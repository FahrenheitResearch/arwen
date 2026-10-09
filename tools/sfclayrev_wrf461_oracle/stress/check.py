"""Mechanical gate: compare every WRF-defined finite word, never hide port NaNs."""
import json
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS

root = Path(os.environ['SFCLAYREV_CHECK_ROOT'])
mode, kind = sys.argv[1:]
data = np.load(root / f'{kind}-{mode}-words.npz')
w, p, mask = (data[k] for k in ('wrf', 'port', 'mask'))
finite = mask & np.isfinite(w)
different = (w.view(np.uint32) != p.view(np.uint32)) & finite
nonfinite = ~np.isfinite(p)
fields = {}
for f, name in enumerate(SFCLAY_OUTPUTS):
    m = finite[..., f]
    fields[name] = dict(words=int(m.sum()),
        differences=int(different[..., f].sum()),
        max_ulp=int(fp32_ulp_distance(w[..., f][m], p[..., f][m]).max()),
        port_nonfinite=int(nonfinite[..., f].sum()),
        wrf_finite_sha256=hashlib.sha256(w[..., f][m].tobytes()).hexdigest(),
        port_at_wrf_finite_sha256=hashlib.sha256(p[..., f][m].tobytes()).hexdigest(),
        port_full_field_sha256=hashlib.sha256(p[..., f].tobytes()).hexdigest())
report = dict(mode=mode, kind=kind, shape=list(p.shape),
    wrf_finite_words=int(finite.sum()),
    finite_differences=int(different.sum()),
    max_ulp=max(v['max_ulp'] for v in fields.values()),
    port_nonfinite=int(nonfinite.sum()),
    wrf_nan=int((np.isnan(w) & mask).sum()),
    wrf_inf=int((np.isinf(w) & mask).sum()), fields=fields)
(root / 'results' / f'gate-{kind}-{mode}.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'fields'}))
raise SystemExit(bool(report['finite_differences'] or report['port_nonfinite']))
