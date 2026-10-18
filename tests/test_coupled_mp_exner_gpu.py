"""The real WSM6 adapter must preserve the recorded scalar PII word."""
from types import SimpleNamespace

import numpy as np

from conftest import requires_gpu


@requires_gpu
def test_wsm6_adapter_uses_the_observed_rounded_exner_word():
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.wsm6 import apply

    shape = (4, 1, 1)
    pressure = np.array([0x47ba6953], np.uint32).view(np.float32)[0]
    scratch = {}
    def buffer(shape, name):
        if name not in scratch:
            scratch[name] = cp.zeros(shape, cp.float32)
        return scratch[name]
    state = SimpleNamespace(
        p=cp.asarray([pressure, 85000., 60000., 30000.], cp.float32).reshape(shape),
        thb=cp.full(4, 300., cp.float32), thp=cp.full(shape, 2., cp.float32),
        phb=cp.arange(5, dtype=cp.float32)*np.float32(9810.),
        php=cp.zeros((5,1,1), cp.float32), alt=cp.full(shape, 1., cp.float32),
        qv=cp.full(shape, .003, cp.float32), scratch=buffer,
        h_diabatic=cp.zeros(shape, cp.float32))
    for name in ("qc", "qr", "qi", "qs", "qg", "effc", "effi", "effs"):
        setattr(state, name, cp.zeros(shape, cp.float32))
    cfg = RunConfig(nx=1, ny=1, nz=4, dx=3000., dy=3000., ztop=4000.,
                    dt=15., run_seconds=15., moist=True, mp_physics=6)
    apply(state, cfg, 15.)
    # Stock WRF moist_physics_prep_em at the first differing history cell.
    actual = cp.asnumpy(scratch["wsm6_pii"]).view(np.uint32)[0,0,0]
    from gpuwm.wrf_exact import ENABLED
    # Independent strict WRF and base arithmetic words differ by one ULP.
    assert actual == (0x3f7c9c64 if ENABLED else 0x3f7c9c65)
