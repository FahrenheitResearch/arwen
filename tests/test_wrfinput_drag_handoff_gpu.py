"""Exercise the corrected native door with the actual device physics driver."""
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.ingest import wrfinput as wi
from gpuwm.static.orographic import required_static_fields
from test_wrfinput_physics_fields import _dimensions, _input, _from_driver

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("topo,gwd", [(0, 0), (1, 0), (2, 0), (0, 1), (0, 3), (2, 3)])
def test_native_drag_initializes_and_steps_real_driver(tmp_path, topo, gwd):
    import cupy as cp
    from gpuwm.core.dycore import step
    from test_ruc_runtime import _build

    state, initial_cfg, template = _build(nx=8, ny=6, nz=20, dt=1.)
    cfg = replace(initial_cfg, topo_wind=topo, gwd_opt=gwd)
    raw = _from_driver(template, cp)
    raw["HGT"] = cp.asnumpy(state.ht)
    for name in required_static_fields(topo, gwd):
        value = (2500. if name == "VAR_SSO" else 100. if name.startswith("VAR")
                 else .5 if name.startswith("CON") else .4 if name.startswith("OA") else .1)
        raw[name] = np.full((cfg.ny, cfg.nx), value, np.float32)
    restored = wi.read_wrfinput(
        _input(tmp_path / "input", cfg, raw), cfg=cfg,
        expected_dimensions=_dimensions(cfg), require_complete=False)
    driver = wi.initialize_wrfinput_physics(state, restored, cfg)
    assert (driver.terrain_drag is not None) == bool(topo or gwd)
    driver.set_forcing(gsw=0.)
    for _ in range(2):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()
    assert state.physics is driver
    for name in ("u", "v", "thp", "qv", "mup", "php"):
        assert bool(cp.isfinite(getattr(state, name)).all()), name
    assert bool(cp.isfinite(driver.fields["tsk"]).all())
