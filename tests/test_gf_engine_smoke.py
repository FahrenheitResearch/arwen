"""Device smoke for the cu_physics=3 route through the ENGINE's call path.

Not a parity harness: no oracle CSV is read here.  The parity truth lives
in tests/test_gf_gfdrv_cuda.py (GFDRV bitwise at the WRF boundary); this
file proves the other half of admission -- that a RunConfig asking for
Grell-Freitas gets Grell-Freitas through initialize_physics and
PhysicsDriver.compute, on the same seams every scheme crosses: the
resolver binding, the bind_driver hook, the Task-1 CumulusResult contract,
the cu_rates hold, and the RAINC accumulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

cp = pytest.importorskip("cupy")


def _state_for(cfg):
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    return init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.010 * np.exp(-np.asarray(z) / 2200.0))


@pytest.mark.gpu
@requires_gpu
def test_cu_physics_3_routes_to_grell_freitas_and_steps():
    from gpuwm.config import RunConfig, validate_run_config
    from gpuwm.core.gf import GrellFreitas
    from gpuwm.core.physics import initialize_physics

    cfg = validate_run_config(RunConfig(
        nx=4, ny=2, nz=16, dx=12000.0, dy=12000.0, ztop=9000.0,
        dt=60.0, run_seconds=0.0, moist=True, mp_physics=10,
        bl_pbl_physics=1, sf_sfclay_physics=91, bldt=0.0,
        cu_physics=3, cudt_minutes=0.0, ishallow=1))
    state = _state_for(cfg)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=290.0)
    # cu_physics=3 auto-binds the Grell-Freitas adapter, the same
    # scheme-ID binding law as KF's.
    assert isinstance(driver.cumulus_callable, GrellFreitas)

    tendencies = driver.compute(state, cfg)
    assert driver.call_counts["cumulus"] == 1
    # GF runs on the model step: a second step is a second due call.
    state.elapsed_seconds = cfg.dt
    driver.compute(state, cfg)
    assert driver.call_counts["cumulus"] == 2

    # The bind_driver hook connected the adapter to the held radiation
    # rates (zero here -- radiation off -- but the wiring must exist).
    assert driver.cumulus_callable._driver is driver

    shape = state.p.shape
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten"):
        rate = driver.cu_rates[name]
        assert rate.shape == shape
        assert bool(cp.isfinite(rate).all()), name
    # Task-1 contract: RAINC accumulated the kernel's RAINCV increment.
    assert driver.rainc.shape == shape[1:]
    assert bool(cp.isfinite(driver.rainc).all())
    assert bool((driver.rainc >= 0).all())
    assert bool(cp.isfinite(tendencies.rtheta).all())
    assert bool(cp.isfinite(tendencies.rqv).all())


@pytest.mark.gpu
@requires_gpu
def test_the_engine_identity_is_the_corrected_k22():
    """The shipped algorithm IS the corrected indexing, stated where a
    restart would bind it; the WRF-faithful flag has no RunConfig spelling
    and is reachable only by the parity suites' direct kernel launches."""
    import inspect

    from gpuwm.core import gf as gf_module
    from gpuwm.io.restart import CUMULUS_ALGORITHM_IDENTITIES

    assert "corrected-k22" in CUMULUS_ALGORITHM_IDENTITIES[3]
    source = inspect.getsource(gf_module)
    assert "SHIPPED default: corrected k22" in source
    from gpuwm.config import RunConfig
    assert "k22_wrf_faithful" not in RunConfig.__dataclass_fields__


def test_grell_freitas_requires_a_pbl_scheme_and_pinned_cudt():
    from gpuwm.config import RunConfig, validate_run_config

    base = dict(nx=4, ny=2, nz=16, dx=12000.0, dy=12000.0, ztop=9000.0,
                dt=60.0, run_seconds=0.0, moist=True, mp_physics=10)
    with pytest.raises(ValueError, match="requires a PBL scheme"):
        validate_run_config(RunConfig(
            **base, cu_physics=3, cudt_minutes=0.0))
    with pytest.raises(ValueError, match="cudt_minutes=0"):
        validate_run_config(RunConfig(
            **base, bl_pbl_physics=1, sf_sfclay_physics=91,
            cu_physics=3, cudt_minutes=5.0))
    with pytest.raises(ValueError, match="Grell-family keys"):
        validate_run_config(RunConfig(
            **base, cu_physics=1, cudt_minutes=5.0, ishallow=1))
