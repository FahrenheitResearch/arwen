"""Focused contracts for the native-HRRR proof harness."""

from datetime import datetime
import json
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from tools.hrrr_state_proof import (
    _physics_receipt,
    _physics_update_counts,
    _strict_json,
)
from gpuwm.core.physics import DECLARED_CONSTANT_GLW_WM2  # noqa: E402

#: The idealised constant downward longwave these fixtures declare.
#:
#: ``gpuwm.core.physics.initialize_physics`` no longer defaults ``glw``
#: (300.0 through 1.8.7): a land-surface suite with no longwave scheme
#: must state where its downward longwave comes from instead of being
#: handed a plausible-looking 300 W m-2 nobody chose.  These are
#: idealised columns; the constant is the right answer for them and this
#: is where they say so.  The VALUE is 1.8.7's default, so every fixture
#: below integrates exactly the numbers it always did.
_IDEALISED_GLW = DECLARED_CONSTANT_GLW_WM2


def test_receipt_reads_public_physics_driver_update_counters():
    driver = SimpleNamespace(
        radiation_callable=SimpleNamespace(update_count=3),
        microphysics_updates=7,
        # The microphysics object is diagnostics, not a counter owner.
        microphysics=SimpleNamespace(),
    )
    assert _physics_update_counts(driver) == {
        "radiation_update_count": 3,
        "microphysics_update_count": 7,
    }


@requires_gpu
@pytest.mark.gpu
def test_entire_physics_receipt_serializes_from_initialized_driver():
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics

    cfg = RunConfig(
        nx=6, ny=4, nz=16, dx=3000.0, dy=3000.0,
        ztop=12000.0, dt=15.0, run_seconds=15.0,
        time_step_sound=4, moist=True, mp_physics=6,
        ra_physics=0, ra_lw_physics=0, ra_sw_physics=1,
        sf_sfclay_physics=0, sf_surface_physics=0,
        bl_pbl_physics=0, cu_physics=0,
    )
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 298.0 + 0.004 * np.asarray(z),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.008 * np.exp(-np.asarray(z) / 2500.0))
    driver = initialize_physics(
        state, cfg,
        radiation_start_time=datetime(2026, 7, 18),
        radiation_latitude=np.full((cfg.ny, cfg.nx), 35.0),
        radiation_longitude=np.full((cfg.ny, cfg.nx), -98.0),
        glw=_IDEALISED_GLW)
    receipt = _strict_json({"physics": _physics_receipt(driver, cp)})
    # allow_nan=False is the actual JSON finiteness gate, not just a type check.
    encoded = json.dumps(receipt, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)["physics"]
    assert decoded["resolved_lw_sw"] == [0, 1]
    assert decoded["radiation_update_count"] == 0
    assert decoded["microphysics_update_count"] == 0
    assert decoded["rainnc_max_mm"] == 0.0
    assert decoded["swdown_min_wm2"] == 0.0
    assert decoded["swdown_max_wm2"] == 0.0
