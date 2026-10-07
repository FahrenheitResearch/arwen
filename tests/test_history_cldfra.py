"""WOOF writes its own radiation cloud fraction (CLDFRA) to history.

The clean-room GRIB2 post draws low/mid/high cloud cover, cloud base, cloud
top and ceiling from CLDFRA.  The field is the cloud fraction the model's
radiation radiated through (icloud=1 cal_cldfra1 plus the MYNN subgrid merge),
held between radiation calls, carried across a restart, and written on by
default for every scheme that computes one.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.io.history_layout import produced_history_shapes
from gpuwm.io.history_selection import HISTORY_PRESETS


def _cfg(**overrides):
    base = RunConfig(nx=32, ny=24, nz=12, dx=3000.0, dy=3000.0,
                     ztop=12000.0, dt=6.0, run_seconds=120.0,
                     moist=True, mp_physics=8, radt=10.0)
    return replace(base, **overrides)


@pytest.mark.parametrize("lw, sw", [(4, 4), (4, 0), (0, 4), (1, 1), (1, 0)])
def test_cldfra_is_in_the_default_history_where_radiation_builds_one(lw, sw):
    shapes = produced_history_shapes(
        _cfg(ra_physics=0, ra_lw_physics=lw, ra_sw_physics=sw))
    assert shapes["CLDFRA"] == (12, 24, 32)


@pytest.mark.parametrize("lw, sw", [(0, 1), (0, 0)])
def test_cldfra_is_absent_where_no_scheme_computes_a_fraction(lw, sw):
    shapes = produced_history_shapes(
        _cfg(ra_physics=0, ra_lw_physics=lw, ra_sw_physics=sw))
    assert "CLDFRA" not in shapes


def test_cldfra_schema_preset_and_restart_classification():
    from gpuwm.io import restart
    from gpuwm.io.wrf_output_schema import (RADIATION_CLDFRA_ATTRIBUTES,
                                            REGISTRY_VAR_META)

    # Registry.EM_COMMON:1699 verbatim.
    assert REGISTRY_VAR_META["CLDFRA"] == ("CLOUD FRACTION", "")
    assert {"source", "sampling", "initial_frame"} <= set(
        RADIATION_CLDFRA_ATTRIBUTES)
    # The post's cloud fields need it in the storm-volume preset too.
    assert "CLDFRA" in HISTORY_PRESETS["severe"]
    assert "CLDFRA" not in HISTORY_PRESETS["minimal"]
    # Carried by the checkpoint like OLR, never rebuilt as zeros.
    assert "cldfra" in restart.DRIVER_CHECKPOINT_ONLY_ATTRS
    assert "cldfra" not in restart.DRIVER_REBUILT_ATTRS
    assert "cldfra" not in restart.DRIVER_SERIALIZED_ATTRS


def test_scheme_declarations():
    from gpuwm.core.radiation_composition import ComposedRadiation
    from gpuwm.core.rrtm_lw import RRTMDudhiaRadiation, RRTMLongwaveRadiation
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
    from gpuwm.core.rrtmgp import RRTMGPRadiation

    for scheme in (RRTMGPRadiation, RRTMGLegacyRadiation,
                   RRTMLongwaveRadiation, RRTMDudhiaRadiation):
        assert scheme.publishes_cldfra is True

    class Silent:
        publishes_cldfra = False

    class Builds:
        publishes_cldfra = True

    assert ComposedRadiation(None, None, None, longwave_adapter=None,
                             shortwave_adapter=Builds()).publishes_cldfra
    assert not ComposedRadiation(None, None, None, longwave_adapter=None,
                                 shortwave_adapter=Silent()).publishes_cldfra


@pytest.mark.gpu
@requires_gpu
def test_driver_holds_publishes_and_requires_the_radiation_cldfra():
    """Zero before the first call, filled in place on each due call and
    held between calls, and a scheme that declares it must return it."""
    import cupy as cp

    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.physics import RadiationResult
    from test_physics_driver import _full_state

    value = {"cloud": 0.25}

    def surface_only(*, atmosphere, fields, state, cfg):
        zeros = cp.zeros_like(state.p)
        surface = cp.full(state.mup.shape, 300.0, cp.float32)
        return RadiationResult(zeros, zeros, surface, surface)

    def with_cloud(*, atmosphere, fields, state, cfg):
        result = surface_only(atmosphere=atmosphere, fields=fields,
                              state=state, cfg=cfg)
        result.cldfra = cp.full(state.p.shape, value["cloud"], cp.float32)
        return result

    with_cloud.publishes_cldfra = True

    def declared_but_silent(**kwargs):
        return surface_only(**kwargs)

    declared_but_silent.publishes_cldfra = True

    # No producer: absent, not a zero field that reads as clear sky.
    state, cfg, driver = _full_state(
        nx=1, ny=1, ra_physics=90, radt_minutes=0.0, radiation=surface_only)
    assert driver.cldfra is None
    assert "CLDFRA" not in driver.output_fields()

    state, cfg, driver = _full_state(
        nx=1, ny=1, ra_physics=90, radt_minutes=0.0, radiation=with_cloud)
    published = driver.output_fields()["CLDFRA"]
    assert published.shape == state.p.shape
    assert published.dtype == cp.float32
    assert float(cp.abs(published).max()) == 0.0
    update_diagnostics(state)
    driver.compute(state, cfg)
    assert driver.output_fields()["CLDFRA"] is published
    np.testing.assert_array_equal(cp.asnumpy(published), np.float32(0.25))

    state, cfg, driver = _full_state(
        nx=1, ny=1, ra_physics=90, radt_minutes=0.0,
        radiation=declared_but_silent)
    update_diagnostics(state)
    with pytest.raises(ValueError, match="publishes_cldfra"):
        driver.compute(state, cfg)
