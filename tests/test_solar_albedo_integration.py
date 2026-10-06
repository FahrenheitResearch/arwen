"""Live RUC ownership, interpolation and optional history field contracts."""
from dataclasses import replace
from types import SimpleNamespace
from datetime import datetime

import numpy as np
import pytest


def test_default_history_inventory_keeps_solar_fields_absent():
    from gpuwm.config import RunConfig
    from gpuwm.core.preflight import physics_field_names_2d
    from gpuwm.io.history_layout import produced_history_shapes
    cfg = RunConfig(nx=6, ny=4, nz=5, dx=3000., dy=3000., ztop=20000.,
                    dt=20., run_seconds=0.)
    assert "albsol" not in physics_field_names_2d(cfg)
    assert "ALBSOL" not in produced_history_shapes(cfg)
    enabled = replace(cfg, alb_sol=1, ra_lw_physics=4, ra_sw_physics=4)
    assert {"albsol", "albbcksol"} <= set(physics_field_names_2d(enabled))
    assert {"ALBSOL", "ALBBCKSOL"} <= set(produced_history_shapes(enabled))


@pytest.mark.gpu
@pytest.mark.parametrize("fractional", [0, 1])
def test_ruc_consumes_solar_arrays_and_keeps_original_albedo(fractional):
    import cupy as cp
    from test_ruc_lake_runtime import _config, _build
    from gpuwm.core.physics import _prepare_atmosphere
    cfg = replace(_config(nx=24, ny=20), fractional_seaice=fractional)
    state, driver = _build(cfg)
    control, control_driver = _build(cfg)
    # Exercise the actual fused sea-ice pre/post blend through the aliases,
    # including either side of both admitted thresholds.
    for current in (driver, control_driver):
        current.fields["xice"][0, :6] = cp.asarray(
            [.019, .02, .499, .5, .9, 1.], dtype=cp.float32)
    fields = driver.fields
    original = {name: fields[name].copy() for name in ("albedo", "albbck")}
    fields["albsol"] = cp.full_like(fields["albedo"], np.float32(.42))
    fields["albbcksol"] = cp.full_like(fields["albbck"], np.float32(.42))
    ice = fields["xice"] >= np.float32(driver.ruc_params.xice_threshold)
    fields["albsol"][ice] = (np.float32(.42) * fields["xice"][ice]
        + (np.float32(1.) - fields["xice"][ice]) * np.float32(.08))
    control_driver.fields["albedo"][...] = fields["albsol"]
    control_driver.fields["albbck"][...] = fields["albbcksol"]
    driver._run_ruc(_prepare_atmosphere(state), replace(cfg, alb_sol=1), 1)
    control_driver._run_ruc(_prepare_atmosphere(control), cfg, 1)
    for name in original:
        np.testing.assert_array_equal(cp.asnumpy(fields[name]), cp.asnumpy(original[name]))
    for name in ("tsk", "lh", "hfx", "q2", "t2", "smois", "tslb"):
        np.testing.assert_array_equal(cp.asnumpy(fields[name]), cp.asnumpy(control_driver.fields[name]))
    np.testing.assert_array_equal(cp.asnumpy(fields["albsol"]),
                                  cp.asnumpy(control_driver.fields["albedo"]))


@pytest.mark.gpu
def test_swint_reads_live_solar_albedo_between_radiation_calls():
    import cupy as cp
    from gpuwm.core.swint import ShortwaveInterpolation, STATE_FIELDS
    shape = (1, 4)
    carrier = ShortwaveInterpolation(start_time=datetime(2026, 10, 2, 12),
        latitude_deg=np.full(shape, 38., np.float32),
        longitude_deg=np.full(shape, -100., np.float32), shape=shape)
    fields = {name: cp.zeros(shape, cp.float32)
              for name in (*STATE_FIELDS, "gsw", "swdown")}
    fields["albedo"] = cp.full(shape, .2, cp.float32)
    fields["albsol"] = cp.full(shape, .4, cp.float32)
    fields["swdown"][...] = 500.
    cosine = carrier.coszen_loc(7200.).reshape(shape).copy()
    carrier.radiation_step(fields, coszen=cosine,
        swddir=cp.full(shape, 300., cp.float32), elapsed_seconds=7200.,
        albedo=fields["albsol"])
    assert bool((fields["gsw"] > 0.).all())
    fields["albsol"][...] = np.float32(.6)
    carrier.between_calls(fields, elapsed_seconds=7220.)
    expected = fields["swdown"] * np.float32(1. - np.float32(.6))
    np.testing.assert_array_equal(cp.asnumpy(fields["gsw"]), cp.asnumpy(expected))
    np.testing.assert_array_equal(cp.asnumpy(fields["swint_albedo"]),
                                  np.full(shape, .4, np.float32))
