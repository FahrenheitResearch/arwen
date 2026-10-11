"""The TEMPO column contract must carry every moment and keep them out of mass."""
from gpuwm.core.mpas_column_batch import (
    _ColumnBatchState, _SPECIES_BY_SCHEME, _MP_PHYSICS_BY_SCHEME,
    _STATE_RESTART_FIELDS_BY_SCHEME,
)
from gpuwm.config import RunConfig
import numpy as np
import pytest


def test_named_tempo_configuration_keeps_chem_shortcut_inert():
    cfg = RunConfig(nx=2, ny=1, nz=3, dx=3000., dy=3000., ztop=9000.,
                    dt=20., run_seconds=0., mp_physics="tempo")
    assert cfg.mp_physics == 900
    assert not cfg.fire_smoke
    assert _MP_PHYSICS_BY_SCHEME["tempo"] == 900
    from gpuwm.core.mynn_pbl_runtime import mynn_flag_qs
    assert mynn_flag_qs(900) is True
    assert mynn_flag_qs(50) is False


def test_tempo_column_allocates_emissions_and_every_transport_field():
    cfg = RunConfig(nx=2, ny=1, nz=3, dx=3000., dy=3000., ztop=9000.,
                    dt=20., run_seconds=0., mp_physics="tempo")
    names = _SPECIES_BY_SCHEME["tempo"]
    assert len(names) == 13
    state = _ColumnBatchState(cfg, np.ones(3), np.ones(3), 10000.,
                             np.zeros(2), np, species=names)
    for name in names:
        assert getattr(state, name).shape == (3, 1, 2)
    assert state.nwfa2d.shape == (1, 2)
    assert state.tempo_first_call is True
    assert "nwfa2d" in _STATE_RESTART_FIELDS_BY_SCHEME["tempo"]
    assert names[-2:] == ("ng", "volg")


def test_named_convection_cadence_reaches_the_existing_native_slot():
    from gpuwm.core.mpas_column_batch import resolve_column_batch_cadences
    row = resolve_column_batch_cadences(dt=60, radiation_seconds=1800,
        surface_pbl_seconds=60, cumulus_seconds=60, cumulus_scheme="ntiedtke")
    assert row["cu_physics"] == 16
    assert row["stepcu"] == 1
    assert row["cudt_minutes"] == 0.
    with pytest.raises(ValueError, match="requires cumulus_seconds"):
        resolve_column_batch_cadences(dt=60, radiation_seconds=1800,
            surface_pbl_seconds=60, cumulus_seconds=300, cumulus_scheme="ntiedtke")


def test_tempo_logical_options_do_not_become_string_truthiness_or_unread_switches():
    from gpuwm.config import validate_run_config
    cfg = RunConfig(nx=8, ny=8, nz=20, dx=3000., dy=3000., ztop=16000.,
                    dt=20., run_seconds=0., moist=True, mp_physics="tempo", tempo_ml_for_bl_nc="false")
    with pytest.raises(ValueError, match="must be logical"):
        validate_run_config(cfg)
    import dataclasses
    cfg = dataclasses.replace(cfg, tempo_ml_for_bl_nc=True, mp_physics=6)
    with pytest.raises(ValueError, match="does not execute"):
        validate_run_config(cfg)


def test_aerosol_optics_accepts_the_named_tracer_inventory_and_refuses_missing_numbers():
    from gpuwm.config import validate_radiation_driver_options
    import dataclasses
    cfg = RunConfig(nx=8, ny=8, nz=20, dx=3000., dy=3000., ztop=16000.,
        dt=20., run_seconds=0., moist=True, mp_physics="tempo", ra_physics=4,
        ra_rrtmg_variant="rrtmg_legacy", aer_opt=3)
    validate_radiation_driver_options(cfg)
    with pytest.raises(ValueError, match="aerosol numbers"):
        validate_radiation_driver_options(dataclasses.replace(cfg, mp_physics=6))


def test_column_suite_adds_raw_cumulus_momentum_before_edge_reconstruction(monkeypatch):
    from types import SimpleNamespace
    from gpuwm.core.mpas_column_batch import MpasColumnBatchPhysics
    from gpuwm.core.mpas_column_suite import MpasColumnSuitePhysics
    seam = object.__new__(MpasColumnSuitePhysics)
    seam._nz, seam._ncol, seam._cp = 3, 2, np
    seam._out = {k: np.zeros((3, 2), np.float32) for k in ("du", "dv")}
    seam._driver = SimpleNamespace(pbl_raw_rates={
        "du": np.full((3, 2), 2., np.float32),
        "dv": np.full((3, 2), -3., np.float32)})
    seam._momentum_cumulus = SimpleNamespace(last_momentum=(
        np.full((3, 2), 0.5, np.float32), np.full((3, 2), -0.25, np.float32)))
    result = SimpleNamespace(du=seam._out["du"], dv=seam._out["dv"])
    monkeypatch.setattr(MpasColumnBatchPhysics, "run_phase1", lambda self, **kwargs: result)
    assert seam.run_phase1() is result
    np.testing.assert_array_equal(result.du, 2.5)
    np.testing.assert_array_equal(result.dv, -3.25)
    seam._momentum_cumulus = None
    seam.run_phase1()
    np.testing.assert_array_equal(result.du, 2.)
