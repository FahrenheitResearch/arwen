"""Keep WRF's two scalar paths distinct and new MYNN choices reachable."""

from dataclasses import asdict

import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.io.restart import (
    RestartMismatchError, _drop_default_off_run_keys, _require_config_match,
)


def _config(**overrides):
    values = dict(nx=8, ny=8, nz=50, dx=3000.0, dy=3000.0,
                  dt=15.0, run_seconds=300.0, ztop=20000.0,
                  moist=True, mp_physics=28, bl_pbl_physics=5,
                  sf_sfclay_physics=5, sf_surface_physics=3,
                  num_soil_layers=9, bldt=0.0)
    values.update(overrides)
    return RunConfig(**values)


@pytest.mark.parametrize("length", [1, 2])
@pytest.mark.parametrize("scalar", [0, 1])
def test_supported_mynn_options_reach_config(length, scalar):
    cfg = _config(bl_mynn_mixlength=length, scalar_pblmix=scalar)
    validate_run_config(cfg)
    assert cfg.bl_mynn_mixlength == length
    assert cfg.scalar_pblmix == scalar
    assert cfg.bl_mynn_mixscalars == 0


@pytest.mark.parametrize("value", [0, 3, True, 2.0])
def test_unimplemented_mixing_length_is_never_substituted(value):
    with pytest.raises(ValueError, match="bl_mynn_mixlength"):
        validate_run_config(_config(bl_mynn_mixlength=value))


@pytest.mark.parametrize("value", [-1, 2, True, 1.0])
def test_scalar_selector_is_an_integer_switch(value):
    with pytest.raises(ValueError, match="scalar_pblmix"):
        validate_run_config(_config(scalar_pblmix=value))


@pytest.mark.parametrize("overrides, reason", [
    ({"bl_mynn_mixscalars": 1}, "cannot be combined"),
    ({"mp_physics": 8}, "Thompson aerosol scalar fields"),
    ({"bldt": 1.0}, "restart"),
])
def test_scalar_mixing_refuses_missing_carriers_or_dropped_rates(overrides, reason):
    with pytest.raises((ValueError, NotImplementedError), match=reason):
        validate_run_config(_config(scalar_pblmix=1, **overrides))


def test_default_off_keeps_old_restart_identity_and_on_is_bound():
    old = asdict(_config())
    old.pop("scalar_pblmix")
    off = asdict(_config(scalar_pblmix=0))
    on = asdict(_config(scalar_pblmix=1))
    for values in (old, off, on):
        _drop_default_off_run_keys(values)
    assert old == off
    assert on.pop("scalar_pblmix") == 1
    assert on == off
    # The reader must apply the same absence rule as the writer. Otherwise
    # even a checkpoint written by this build cannot restore with option 0.
    _require_config_match(old, _config(), "off-checkpoint")
    with pytest.raises(RestartMismatchError, match="scalar_pblmix"):
        _require_config_match(old, _config(scalar_pblmix=1), "off-checkpoint")
    _require_config_match(asdict(_config(scalar_pblmix=1)),
                          _config(scalar_pblmix=1), "on-checkpoint")
