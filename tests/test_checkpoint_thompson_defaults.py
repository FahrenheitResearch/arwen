"""Thompson default omissions restore and generation changes are refused."""
from dataclasses import replace

import pytest

from gpuwm.io import restart
from test_restart import _cfg


@pytest.mark.parametrize("snow_fall", ("blend", "wrf_39_noaa"))
def test_default_thompson_echo_restores_and_refuses_a_fork_generation(snow_fall):
    cfg = _cfg(mp_physics=28, moist=True)
    written = restart.configuration_echo(cfg)
    assert "thompson_version" not in written
    assert "thompson_fork_snow_fall" not in written
    restart._require_config_match(written, cfg, "default-checkpoint.npz")
    active = replace(cfg, thompson_version="wrf_39_noaa",
                     thompson_fork_snow_fall=snow_fall)
    with pytest.raises(restart.RestartMismatchError, match="thompson_version"):
        restart._require_config_match(written, active, "default-checkpoint.npz")
    moved = restart.configuration_echo(active)
    restart._require_config_match(moved, active, "fork-checkpoint.npz")


def test_fork_blend_omission_restores_and_refuses_a_changed_snow_fall():
    cfg = _cfg(mp_physics=28, moist=True, thompson_version="wrf_39_noaa")
    written = restart.configuration_echo(cfg)
    assert written["thompson_version"] == "wrf_39_noaa"
    assert "thompson_fork_snow_fall" not in written
    restart._require_config_match(written, cfg, "fork-blend-checkpoint.npz")
    changed = replace(cfg, thompson_fork_snow_fall="wrf_39_noaa")
    with pytest.raises(restart.RestartMismatchError, match="thompson_fork_snow_fall"):
        restart._require_config_match(written, changed, "fork-blend-checkpoint.npz")
