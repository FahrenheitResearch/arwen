"""Fire off must preserve existing checkpoint identity; fire on binds setup."""
from dataclasses import asdict, replace

import pytest

from gpuwm.config import RunConfig
from gpuwm.io import restart
from gpuwm.sfire_config import FIRE_CONFIG_FIELDS


def test_inactive_fire_preserves_prior_configuration_and_resume():
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=100.0, dy=100.0, dt=0.5,
                    ztop=4000.0, run_seconds=2.0)
    earlier = asdict(cfg)
    for name in FIRE_CONFIG_FIELDS:
        earlier.pop(name)
    assert restart.configuration_echo(cfg) == restart.configuration_echo(
        replace(cfg, fire_num_ignitions=3, fire_static="unused.npz"))
    assert not set(FIRE_CONFIG_FIELDS) & restart.configuration_echo(cfg).keys()
    restart._require_config_match(earlier, cfg, "prior-fire-off-checkpoint")


def test_active_fire_keeps_ignition_and_refinement_in_identity():
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=100.0, dy=100.0, dt=0.5,
                    ztop=4000.0, run_seconds=2.0,
                    moist=True, ifire=2, sr_x=4, sr_y=3)
    echo = restart.configuration_echo(cfg)
    assert set(FIRE_CONFIG_FIELDS) <= echo.keys()
    with pytest.raises(restart.RestartMismatchError, match="sr_x"):
        restart._require_config_match(echo, replace(cfg, sr_x=5), "active-fire-checkpoint")
