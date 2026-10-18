import pytest
from gpuwm.config import RunConfig
from gpuwm.io import restart


def config():
    return RunConfig(nx=8, ny=8, nz=8, dx=9000, dy=9000,
                     ztop=12000, dt=10, run_seconds=20)


def test_current_writer_echo_is_readable_at_default_lower_boundary():
    cfg = config()
    restart._require_config_match(restart.configuration_echo(cfg), cfg, "current.npz")


def test_current_reader_keeps_active_timestep_identity():
    cfg = config()
    echo = restart.configuration_echo(cfg)
    echo["dt"] = cfg.dt * 2
    with pytest.raises(restart.RestartMismatchError, match="dt"):
        restart._require_config_match(echo, cfg, "different.npz")
