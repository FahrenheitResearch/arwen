"""Default radiation option echoes restore; old echoes refuse active choices."""
from dataclasses import replace

import pytest

from gpuwm.io import restart
from test_radiation_driver_identity import OPTIONS, _experiment


@pytest.mark.parametrize("name,value", OPTIONS.items())
def test_default_radiation_echo_restores_and_old_header_refuses_active_option(name, value):
    default = _experiment(**{name: 0}).domains[0].run
    written = restart.configuration_echo(default)
    assert name not in written
    restart._require_config_match(written, default, "default-checkpoint.npz")
    active = replace(default, **{name: value})
    with pytest.raises(restart.RestartMismatchError, match=name):
        restart._require_config_match(written, active, "default-checkpoint.npz")
