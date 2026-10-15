"""The new scheme leaves inactive configuration and checkpoint bytes intact."""
from dataclasses import asdict, replace
from gpuwm.config import RunConfig, drop_inert_g3_fields
from gpuwm.experiment import _public_config_value
from gpuwm.io.restart import configuration_echo


def test_inactive_g3_fields_are_absent_from_public_and_checkpoint_documents():
    cfg = RunConfig(nx=16, ny=16, nz=30, dx=12000., dy=12000., ztop=16000., dt=60., run_seconds=0.)
    assert "cugd_avedx" not in _public_config_value(cfg)
    assert "cugd_avedx" not in configuration_echo(cfg)
    value = asdict(cfg)
    from gpuwm.config import G3_RUN_DEFAULTS
    for key in G3_RUN_DEFAULTS: value.pop(key)
    current = asdict(cfg)
    drop_inert_g3_fields(current)
    assert current == value


def test_g3_fields_bind_the_selected_scheme():
    cfg = RunConfig(nx=16, ny=16, nz=30, dx=12000., dy=12000., ztop=16000., dt=60., run_seconds=0., cu_physics=5)
    assert configuration_echo(cfg)["cugd_avedx"] == 1
    assert _public_config_value(replace(cfg, cugd_avedx=3))["cugd_avedx"] == 3
