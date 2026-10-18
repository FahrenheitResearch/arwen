from gpuwm.config import RunConfig
from gpuwm.prepared_single_domain_forecast import _non_physics_descriptor_sha256
from tools.hrrr_single_domain_benchmark import _forward_profile_switches

KEYS = ('oml_gamma', 'oml_hml0', 'oml_relaxation_time', 'sf_ocean_physics')

def config():
    return RunConfig(nx=20, ny=20, nz=32, dx=12000., dy=12000., ztop=20000., dt=60., run_seconds=60.)

def test_new_parameters_reach_the_shared_descriptor():
    cfg = config()
    values = {key: getattr(cfg, key) for key in KEYS}
    raw = {"shared": {}, "domain": [{"nx": 20, "ny": 20}]}
    _forward_profile_switches(raw, values)
    assert raw["shared"] == values
    assert raw["domain"] == [{"nx": 20, "ny": 20}]

def test_new_physics_preserves_non_physics_descriptor_controls():
    raw = {"shared": {"nz": 32}, "domain": [{"nx": 20, "ny": 20}]}
    before = _non_physics_descriptor_sha256(raw)
    raw["shared"].update({key: getattr(config(), key) for key in KEYS})
    assert _non_physics_descriptor_sha256(raw) == before


def test_prepared_protocol_without_an_ocean_selector_keeps_the_off_default():
    from types import SimpleNamespace
    from gpuwm.surface_ocean import ocean_input_fields
    assert ocean_input_fields(SimpleNamespace(sf_surface_physics=2), {}) == {}
