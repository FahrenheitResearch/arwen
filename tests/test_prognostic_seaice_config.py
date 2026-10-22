from types import SimpleNamespace
import pytest
from gpuwm.ocean.seaice import validate

def config(**kw):
    d=dict(seaice_prognostic=True, seaice_initial_thickness=1.,
           seaice_new_ice_thickness=.5, seaice_ocean_heat_flux=2., sf_surface_physics=2)
    return SimpleNamespace(**(d|kw))

@pytest.mark.parametrize('key,value', [('seaice_initial_thickness',0),
    ('seaice_new_ice_thickness',float('nan')),('seaice_ocean_heat_flux',float('inf')),
    ('seaice_prognostic',1)])
def test_invalid_physical_parameters(key,value):
    with pytest.raises(ValueError,match=key): validate(config(**{key:value}))



def test_ice_state_survives_the_real_restart_manifest(monkeypatch, tmp_path):
    import numpy as np
    from gpuwm.io import restart
    from gpuwm.ocean.seaice import STATE, OUTPUT
    from test_restart import _cfg, _shim_driver_state, _fill_setup
    cfg = _cfg()
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    names = set(STATE) | set(OUTPUT)
    for i, name in enumerate(sorted(names)):
        driver.fields[name] = np.full(state.mup.shape, i + .125, np.float32)
    path = restart.write_restart(tmp_path / "ice.npz", state, cfg)
    fresh, fresh_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(fresh)
    for name in names:
        fresh_driver.fields[name] = np.zeros(state.mup.shape, np.float32)
    restart.restore_restart(path, fresh, cfg)
    for name in names:
        assert driver.fields[name].tobytes() == fresh_driver.fields[name].tobytes(), name


def test_noah_prescribed_solver_keeps_lake_ice(monkeypatch):
    import numpy as np
    import gpuwm.core.physics as physics
    import gpuwm.core.noah as noah
    observed = {}
    def launch(fields, *args, **kwargs):
        observed.update(fields)
    monkeypatch.setattr(physics, "cp", np)
    monkeypatch.setattr(noah, "launch_noah_seaice", launch)
    fields = dict(xice=np.array([[.8,.7]], np.float32),
                  ice_ocean_mask=np.array([[1.,0.]], np.float32), ice_volume=True)
    driver = SimpleNamespace(fields=fields, noah_seaice_dt=60., noah_seaice_albedo_default=.65)
    physics.PhysicsDriver._run_noah_seaice(driver, None, False, xice_threshold=.5)
    np.testing.assert_array_equal(observed['xice'], [[0.,np.float32(.7)]])
    np.testing.assert_array_equal(fields['xice'], [[np.float32(.8),np.float32(.7)]])
