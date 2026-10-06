"""Source-form preparation words and non-restart radii on the device."""
from pathlib import Path
import importlib.util

import numpy as np
import pytest

cp = pytest.importorskip('cupy')

spec = importlib.util.spec_from_file_location(
    'source_cloud_device_helpers', Path(__file__).with_name('test_rrtmg_legacy_prep_device.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.mark.parametrize('form', ['wrf_461', 'noaa_wrf39'])
@pytest.mark.parametrize('sw', [False, True])
@pytest.mark.parametrize('ozone', [0, 2])
def test_source_form_matches_statement_rounded_host_prep(form, sw, ozone, monkeypatch):
    from gpuwm.core import rrtmg_legacy_prep as ref
    monkeypatch.setattr(ref, '_PERFWAVE_DEVICE_XP', None)
    kwargs = helpers.side_kwargs(helpers.synthetic(7, nz=12), sw)
    kwargs.update(rrtmg_cloud_optics_form=form, o3input=ozone)
    helpers.dual(kwargs, sw)


def test_source_form_seeds_the_public_non_restart_radii():
    from gpuwm.config import RunConfig
    from gpuwm.core.state import DomainState
    for form, expected in (
            ('wrf_461', (2.49, 4.99, 9.99)),
            ('noaa_wrf39', (2.51, 5.01, 10.01))):
        config = RunConfig(nx=8, ny=8, nz=4, dx=3000, dy=3000, ztop=20000,
                           dt=20, run_seconds=0, moist=True, moist_cq=True,
                           mp_physics=28, ra_physics=4,
                           ra_rrtmg_variant='rrtmg_legacy',
                           rrtmg_cloud_optics_form=form)
        state = DomainState(config)
        for name, value in zip(('effc', 'effi', 'effs'), expected):
            actual = cp.asnumpy(getattr(state, name))
            np.testing.assert_array_equal(actual.view('u4'),
                np.full(actual.shape, value, dtype='f4').view('u4'))
        del state
