"""A real PBL-top eruption: entrainment must not overdraw its vapor donor."""
from pathlib import Path
import numpy as np
import pytest
from conftest import requires_gpu

DATA = Path(__file__).parent / 'data/ysu_longstep/column.npz'

@requires_gpu
@pytest.mark.parametrize('dt', [300., 600., 1200.])
def test_explicit_entrainment_does_not_overdraw_real_donor(dt):
    import cupy as cp
    from gpuwm.core.ysu import launch_ysu
    from gpuwm.wrf_exact import ENABLED
    if ENABLED:
        pytest.skip('strict reproduction deliberately preserves upstream entrainment')
    with np.load(DATA) as data:
        args = {k: cp.asarray(data[k]) for k in data
                if not k.startswith('out_') and data[k].ndim}
        args.update(dt=dt, ysu_version='wrf_480', ysu_topdown_pblmix=1)
        out = launch_ysu(**args)
        q = data['qv'].astype(np.float64)
        rate = cp.asnumpy(out['dqv']).astype(np.float64)
        mass = -np.diff(data['p_interface'].astype(np.float64), axis=0)
        evolved = q + 2*dt*rate
        # A conservative mixing operator cannot create a wetter layer than
        # its wettest donor plus the surface's maximum possible increment.
        surface_increment = 2*dt*float(data['qfx'].item())*9.81/mass[0].item()
        assert evolved.min() >= q.min()-2e-8
        assert evolved.max() <= q.max()+surface_increment+2e-8
        # Paired interface fluxes cancel in pressure mass, regardless of
        # their limiter; only the prescribed surface source remains.
        assert abs(float((mass*rate).sum())-float(data['qfx'].item())*9.81) < 2e-6
        # The uncorrected captured call violates the same physical bound.
        if dt == 600.:
            old = q + 2*dt*data['out_dqv']
            assert old.max() > q.max()+surface_increment+0.01

@requires_gpu
def test_strict_reproduction_retains_captured_upstream_words():
    import cupy as cp
    from gpuwm.core.ysu import launch_ysu
    from gpuwm.wrf_exact import ENABLED
    if not ENABLED:
        pytest.skip('run again under GPUWM_WRF_EXACT=1')
    with np.load(DATA) as data:
        args = {k: cp.asarray(data[k]) for k in data
                if not k.startswith('out_') and data[k].ndim}
        out = launch_ysu(**args, dt=600., ysu_version='wrf_480', ysu_topdown_pblmix=1)
        for name, value in out.items():
            np.testing.assert_array_equal(cp.asnumpy(value).view(np.uint32),
                                          data['out_'+name].view(np.uint32))


def test_corrected_and_strict_restart_identities_are_distinct():
    from gpuwm.ysu_generations import ysu_algorithm_identity
    assert ysu_algorithm_identity('wrf_480') != ysu_algorithm_identity('wrf_480', strict=True)
    assert ysu_algorithm_identity('wrf_480', strict=True) == 'ysu-wrf-v4.8.0-topdown-cap-noftz-v1'
