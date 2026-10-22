"""The HRRR v4.1.21 fork's smoke processes on the GPU vertical mixing.

enh_vermix (smoke/dry_dep_driver.F:178-195): FRP > 1 MW raises ekmfull at
kts+1..kts+4 to MAX(1.1*ekm, 10); diurnal FRP (coef_bb_dc * FRP) > 250 MW
then raises kts+1..kts+5 to MAX(1.1*ekm, 20).  aer_drydep_opt = 111
(:244-256): vd = MIN(0.5, ust/500 * unstable and humid corrections).
"""
import numpy as np
import pytest

from gpuwm.core.chem_vertmx import launch_vertmx, smoke_deposition_velocity
from gpuwm.verify.chem_oracle import ORACLE_ROOT, load

pytestmark = pytest.mark.gpu
CASES = load(ORACLE_ROOT / 'core' / 'vertmx')


def test_enh_vermix_raises_the_mixing_over_burning_columns_only():
    import cupy as cp
    a = CASES[sorted(CASES)[0]]
    nz = a['alt'].size
    ncol = 3

    def col(key):
        return cp.asarray(np.repeat(a[key][:, None, None], ncol, axis=2))
    chem = cp.asarray(np.repeat(a['input'][:, 1, None, None], ncol, axis=2))
    args = (chem, col('alt'), col('z_at_w'), col('z'), col('dz8w'), col('exch_h'),
            cp.zeros((1, ncol), cp.float32), a['dt'])
    ek = cp.empty((nz + 1, 1, ncol), cp.float32)
    mixed = cp.empty_like(chem)
    dd = cp.empty((1, ncol), cp.float32)
    accum = cp.zeros((1, ncol), cp.float32)
    frp = cp.asarray([[0.5, 5.0, 300.0]], cp.float32)
    coef = cp.ones((1, ncol), cp.float32)
    launch_vertmx(*args, accum, phase='aerosol', mixed=mixed, ekmfull=ek,
                  ddmassn=dd, enh_frp=frp, enh_coef=coef)
    got = cp.asnumpy(ek)[:, 0, :]
    base = np.maximum(np.float32(1e-6), a['exch_h'].astype(np.float32))
    base = np.concatenate([[0.0], base[1:nz], [0.0]]).astype(np.float32)
    # No CO, no urban: the stock floors leave max(1e-6, exch_h).
    np.testing.assert_array_equal(got[:, 0].view(np.uint32), base.view(np.uint32))
    one = base.copy()
    one[1:5] = np.maximum(np.float32(1.1) * one[1:5], np.float32(10))
    np.testing.assert_array_equal(got[:, 1].view(np.uint32), one.view(np.uint32))
    two = one.copy()
    two[1:6] = np.maximum(np.float32(1.1) * two[1:6], np.float32(20))
    np.testing.assert_array_equal(got[:, 2].view(np.uint32), two.view(np.uint32))
    np.testing.assert_array_equal(got[6:, 2], base[6:])


def test_smoke_deposition_velocity_matches_the_fork():
    import cupy as cp
    ust = np.array([[0.3, 0.3, 0.6, 400.0]], np.float32)
    rmol = np.array([[0.01, -0.02, -0.02, 0.0]], np.float32)
    t = np.full((2, 1, 4), 290.0, np.float32)
    p = np.full((2, 1, 4), 95000.0, np.float32)
    qv = np.full((2, 1, 4), 0.004, np.float32)
    qv[0, 0, 2] = 0.0118   # near saturation: the humid correction applies
    got = cp.asnumpy(smoke_deposition_velocity(
        cp.asarray(ust), cp.asarray(rmol), cp.asarray(qv), cp.asarray(t), cp.asarray(p)))

    def ref(u, r, q, tt, pp):
        rh = max(.1, min(.95, q / (3.80 * np.exp(17.27 * (tt - 273.) / (tt - 36.)) / (.01 * pp))))
        v = u / 500.
        if r < 0:
            v *= 1. + (-300. * r) ** 0.66667
        if rh > .8:
            v *= 1. + 0.37 * np.exp((rh - .8) / .2)
        return min(.5, v)
    want = [ref(float(ust[0, i]), float(rmol[0, i]), float(qv[0, 0, i]),
                290.0, 95000.0) for i in range(4)]
    np.testing.assert_allclose(got[0], want, rtol=2e-6)
    assert got[0, 3] == np.float32(0.5)
    assert got[0, 2] > got[0, 1] > got[0, 0]
