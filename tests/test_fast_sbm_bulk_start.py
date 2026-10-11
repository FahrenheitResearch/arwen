"""FAST_SBM bulk-to-bin start: every category's mass lands in its bins exactly.

Breakage prevented: an HRRR/RAP start on mp_physics=30 either refused (no
spectra) or, if the bins were left empty, silently lost the analysis' cloud
water. These pin mass closure per cell, the category-to-group routing and a
physically placed spectrum (cloud mode near 10 um, rain mode in mm sizes).
"""
import numpy as np
import pytest

from gpuwm.core import fast_sbm_bulk_start as b


def _case(n=7):
    rng = np.random.default_rng(3)
    shape = (n, 4, 5)
    bulk = {k: np.where(rng.random(shape) < 0.6, 10 ** rng.uniform(-9, -2.5, shape), 0.0)
            for k in ("qc", "qr", "qi", "qs", "qg")}
    bulk["qc"][0, 0, 0] = 1e-14  # below threshold: ignored
    rho = rng.uniform(0.4, 1.2, shape)
    xland = np.where(rng.random((1, 4, 5)) < 0.5, 1.0, 2.0)
    return bulk, rho, xland


def test_mass_closes_per_cell_and_category():
    bulk, rho, xland = _case()
    out = b.bins_from_bulk(np, bulk, rho, xland)
    assert out.shape == (99,) + rho.shape
    assert np.all(out >= 0)
    liquid = np.where(bulk["qc"] > b.THRESHOLD, bulk["qc"], 0) + np.where(bulk["qr"] > b.THRESHOLD, bulk["qr"], 0)
    snow = np.where(bulk["qi"] > b.THRESHOLD, bulk["qi"], 0) + np.where(bulk["qs"] > b.THRESHOLD, bulk["qs"], 0)
    dense = np.where(bulk["qg"] > b.THRESHOLD, bulk["qg"], 0)
    np.testing.assert_allclose(out[:33].sum(0), liquid, rtol=1e-12, atol=0)
    np.testing.assert_allclose(out[33:66].sum(0), snow, rtol=1e-12, atol=0)
    np.testing.assert_allclose(out[66:].sum(0), dense, rtol=1e-12, atol=0)


def test_spectra_are_placed_physically():
    x = b.mass_grid_kg()
    dc, _, _ = b.diameter_edges_m(x)
    shape = (1,)
    one = lambda **kw: {k: np.full(shape, kw.get(k, 0.0)) for k in ("qc", "qr", "qi", "qs", "qg")}
    rho = np.full(shape, 1.0)
    cloud = b.bins_from_bulk(np, one(qc=3e-4), rho, np.ones(shape))[:33, 0]
    # 0.3 g/kg at 100 cm-3 -> mean volume diameter ~ 18 um
    dmode = dc[np.argmax(cloud)]
    assert 8e-6 < dmode < 40e-6, dmode
    rain = b.bins_from_bulk(np, one(qr=1e-3), rho, np.ones(shape))[:33, 0]
    assert 0.5e-3 < dc[np.argmax(rain)] < 4e-3
    graup = b.bins_from_bulk(np, one(qg=1e-3), rho, np.ones(shape))
    assert graup[:66].sum() == 0 and graup[66:].sum() == pytest.approx(1e-3, rel=1e-12)


def test_off_grid_mass_is_kept():
    shape = (2,)
    bulk = {k: np.zeros(shape) for k in ("qc", "qr", "qi", "qs", "qg")}
    bulk["qr"][:] = [1e-12, 0.05]   # vanishing and enormous rain
    out = b.bins_from_bulk(np, bulk, np.ones(shape), np.ones(shape))
    np.testing.assert_allclose(out[:33].sum(0), bulk["qr"], rtol=1e-12)


def test_frozen_fraction_never_exceeds_one():
    """SR from FAST_SBM's three accumulators stays in [0, 1] in FP32.

    Breakage prevented: an all-snow column whose snow+graupel sum rounds one
    ulp above RAINNCV stopped every winter mp30 run at its first snowing step.
    """
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001
        pytest.skip("no CUDA device")
    from gpuwm.core.fast_sbm_runtime import frozen_fraction
    rng = np.random.default_rng(5)
    s = rng.uniform(0, 1e-3, 200000).astype(np.float32)
    g = rng.uniform(0, 1e-3, 200000).astype(np.float32)
    total = np.float32(0) + s
    total = (total + g).astype(np.float32)  # same values, other order: ulp ties
    total = np.nextafter(total, np.float32(0)).astype(np.float32)  # force sum > total
    sr = cp.asnumpy(frozen_fraction(cp.asarray(total), cp.asarray(s), cp.asarray(g)))
    assert sr.max() <= 1.0 and sr.min() >= 0.0
    mixed = cp.asnumpy(frozen_fraction(cp.asarray(np.float32([2e-3, 0.0])), cp.asarray(np.float32([5e-4, 0.0])),
                                       cp.asarray(np.float32([5e-4, 0.0]))))
    assert mixed[0] == (np.float32(5e-4) + np.float32(5e-4)) / np.float32(2e-3) and mixed[1] == 0
