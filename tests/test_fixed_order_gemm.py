"""The LETKF's ensemble-space products give the same bytes on every card.

THE BREAKAGE THIS PREVENTS
--------------------------
One engine, one seed, two cards: a storm-scale cycle read footprint rain 0.31
on an RTX 5090 and 0.20 on an RTX 5070 Ti from byte-identical model legs,
which read for a day as a card effect on the physics.  The einsum
contractions of the LETKF transform reach cuBLAS through ``cupy.matmul`` as
matrix-vector shapes, and cuBLAS sums those in an order it picks per card
(49 to 82 % of the values differ between the two cards, median 512 ulp of
cancelled sums), so the first analysis differed from card to card in the last
bit of about 0.3 % of its increment values and the cycle grew that into a
different storm.

``gpuwm.da.fixed_order_gemm`` forms every output element as one thread's
sequential fused multiply-add in index order.  What is asserted here is the
property that makes it card-independent BY CONSTRUCTION: the device result is
bit for bit the correctly rounded fused multiply-add chain computed exactly on
the host.  Any card that matches that reference matches every other card.

The CPU tests import no cupy and run in the ``-m "not gpu"`` tier; the
device tests skip where no card is visible.
"""

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from gpuwm.da.letkf import MATMUL_MODES, LetkfConfig, LetkfError, Localization


def _fma(a: float, b: float, c: float) -> float:
    """IEEE fused multiply-add in float64, exactly (math.fma where Python has it)."""
    if hasattr(math, "fma"):
        return math.fma(a, b, c)
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def _reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    g, m, k = a.shape
    n = b.shape[2]
    out = np.empty((g, m, n), dtype=np.float64)
    for gi in range(g):
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for p in range(k):
                    acc = _fma(float(a[gi, i, p]), float(b[gi, p, j]), acc)
                out[gi, i, j] = acc
    return out


def test_the_modes_are_the_two_documented_ones_and_the_default_is_fixed_order():
    assert MATMUL_MODES == ("fixed-order", "library")
    cfg = LetkfConfig(localization=Localization(horizontal_m=3500.0, vertical_m=1500.0),
                      analysis_fields=("theta",), rtps_alpha=0.6)
    assert cfg.matmul == "fixed-order"


def test_an_unknown_mode_is_refused_at_construction():
    with pytest.raises(LetkfError, match="matmul must be one of"):
        LetkfConfig(localization=Localization(horizontal_m=3500.0, vertical_m=1500.0),
                    analysis_fields=("theta",), rtps_alpha=0.6, matmul="cublas")


def _cupy():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device visible")
    except Exception as exc:  # no driver, or CUDA_VISIBLE_DEVICES=""
        pytest.skip(f"no CUDA device visible: {exc}")
    return cp


def test_the_device_product_is_the_exact_fused_chain_bit_for_bit():
    cp = _cupy()
    from gpuwm.da.fixed_order_gemm import bgemm

    rng = np.random.default_rng(20260925)
    a = rng.standard_normal((5, 3, 17))
    b = rng.standard_normal((5, 17, 4))
    got = cp.asnumpy(bgemm(cp.asarray(a), cp.asarray(b)))
    assert got.tobytes() == _reference(a, b).tobytes()
    # transposed and broadcast views, as the transform hands them over
    e = rng.standard_normal((4, 6, 6))
    got_t = cp.asnumpy(bgemm(cp.asarray(e), cp.swapaxes(cp.asarray(e), 1, 2)))
    assert got_t.tobytes() == _reference(e, np.swapaxes(e, 1, 2)).tobytes()
    v = rng.standard_normal((4, 6))
    got_v = cp.asnumpy(bgemm(cp.asarray(e), cp.asarray(v)[:, :, None]))
    assert got_v.tobytes() == _reference(e, v[:, :, None]).tobytes()


def test_the_four_contractions_match_numpy_to_rounding():
    cp = _cupy()
    from gpuwm.da.fixed_order_gemm import einsum_fixed_order

    rng = np.random.default_rng(7)
    g, r, p = 50, 10, 40
    cmat, dvec = rng.standard_normal((g, r, p)), rng.standard_normal((g, p))
    pa, wbar = rng.standard_normal((g, r, r)), rng.standard_normal((g, r))
    xbg = rng.standard_normal((r, g))
    cases = [("grp,gp->gr", cmat, dvec), ("grs,gs->gr", pa, wbar),
             ("mg,gm->g", xbg, wbar), ("mg,gmk->kg", xbg, pa)]
    for spec, x, y in cases:
        got = cp.asnumpy(einsum_fixed_order(spec, cp.asarray(x), cp.asarray(y)))
        want = np.einsum(spec, x, y)
        assert got.shape == want.shape, spec
        assert float(np.abs(got - want).max()) <= 1e-12 * max(1.0, float(np.abs(want).max())), spec
    with pytest.raises(ValueError, match="no fixed-order mapping"):
        einsum_fixed_order("ij,jk->ik", cp.asarray(xbg), cp.asarray(xbg.T))


def test_the_product_does_not_depend_on_where_a_matrix_sits_in_the_batch():
    """The chunk partition cannot move a gridpoint's answer (cuBLAS's could by an ulp)."""
    cp = _cupy()
    from gpuwm.da.fixed_order_gemm import bgemm

    rng = np.random.default_rng(3)
    a = cp.asarray(rng.standard_normal((600, 10, 200)))
    b = cp.asarray(rng.standard_normal((600, 200, 10)))
    whole = cp.asnumpy(bgemm(a, b))
    parts = np.concatenate([cp.asnumpy(bgemm(a[s:e], b[s:e]))
                            for s, e in ((0, 37), (37, 290), (290, 600))])
    assert whole.tobytes() == parts.tobytes()


def _tiny_case(members=6, nz=3, ny=6, nx=6, seed=11, fields=("theta", "u")):
    from gpuwm.da.letkf import GriddedObs, GridGeometry

    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)
    grid = GridGeometry(dx_m=1000.0, dy_m=1000.0,
                        heights_m=np.array([250.0, 800.0, 1600.0][:nz]))
    prior = {f: rng.standard_normal((members,) + shape) + 5.0 * i
             for i, f in enumerate(fields)}
    mask = np.zeros(shape, dtype=bool)
    mask.reshape(-1)[rng.choice(nz * ny * nx, size=10, replace=False)] = True
    sim = prior[fields[0]] * 0.8 + 0.1
    values = np.where(mask, sim.mean(axis=0) + rng.standard_normal(shape) * 0.4,
                      np.nan)
    obs = GriddedObs(name="probe", values=values, errors=0.4, simulated=sim,
                     mask=mask)
    return grid, prior, obs, fields


@pytest.mark.parametrize("matmul", MATMUL_MODES)
def test_the_host_analysis_records_numpy_and_does_not_depend_on_the_chunk_split(matmul):
    """On the host both modes are numpy's own per-matrix products: the receipt
    says so, and a gridpoint's increment is the same bytes whichever chunk it
    sat in (the property the device kernel gives the card path)."""
    from gpuwm.da.letkf import LetkfDiagnostics, analyze

    grid, prior, obs, fields = _tiny_case()
    loc = Localization(horizontal_m=3000.0, vertical_m=1500.0)
    runs = []
    for chunk in (1, 13, 108):
        d = LetkfDiagnostics()
        runs.append(analyze(prior, [obs], grid, LetkfConfig(
            localization=loc, analysis_fields=fields, rtps_alpha=0.3,
            chunk_points=chunk, matmul=matmul), diagnostics=d))
        assert d.matmul == "numpy"
    for f in fields:
        assert runs[0][f].tobytes() == runs[1][f].tobytes() == runs[2][f].tobytes(), f


def test_the_device_analysis_takes_the_fixed_order_route_by_default():
    """End to end on a card: the default route is the fixed-order kernel, the
    receipt names it, and the increments agree with the host reference to
    rounding.  (The eigensolver's batching is not part of this contract; the
    products' split invariance is held by the bgemm test above.)"""
    cp = _cupy()
    from gpuwm.da.letkf import LetkfDiagnostics, analyze

    grid, prior, obs, fields = _tiny_case()
    loc = Localization(horizontal_m=3000.0, vertical_m=1500.0)
    host = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.3))
    d = LetkfDiagnostics()
    dev = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.3),
        diagnostics=d, solve_namespace=cp)
    assert d.matmul == "fixed-order"
    for f in fields:
        got = np.asarray(cp.asnumpy(dev[f]) if hasattr(dev[f], "get") else dev[f])
        assert np.allclose(got, host[f], rtol=0.0, atol=1e-9), f
