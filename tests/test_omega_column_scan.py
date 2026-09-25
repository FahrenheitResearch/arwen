"""The per-column Omega kernel against the float32 transcription of calc_ww_cp.

``gpuwm.core.dycore._omega_ref`` evaluates WRF ``calc_ww_cp`` one column
per thread in the Fortran's own arithmetic; ``gpuwm.verify.npref.
np_calc_ww_cp`` is the same loop on NumPy float32 scalars.  The two are
compared as raw 32-bit words, with zero differing words expected, because a
tolerance would hide exactly what this file guards: a product grouped the
other way round, a tree reduction where the Fortran sums sequentially, a
pre-added recurrence operand, or NVRTC contracting a multiply-add.

Shapes are chosen so nothing lines up by accident: an nz that is not a
multiple of 32, a column count that is not a multiple of the 128-thread
block, and one that is exactly a multiple.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.verify.npref import np_calc_ww_cp

F32 = np.float32
EPS32 = float(np.finfo(np.float32).eps)

#: (nz, ny, nx): nz not a multiple of 32; ny*nx off the 128-thread block
#: (55, 153) and exactly on it (128).
SHAPES = ((7, 5, 11), (33, 9, 17), (49, 8, 16))


def _inputs(rng, nz, ny, nx, has_msf):
    """Random fluxes with realistic dynamic range and a closing coordinate.

    ``dnw`` and ``c1h`` are scaled so ``sum(dnw*c1h) = -1`` in float64
    before the float32 cast, which is the property WRF's top condition
    rests on; the cast leaves the closure open by O(nz*eps), which the
    residual bound below allows for.
    """
    def field(shape, lo=-2.0, hi=4.0):
        return (rng.standard_normal(shape)
                * 10.0 ** rng.uniform(lo, hi, shape)).astype(F32)
    ru = field((nz, ny, nx + 1))
    rv = field((nz, ny + 1, nx))
    c1h = rng.uniform(0.5, 1.0, nz)
    dnw = -rng.uniform(0.5, 1.5, nz)
    dnw /= -np.sum(dnw * c1h)
    dnw = dnw.astype(F32)
    c1h = c1h.astype(F32)
    msft = (1.0 + 0.05 * rng.random((ny, nx))).astype(F32) if has_msf else None
    dx, dy = 3000.0, 1250.0
    return ru, rv, dnw, c1h, msft, dx, dy


def _fake_state(nz, ny, nx, dnw, c1h, msft):
    """The slice of ``DomainState`` that ``_omega_ref`` reads."""
    import cupy as cp
    slots = {}

    def scratch(shape, slot, dtype=None):
        buf = slots.get(slot)
        if buf is None:
            buf = slots[slot] = cp.zeros(shape, dtype=np.float32)
        return buf

    return SimpleNamespace(
        p=SimpleNamespace(shape=(nz, ny, nx)), scratch=scratch,
        dnw=cp.asarray(dnw), c1h=cp.asarray(c1h),
        has_msf=msft is not None,
        msft=cp.asarray(msft) if msft is not None else cp.ones((ny, nx), F32))


def _float64_mirror(ru, rv, dnw, c1h, msft, dx, dy):
    """The recurrence in float64, for the loose sanity check and the bound."""
    ru = ru.astype(np.float64)
    rv = rv.astype(np.float64)
    dnw = dnw.astype(np.float64)[:, None, None]
    c1h = c1h.astype(np.float64)[:, None, None]
    nz = ru.shape[0]
    divv = dnw * ((ru[:, :, 1:] - ru[:, :, :-1]) / dx
                  + (rv[:, 1:, :] - rv[:, :-1, :]) / dy)
    if msft is not None:
        divv *= msft.astype(np.float64)[None]
    dmdt = divv.sum(axis=0)
    ww = np.zeros((nz + 1,) + dmdt.shape)
    ww[1:nz] = -np.cumsum((c1h * dnw)[:nz - 1] * dmdt[None] + divv[:nz - 1],
                          axis=0)
    scale = np.maximum.reduce([
        np.abs(dmdt), np.abs(divv).max(axis=0), np.abs(ww).max(axis=0),
        np.abs((c1h * dnw) * dmdt[None]).max(axis=0)])
    return ww, scale


def _assert_top_residual_within_rounding(residual, scale, nz):
    """WRF zeroes ``ww(kte)`` rather than trusting the sum; the sum it
    distrusts must still close to the rounding of the ~3*nz operations
    that formed it, or the transcription is not computing the closure."""
    bound = 8.0 * nz * EPS32 * scale
    assert np.all(np.abs(residual) <= bound), (
        float(np.abs(residual).max()), float(bound.min()))


def test_oracle_refuses_widened_operands():
    """A float64 operand anywhere would silently widen the whole chain."""
    rng = np.random.default_rng(1)
    ru, rv, dnw, c1h, msft, dx, dy = _inputs(rng, 4, 2, 3, True)
    rdx, rdy = F32(1.0) / F32(dx), F32(1.0) / F32(dy)
    np_calc_ww_cp(ru, rv, dnw, c1h, rdx, rdy, msft=msft)
    with pytest.raises(TypeError, match="ru must be float32"):
        np_calc_ww_cp(ru.astype(np.float64), rv, dnw, c1h, rdx, rdy)
    with pytest.raises(TypeError, match="msft must be float32"):
        np_calc_ww_cp(ru, rv, dnw, c1h, rdx, rdy, msft=msft.astype(np.float64))
    with pytest.raises(TypeError, match="rdx must be a numpy float32"):
        np_calc_ww_cp(ru, rv, dnw, c1h, 1.0 / dx, rdy)
    with pytest.raises(ValueError, match="rv shape"):
        np_calc_ww_cp(ru, rv[:, :-1], dnw, c1h, rdx, rdy)


def test_oracle_scalar_arithmetic_rounds_every_operation():
    """The property the transcription rests on: a NumPy float32 scalar
    product is a float32 rounded once, and a following add is a second
    rounding, never fused.  ``a*b`` here lands exactly on a float32
    midpoint, so a fused ``a*b + c`` keeps the 2^-24 a rounded product
    loses; the oracle must lose it."""
    a = F32(1.0) + F32(2.0 ** -12)
    b = F32(1.0) + F32(2.0 ** -12)
    c = -(F32(1.0) + F32(2.0 ** -11))
    assert type(a * b) is np.float32
    assert a * b + c == F32(0.0)
    assert float(a) * float(b) + float(c) == 2.0 ** -24
    assert F32(1.0) + F32(2.0 ** -24) == F32(1.0)


@pytest.mark.parametrize("has_msf", [False, True])
def test_oracle_agrees_with_a_float64_mirror(has_msf):
    """Bit identity between kernel and oracle proves nothing if both are
    the same wrong formula; the oracle is the WRF recurrence to float64
    tolerance as well."""
    rng = np.random.default_rng(7)
    nz, ny, nx = 9, 3, 5
    ru, rv, dnw, c1h, msft, dx, dy = _inputs(rng, nz, ny, nx, has_msf)
    ww = np_calc_ww_cp(ru, rv, dnw, c1h, F32(1.0) / F32(dx),
                       F32(1.0) / F32(dy), msft=msft)
    ww64, _scale = _float64_mirror(ru, rv, dnw, c1h, msft, dx, dy)
    assert ww.dtype == np.float32
    np.testing.assert_allclose(ww[1:nz].astype(np.float64), ww64[1:nz],
                               rtol=1e-4, atol=1e-4 * np.abs(ww64).max())
    assert not ww[0].any() and not ww[nz].any()


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("has_msf", [False, True])
@pytest.mark.parametrize("shape", SHAPES)
def test_omega_ref_is_bit_identical_to_the_oracle(shape, has_msf):
    import cupy as cp
    from gpuwm.core import dycore

    nz, ny, nx = shape
    rng = np.random.default_rng(20260916 + nz)
    ru, rv, dnw, c1h, msft, dx, dy = _inputs(rng, nz, ny, nx, has_msf)
    state = _fake_state(nz, ny, nx, dnw, c1h, msft)
    cfg = SimpleNamespace(dx=dx, dy=dy)
    ww = dycore._omega_ref(state, cfg, cp.asarray(ru), cp.asarray(rv))
    assert ww.shape == (nz + 1, ny, nx) and ww.dtype == np.float32
    got = cp.asnumpy(ww)

    residual = np.empty((ny, nx), dtype=F32)
    expected = np_calc_ww_cp(ru, rv, dnw, c1h, F32(1.0) / F32(dx),
                             F32(1.0) / F32(dy), msft=msft,
                             top_residual=residual)
    differing = int((got.view(np.uint32) != expected.view(np.uint32)).sum())
    assert differing == 0
    assert np.all(got[nz].view(np.uint32) == 0)      # +0.0 exactly, WRF ww(kte)
    assert np.all(got[0].view(np.uint32) == 0)
    assert np.abs(got[1:nz]).max() > 0.0
    _ww64, scale = _float64_mirror(ru, rv, dnw, c1h, msft, dx, dy)
    _assert_top_residual_within_rounding(residual, scale, nz)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("has_msf", [False, True])
@pytest.mark.parametrize("shape", ((13, 5, 9), (49, 6, 24)))
def test_stage_fluxes_omega_is_bit_identical_to_the_oracle(shape, has_msf):
    """The production route: a seeded, balanced state through
    ``stage_fluxes`` (coupled fluxes from the couple-momentum kernel,
    Omega from the column kernel) against the oracle on the same fluxes."""
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.verify.npref import random_acoustic_state

    nz, ny, nx = shape
    s, cfg = random_acoustic_state(seed=41 + nz, nz=nz, ny=ny, nx=nx,
                                   stretch=1.4, hybrid_opt=2,
                                   msf_amp=0.05 if has_msf else 0.0)
    assert s.has_msf is has_msf
    ru, rv, ww = dycore.stage_fluxes(s, cfg)
    got = cp.asnumpy(ww)
    residual = np.empty((ny, nx), dtype=F32)
    msft = cp.asnumpy(s.msft) if has_msf else None
    ru_h, rv_h = cp.asnumpy(ru), cp.asnumpy(rv)
    dnw, c1h = cp.asnumpy(s.dnw), cp.asnumpy(s.c1h)
    expected = np_calc_ww_cp(ru_h, rv_h, dnw, c1h, F32(1.0) / F32(cfg.dx),
                             F32(1.0) / F32(cfg.dy), msft=msft,
                             top_residual=residual)
    differing = int((got.view(np.uint32) != expected.view(np.uint32)).sum())
    assert differing == 0
    assert np.all(got[nz].view(np.uint32) == 0)
    assert np.abs(got[1:nz]).max() > 0.0
    _ww64, scale = _float64_mirror(ru_h, rv_h, dnw, c1h, msft, cfg.dx, cfg.dy)
    _assert_top_residual_within_rounding(residual, scale, nz)


@pytest.mark.gpu
@requires_gpu
def test_omega_ref_reuses_its_scratch_slot():
    """Two calls land in the same ``rk_ww`` buffer, as the stage contract
    (views valid until the next ``stage_fluxes`` call) requires."""
    import cupy as cp
    from gpuwm.core import dycore

    rng = np.random.default_rng(3)
    nz, ny, nx = 6, 3, 4
    ru, rv, dnw, c1h, msft, dx, dy = _inputs(rng, nz, ny, nx, False)
    state = _fake_state(nz, ny, nx, dnw, c1h, msft)
    cfg = SimpleNamespace(dx=dx, dy=dy)
    first = dycore._omega_ref(state, cfg, cp.asarray(ru), cp.asarray(rv))
    second = dycore._omega_ref(state, cfg, cp.asarray(2.0 * ru),
                               cp.asarray(rv))
    assert first.data.ptr == second.data.ptr
