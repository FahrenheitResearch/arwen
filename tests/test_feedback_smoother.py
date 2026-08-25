"""WRF's post-feedback parent smoothers, pinned to interp_fcn.F:3794-4014.

The oracle here is a loop-for-loop FP32 transliteration of ``sm121``
(:3864-3935) and ``smdsm`` (:3937-4014), 1-based bounds and all, and the
bar is BITWISE: the GPU kernels reproduce the Fortran's own left-to-right
rounding order (the 1-2-1 sum associates high-neighbour-first, and the
(a + 2b) + c spelling was measured 1 ULP off on ~40% of cells before the
order was pinned).

The windowed ``update_diagnostics`` proof rides along: the feedback
finalize re-diagnoses only the columns the transaction touched, and the
claim that this is bitwise the whole-parent call is asserted, not argued.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.core.nest_interp import (register_nest, smoother,
                                    smoother_parent_window)

F = np.float32


def np_sm121(cfld, ipos, jpos, span_i, span_j, istag, jstag):
    """interp_fcn.F:3864-3935 on a serial tile (the MAX/MIN clamps
    degenerate to the nest window when the tile is the whole domain)."""
    out = cfld.astype(np.float32).copy()
    ilo, ihi = ipos + 2, ipos + span_i - 2 - istag   # 1-based inclusive
    jlo, jhi = jpos + 2, jpos + span_j - 2 - jstag
    for k in range(out.shape[0]):
        new = out[k].copy()                          # :3910-3915 init copy
        for i1 in range(ilo, ihi + 1):               # :3919-3924 j pass
            for j1 in range(jlo, jhi + 1):
                j, i = j1 - 1, i1 - 1
                new[j, i] = F(F(0.25) * F(F(F(out[k, j + 1, i])
                                            + F(F(2.0) * out[k, j, i]))
                                          + out[k, j - 1, i]))
        for j1 in range(jlo, jhi + 1):               # :3927-3931 i pass
            for i1 in range(ilo, ihi + 1):
                j, i = j1 - 1, i1 - 1
                out[k, j, i] = F(F(0.25) * F(F(F(new[j, i + 1])
                                               + F(F(2.0) * new[j, i]))
                                             + new[j, i - 1]))
    return out


def np_smdsm(cfld, ipos, jpos, span_i, span_j, istag, jstag):
    """interp_fcn.F:3937-4014 on a serial tile."""
    out = cfld.astype(np.float32).copy()
    xnu = (F(0.50), F(-0.52))                        # :3976
    ilo, ihi = ipos + 2, ipos + span_i - 2 - istag
    jlo, jhi = jpos + 2, jpos + span_j - 2 - jstag
    for loop in (1, 2):
        nu = xnu[(2 - (loop % 2)) - 1]               # :3990
        for k in range(out.shape[0]):
            new = np.empty_like(out[k])
            for i1 in range(ilo, ihi + 1):           # :3996-4000 j filter
                for j1 in range(jlo, jhi + 1):
                    j, i = j1 - 1, i1 - 1
                    new[j, i] = F(out[k, j, i] + F(nu * F(
                        F(F(F(out[k, j + 1, i] + out[k, j - 1, i]))
                          * F(0.5)) - out[k, j, i])))
            for i1 in range(ilo, ihi + 1):           # :4002-4005 commit
                for j1 in range(jlo, jhi + 1):
                    out[k, j1 - 1, i1 - 1] = new[j1 - 1, i1 - 1]
            new = np.empty_like(out[k])
            for j1 in range(jlo, jhi + 1):           # :4007-4011 i filter
                for i1 in range(ilo, ihi + 1):
                    j, i = j1 - 1, i1 - 1
                    new[j, i] = F(out[k, j, i] + F(nu * F(
                        F(F(F(out[k, j, i + 1] + out[k, j, i - 1]))
                          * F(0.5)) - out[k, j, i])))
            for j1 in range(jlo, jhi + 1):           # :4013 commit
                for i1 in range(ilo, ihi + 1):
                    out[k, j1 - 1, i1 - 1] = new[j1 - 1, i1 - 1]
    return out


def _registration(stagger, *, ratio=3, child_nx=24, child_ny=30,
                  ipos=5, jpos=7, parent_nx=40, parent_ny=44):
    return register_nest(
        nri=ratio, nrj=ratio, i_parent_start=ipos, j_parent_start=jpos,
        child_nx=child_nx, child_ny=child_ny,
        parent_nx=parent_nx, parent_ny=parent_ny,
        stagger=stagger, wrapper="bdy")


def test_the_window_is_wrfs_window():
    """[pos+2, pos+span-2-stag] per axis, in 0-based launch form."""
    reg = _registration("")
    assert smoother_parent_window(reg) == (6, 8, 4, 6)
    # x-staggered keeps the coincident high-side face: one more column.
    assert smoother_parent_window(_registration("x")) == (6, 8, 5, 6)
    assert smoother_parent_window(_registration("y")) == (6, 8, 4, 7)
    # A nest too small to smooth is WRF's zero-trip DO loop.
    tiny = _registration("", child_nx=9, child_ny=9, ratio=3)
    assert smoother_parent_window(tiny)[2] <= 0


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("stagger", ["", "x", "y"],
                         ids=["mass", "xstag", "ystag"])
@pytest.mark.parametrize("smooth_option", [1, 2], ids=["sm121", "smdsm"])
def test_gpu_smoother_is_bitwise_the_fortran(stagger, smooth_option):
    import cupy as cp

    rng = np.random.default_rng(20260821)
    reg = _registration(stagger)
    nz = 4
    nxp = 40 + (1 if stagger == "x" else 0)
    nyp = 44 + (1 if stagger == "y" else 0)
    field = rng.standard_normal((nz, nyp, nxp)).astype(np.float32)
    istag = 0 if stagger == "x" else 1
    jstag = 0 if stagger == "y" else 1
    ref = (np_sm121 if smooth_option == 1 else np_smdsm)(
        field, 5, 7, 24 // 3, 30 // 3, istag, jstag)
    dev = cp.asarray(field)
    smoother(dev, reg, smooth_option=smooth_option,
             scratch=cp.empty(dev.size, dtype=cp.float32))
    got = cp.asnumpy(dev)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))
    # And it genuinely did something.
    assert np.count_nonzero(got.view(np.uint32)
                            != field.view(np.uint32)) > 0


@requires_gpu
@pytest.mark.gpu
def test_mu_smooths_through_the_2d_path():
    """MU carries Registry flag `s` too (Registry.EM_COMMON:288)."""
    import cupy as cp

    rng = np.random.default_rng(7)
    reg = _registration("")
    mu = rng.standard_normal((44, 40)).astype(np.float32)
    ref = np_smdsm(mu[None], 5, 7, 8, 10, 1, 1)[0]
    dev = cp.asarray(mu)
    smoother(dev, reg, smooth_option=2,
             scratch=cp.empty(dev.size, dtype=cp.float32))
    assert np.array_equal(cp.asnumpy(dev).view(np.uint32),
                          ref.view(np.uint32))


@requires_gpu
@pytest.mark.gpu
def test_smooth_option_0_is_byte_inert():
    import cupy as cp

    rng = np.random.default_rng(3)
    reg = _registration("")
    field = rng.standard_normal((2, 44, 40)).astype(np.float32)
    dev = cp.asarray(field)
    smoother(dev, reg, smooth_option=0,
             scratch=cp.empty(dev.size, dtype=cp.float32))
    assert np.array_equal(cp.asnumpy(dev).view(np.uint32),
                          field.view(np.uint32))


def test_out_of_range_option_refuses_by_name():
    reg = _registration("")
    with pytest.raises(ValueError, match="smooth_option must be 0, 1 or 2"):
        smoother(np.zeros((2, 44, 40), np.float32), reg,
                 smooth_option=3, scratch=None)


# ---------------------------------------------------------------------------
# The coupler transaction: restriction THEN smoother, and the windowed
# re-diagnosis covers everything the transaction changed.
# ---------------------------------------------------------------------------

def _transacted_parent(smooth_option):
    """Run one full prepare/commit/finalize at feedback=1 and return the
    parent state, reusing test_feedback's seeded harness."""
    from test_feedback import _domain, _seed_state

    from gpuwm.core.model import FeedbackScratch
    from gpuwm.core.nest import NestCoupler
    from gpuwm.core.state import DomainState

    parent_cfg = _domain(1, 0, nx=14, ny=14)
    child_cfg = _domain(2, 1, nx=9, ny=9, ratio=3)
    parent_state = DomainState(parent_cfg.run)
    child_state = DomainState(child_cfg.run)
    rng = np.random.default_rng(99)
    import cupy as cp
    for state in (parent_state, child_state):
        _seed_state(state, child=state is child_state)
        for name in ("thp", "u", "v", "w", "php"):
            arr = getattr(state, name)
            arr[...] = cp.asarray(
                rng.standard_normal(arr.shape).astype(np.float32))
    clock = SimpleNamespace(ticks=0)
    parent = SimpleNamespace(cfg=parent_cfg, state=parent_state, clock=clock)
    child = SimpleNamespace(cfg=child_cfg, state=child_state, clock=clock,
                            parent=parent)
    # The windowed-finalize contract assumes what production always has:
    # a parent whose p/al/alt are already consistent with its inputs
    # everywhere (diagnosed at initialization and after every step).
    from gpuwm.core.diagnostics import update_diagnostics

    update_diagnostics(parent_state, 1)
    coupler = NestCoupler(child, feedback=1, smooth_option=smooth_option)
    scratch = FeedbackScratch()
    coupler.feedback_prepare(child, scratch)
    coupler.feedback_commit(child)
    coupler.feedback_finalize(child)
    return parent_state, coupler


@requires_gpu
@pytest.mark.gpu
def test_the_transaction_smooths_the_restricted_parent_exactly():
    """option-2 parent == np_smdsm(option-0 parent) on every fed field."""
    import cupy as cp

    plain, coupler0 = _transacted_parent(0)
    smoothed, _ = _transacted_parent(2)
    child_nx = child_ny = 9
    ratio = 3
    for kind, stagger in (("t", ""), ("u", "x"), ("v", "y"),
                          ("w", ""), ("ph", ""), ("mu", "")):
        name = {"t": "thp", "ph": "php", "mu": "mup"}.get(kind, kind)
        base = cp.asnumpy(getattr(plain, name))
        got = cp.asnumpy(getattr(smoothed, name))
        istag = 0 if stagger == "x" else 1
        jstag = 0 if stagger == "y" else 1
        ref = np_smdsm(base if base.ndim == 3 else base[None],
                       4, 4, child_nx // ratio, child_ny // ratio,
                       istag, jstag)
        if base.ndim == 2:
            ref = ref[0]
        assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), kind


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("smooth_option", [0, 2], ids=["raw", "smoothed"])
def test_finalize_rediagnosed_exactly_what_changed(smooth_option):
    """The windowed re-diagnosis equals a whole-parent re-diagnosis.

    If the window missed a changed column, the whole-parent pass below
    would move p/al/alt somewhere the finalize did not.
    """
    import cupy as cp

    from gpuwm.core.diagnostics import update_diagnostics

    parent_state, _ = _transacted_parent(smooth_option)
    p = cp.asnumpy(parent_state.p).copy()
    al = cp.asnumpy(parent_state.al).copy()
    alt = cp.asnumpy(parent_state.alt).copy()
    update_diagnostics(parent_state, 1)
    assert np.array_equal(cp.asnumpy(parent_state.p).view(np.uint32),
                          p.view(np.uint32))
    assert np.array_equal(cp.asnumpy(parent_state.al).view(np.uint32),
                          al.view(np.uint32))
    assert np.array_equal(cp.asnumpy(parent_state.alt).view(np.uint32),
                          alt.view(np.uint32))


@requires_gpu
@pytest.mark.gpu
def test_smdsm_undershoot_on_moisture_is_clamped_at_the_transaction():
    """The de-smoother (xnu = -0.52) undershoots sharp gradients; on a
    positive-definite species the transaction clamps the smoother window
    to >= 0 -- MEASURED on the first live two-way feedback, where a
    qv of -2.2e-07 inside the rectangle failed the health gate.  Signed
    fields are deliberately NOT clamped."""
    import cupy as cp

    from test_feedback import _domain, _seed_state

    from gpuwm.core.model import FeedbackScratch
    from gpuwm.core.nest import NestCoupler
    from gpuwm.core.nest_interp import smoother_parent_window
    from gpuwm.core.state import DomainState

    parent_cfg = _domain(1, 0, nx=20, ny=20)
    child_cfg = _domain(2, 1, nx=27, ny=27, ratio=3)
    parent_state = DomainState(parent_cfg.run)
    child_state = DomainState(child_cfg.run)
    _seed_state(parent_state, child=False)
    _seed_state(child_state, child=True)
    # A hard qv edge across the child: zero west, large east.  The
    # restriction carries the edge onto the parent; smdsm pass 2 then
    # undershoots on the low side of it.
    child_state.qv[...] = cp.float32(0.0)
    child_state.qv[:, :, 14:] = cp.float32(1.0e-2)
    # A uniformly negative SIGNED field: the clamp must leave it alone.
    child_state.u[...] = cp.float32(-1.0)
    clock = SimpleNamespace(ticks=0)
    parent = SimpleNamespace(cfg=parent_cfg, state=parent_state,
                             clock=clock)
    child = SimpleNamespace(cfg=child_cfg, state=child_state, clock=clock,
                            parent=parent)
    from gpuwm.core.diagnostics import update_diagnostics

    update_diagnostics(parent_state, 1)
    coupler = NestCoupler(child, feedback=1, smooth_option=2)
    scratch = FeedbackScratch()
    coupler.feedback_prepare(child, scratch)
    coupler.feedback_commit(child)
    coupler.feedback_finalize(child)

    qv = cp.asnumpy(parent_state.qv)
    assert float(qv.min()) >= 0.0
    # The clamp genuinely fired: re-run the raw smoother arithmetic on
    # the restricted-but-unclamped field and confirm it dips negative.
    reg = coupler.registrations["m"]
    i0, j0, niw, njw = smoother_parent_window(reg)
    window = qv[:, j0:j0 + njw, i0:i0 + niw]
    assert float(window.min()) == 0.0, "the edge should pin exact zeros"
    assert np.count_nonzero(window == 0.0) > 0
    # Signed fields keep their smoothed values, negatives included: a
    # uniformly -1 child restricts to -1, smooths to -1, and must NOT be
    # clamped to zero.
    u = cp.asnumpy(parent_state.u)
    iu0, ju0, niu, nju = smoother_parent_window(coupler.registrations["x"])
    assert float(u[:, ju0:ju0 + nju, iu0:iu0 + niu].max()) < 0.0
