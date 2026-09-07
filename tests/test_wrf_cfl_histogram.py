"""``w_cfl_stat``'s vert_cfl histogram, against a CPU reference.

WHY THIS EXISTS.  ``out[0]`` is a maximum -- an order statistic that one
cell decides.  A controller reading it cannot tell "the flow got faster
everywhere" from "one column is having a moment", and upstream's +5%
per-step growth clamp (adapt_timestep_em.F:174) is a rate limiter
standing in for the distribution nobody could afford to measure.  The
histogram measures it.  These gates pin that the bins mean what the host
thinks they mean, because nothing type-checks a .cu against a .py.

The construction below pins ``m = 1``, ``rdnw = 1`` and ``dt = 1`` so
that ``vert_cfl == |ww|`` exactly, which makes the expected histogram
something numpy can state independently rather than a re-derivation of
the kernel's own arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.core.dycore import (          # noqa: E402
    _CFL_HIST_BINS, _CFL_HIST_SCALE, _hist_quantile, _BC_THREADS,
    wrf_cfl_histogram_edges,
)
from gpuwm.core.kernels import get_kernel     # noqa: E402

DT = np.float32


def _run(ww_host: np.ndarray) -> np.ndarray:
    """Fold ``ww_host`` through the kernel; return the 4+BINS out words.

    Every other input is pinned so that vert_cfl reduces to |ww|:
    ``m = c1f*(mub2d + mup) + c2f = 1*(1 + 0) + 0``, ``rdnw = 1``,
    ``dt = 1``.  u and v are zero, so the horizontal arm folds nothing.
    """
    nzp1, ny, nx = ww_host.shape
    nz = nzp1 - 1
    g = lambda a: cp.asarray(a, dtype=DT)                    # noqa: E731
    out = cp.zeros(4 + _CFL_HIST_BINS, dtype=cp.uint32)
    plane = ny * nx
    blocks = ((nz - 1) * plane + _BC_THREADS - 1) // _BC_THREADS
    get_kernel("openbc", "w_cfl_stat")(
        (blocks,), (_BC_THREADS,),
        (g(ww_host), g(np.zeros((ny, nx))), g(np.ones((ny, nx))),
         g(np.ones(nzp1)), g(np.zeros(nzp1)), g(np.ones(nz)),
         g(np.zeros((nz, ny, nx + 1))), g(np.zeros((nz, ny + 1, nx))),
         g(np.ones((ny, nx + 1))), g(np.ones((ny + 1, nx))),
         out, DT(1.0), DT(1.0), DT(1.0),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    return cp.asnumpy(out)


def _expected_hist(ww_host: np.ndarray) -> np.ndarray:
    """What the histogram must be, stated without the kernel's help."""
    nz = ww_host.shape[0] - 1
    vals = np.abs(ww_host[1:nz].astype(np.float32).ravel())   # k = 1..nz-1
    idx = np.where(vals < _CFL_HIST_BINS / _CFL_HIST_SCALE,
                   (vals * _CFL_HIST_SCALE).astype(np.int64),
                   _CFL_HIST_BINS - 1)
    return np.bincount(idx, minlength=_CFL_HIST_BINS)


# ------------------------------------------------------------- the bins

def test_the_histogram_matches_a_cpu_reference():
    rng = np.random.default_rng(20260902)
    ww = rng.uniform(-2.5, 2.5, size=(9, 12, 20)).astype(np.float32)
    got = _run(ww)[4:]
    np.testing.assert_array_equal(got, _expected_hist(ww))


def test_every_visited_cell_lands_in_exactly_one_bin():
    """The histogram must account for out[2], no more and no less."""
    rng = np.random.default_rng(7)
    ww = rng.uniform(-3.0, 3.0, size=(7, 8, 16)).astype(np.float32)
    out = _run(ww)
    assert int(out[4:].sum()) == int(out[2]) == (7 - 1 - 1) * 8 * 16


def test_the_top_bin_absorbs_overflow_and_NaN():
    """Both by construction: every comparison against NaN is false.

    If the kernel used ``min(bins-1, (int)(cfl*scale))`` instead, the NaN
    cast would be undefined and this would be a lottery.
    """
    ww = np.zeros((4, 4, 4), dtype=np.float32)
    ww[1:3] = 0.0
    ww[1, 0, 0] = 99.0                       # far past the top edge
    ww[1, 0, 1] = np.nan
    ww[2, 0, 0] = -50.0                      # |.| still overflows
    out = _run(ww)
    assert int(out[4 + _CFL_HIST_BINS - 1]) == 3
    assert int(out[4:].sum()) == int(out[2])


def test_a_max_below_the_top_edge_leaves_the_top_bin_empty():
    """A positive control: the overflow bin is not a catch-all."""
    rng = np.random.default_rng(11)
    ww = rng.uniform(-1.0, 1.0, size=(6, 6, 8)).astype(np.float32)
    out = _run(ww)
    assert int(out[4 + _CFL_HIST_BINS - 1]) == 0
    assert int(out[4:].sum()) == int(out[2])


def test_the_bins_agree_with_the_maximum_the_same_kernel_reports():
    """out[0] and the histogram are two views of one distribution.

    The max must fall inside the highest OCCUPIED bin -- that is the
    consistency the quantile comparison depends on.
    """
    rng = np.random.default_rng(1234)
    ww = rng.uniform(-1.8, 1.8, size=(8, 10, 10)).astype(np.float32)
    out = _run(ww)
    vmax = np.uint32(out[0]).view(np.float32)
    top = int(np.nonzero(out[4:])[0].max())
    edges = wrf_cfl_histogram_edges()
    assert edges[top] <= float(vmax) < edges[top] + 1.0 / _CFL_HIST_SCALE


# -------------------------------------------------------- the quantiles

def test_host_edges_match_the_kernels_own_bin_arithmetic():
    """The one thing NVRTC cannot check for us."""
    edges = wrf_cfl_histogram_edges()
    assert len(edges) == _CFL_HIST_BINS
    for b, e in enumerate(edges):
        probe = np.float32(e + 0.5 / _CFL_HIST_SCALE)        # bin centre
        assert int(probe * _CFL_HIST_SCALE) == b


def test_a_quantile_bounds_the_true_value_from_above():
    """_hist_quantile returns an upper edge, and must never undershoot."""
    rng = np.random.default_rng(99)
    vals = rng.uniform(0.0, 1.9, size=50_000).astype(np.float32)
    idx = (vals * _CFL_HIST_SCALE).astype(np.int64)
    hist = np.bincount(idx, minlength=_CFL_HIST_BINS)[None, :]
    cum = np.cumsum(hist, axis=1)
    for q in (0.5, 0.9, 0.999, 0.9999):
        got = _hist_quantile(cum, cum[:, -1], q)[0]
        truth = float(np.quantile(vals, q))
        assert got >= truth, (q, got, truth)
        assert got - truth < 1.0 / _CFL_HIST_SCALE + 1e-6, (q, got, truth)


def test_an_empty_step_is_NaN_and_not_a_spurious_zero():
    """A step that folded nothing must not read as 'CFL 0.0625'."""
    hist = np.zeros((1, _CFL_HIST_BINS), dtype=np.int64)
    cum = np.cumsum(hist, axis=1)
    assert np.isnan(_hist_quantile(cum, cum[:, -1], 0.999)[0])


def test_a_quantile_landing_in_the_open_top_bin_is_inf_not_a_number():
    """The top bin has no upper edge, so no finite answer is defensible."""
    hist = np.zeros((1, _CFL_HIST_BINS), dtype=np.int64)
    hist[0, -1] = 10
    cum = np.cumsum(hist, axis=1)
    assert np.isinf(_hist_quantile(cum, cum[:, -1], 0.5)[0])


# ------------------------------------------- the report's ratio statistic

def test_a_saturated_step_does_not_drag_the_ratio_toward_zero():
    """A quantile in the open top bin is +inf, and vmax/inf is 0.0.

    0.0 is FINITE, so without an explicit isfinite guard it survives the
    filter and pulls ``median_max_over_p9999`` toward zero -- reporting
    "the max IS the distribution" for exactly the saturated steps where
    that is least true, i.e. failing in the direction that would kill the
    quantile-controller idea on false evidence.
    """
    from gpuwm.core import dycore

    key = 991
    rows = np.zeros((4, dycore._WRF_CFL_WORDS), dtype=np.uint32)
    # one in-range step: max 0.5, all mass in the bin containing 0.5
    rows[0, 0] = np.float32(0.5).view(np.uint32)
    rows[0, 2] = 1000
    rows[0, 4 + int(0.5 * _CFL_HIST_SCALE)] = 1000
    # three saturated steps: finite max, all mass in the OPEN top bin
    for r in range(1, 4):
        rows[r, 0] = np.float32(5.0).view(np.uint32)
        rows[r, 2] = 1000
        rows[r, 4 + _CFL_HIST_BINS - 1] = 1000

    saved = (dycore._WRF_CFL_STAT.get(key), dycore._WRF_CFL_CALLS.get(key))
    dycore._WRF_CFL_STAT[key] = cp.asarray(rows)
    dycore._WRF_CFL_CALLS[key] = 12                  # 4 steps x 3 RK stages
    try:
        report = [r for r in dycore.wrf_vertical_cfl_report()
                  if r["steps"] == 4][0]
    finally:
        for store, val in ((dycore._WRF_CFL_STAT, saved[0]),
                           (dycore._WRF_CFL_CALLS, saved[1])):
            if val is None:
                store.pop(key, None)
            else:
                store[key] = val

    got = report["median_max_over_p9999"]
    # Only the in-range step contributes: 0.5 / (9/16) = 0.888...
    assert got == pytest.approx(0.5 / (9.0 / _CFL_HIST_SCALE), rel=1e-6), got
    assert got > 0.5, f"ratio {got} collapsed -- the saturated steps leaked in"


# ---------------------------------------------- the per-step slot ring

def test_a_run_past_the_slot_count_does_not_pin_one_row():
    """The saturating slot made the reported CFL a RUNNING maximum.

    ``slot = min(calls // 3, _WRF_CFL_SLOTS - 1)`` lands every fold past
    step 32768 in one row that nothing ever clears, and ``atomicMax``
    then makes ``out[0]`` monotonically non-decreasing.  From that step on
    the controller can only shrink dt, and a long run (a 24 h nest at
    dt = 2 s is 43,200 steps) collapses toward ``min_time_step`` for a
    reason no diagnostic reports.
    """
    from gpuwm.core import dycore

    slots = dycore._WRF_CFL_SLOTS
    words = dycore._WRF_CFL_WORDS
    buf = cp.zeros((slots, words), dtype=cp.uint32)
    try:
        dycore._WRF_CFL_STAT[9001] = buf
        dycore._WRF_CFL_CALLS[9001] = 0

        def fold(calls, value):
            """What record_wrf_vertical_cfl does, without the kernel."""
            slot = (calls // 3) % slots
            if calls % 3 == 0:
                buf[slot].fill(0)
            bits = int(np.float32(value).view(np.uint32))
            buf[slot, 0] = max(int(buf[slot, 0]), bits)

        # One high step, then a step at the SAME slot one full ring later.
        fold(0, 4.0)
        dycore._WRF_CFL_CALLS[9001] = 3
        assert dycore.take_wrf_cfl(9001)[0] == pytest.approx(4.0)

        calls = 3 * slots
        fold(calls, 0.25)
        dycore._WRF_CFL_CALLS[9001] = calls + 3
        assert dycore.take_wrf_cfl(9001)[0] == pytest.approx(0.25), (
            "the row carried the previous lap's maximum, so the "
            "controller reads a running max and can only shrink dt")
    finally:
        dycore._WRF_CFL_STAT.pop(9001, None)
        dycore._WRF_CFL_CALLS.pop(9001, None)


def test_the_reset_puts_the_probe_back_where_the_environment_left_it():
    """A second experiment in one process inherited the first's fold."""
    from gpuwm.core import dycore

    was = dycore._WRF_CFL_PROBE
    try:
        dycore.enable_wrf_cfl_recording()
        assert dycore._WRF_CFL_PROBE is True
        dycore._WRF_CFL_STAT[9002] = cp.zeros(
            (1, dycore._WRF_CFL_WORDS), dtype=cp.uint32)
        dycore._WRF_CFL_CALLS[9002] = 3
        dycore.reset_wrf_cfl_recording()
        assert dycore._WRF_CFL_PROBE is dycore._env_flag("GPUWM_WRF_CFL_PROBE")
        assert 9002 not in dycore._WRF_CFL_STAT
        assert dycore.take_wrf_cfl(9002) == (0.0, 0.0)
    finally:
        dycore._WRF_CFL_PROBE = was
        dycore._WRF_CFL_STAT.pop(9002, None)
        dycore._WRF_CFL_CALLS.pop(9002, None)
