"""Threaded ensemble statistics give the single numpy call's bytes.

CPU only.  THE BREAKAGE THIS PREVENTS: the cycle driver's field selection
and leg record summarised whole-ensemble float64 stacks on one core, about
80 s of every 9 km CONUS analysis.  gpuwm.da.ensemble_stats spreads the
same work over threads; a statistic that came back different in the last
bit would change a receipt (or, for the field selection, possibly an
analysis), so every one is compared with the expression it replaces by
exact equality.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.da import ensemble_stats as es


def _members(rng, count=7, shape=(5, 13, 211), zeros=0.4, dtype=np.float64):
    out = []
    for _ in range(count):
        x = rng.standard_normal(shape) * np.exp(rng.standard_normal(shape) * 3)
        x[rng.random(shape) < zeros] = 0.0
        out.append(x.astype(dtype))
    return out


@pytest.mark.parametrize("n", [1, 7, 129, 4097, 2 * (1 << 20) + 1,
                               5 * (1 << 20) + 333])
def test_exact_sum_is_numpys_sum(n):
    rng = np.random.default_rng(n)
    a = rng.standard_normal(n) * np.exp(rng.standard_normal(n) * 6)
    assert es.exact_sum(a, workers=7) == np.add.reduce(a)
    assert es.exact_sum(a, workers=1) == np.add.reduce(a)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_field_spread_is_the_stack_expression(dtype):
    rng = np.random.default_rng(1)
    members = _members(rng, dtype=dtype)
    stack = np.stack(members).astype(np.float64)
    expected = (float(np.abs(stack).max()),
                float(stack.std(axis=0, ddof=1).max()))
    for workers in (1, 3, 16):
        assert es.field_spread(members, workers=workers) == expected


def test_field_spread_keeps_numpys_nan():
    rng = np.random.default_rng(2)
    members = _members(rng)
    members[3][2, 4, 5] = np.nan
    spread = es.field_spread(members, workers=5)
    assert np.isnan(spread[0]) and np.isnan(spread[1])


@pytest.mark.parametrize("shape", [(5, 13, 211), (9, 60, 4000)])
def test_increment_stats_are_the_stack_expressions(shape):
    rng = np.random.default_rng(3)
    members = _members(rng, count=6, shape=shape)
    stack = np.stack(members)
    expected = {
        "finite": bool(np.isfinite(stack).all()),
        "max_abs": float(np.abs(stack).max()),
        "rms_where_nonzero": float(np.sqrt(np.mean(
            stack[stack != 0.0] ** 2))),
        "nonzero_fraction": float(np.mean(stack != 0.0)),
    }
    for workers in (1, 4, 13):
        assert es.increment_stats(members, workers=workers) == expected


def test_increment_stats_of_all_zero_increments():
    members = [np.zeros((3, 4, 5)) for _ in range(4)]
    assert es.increment_stats(members, workers=3) == {
        "finite": True, "max_abs": 0.0, "rms_where_nonzero": 0.0,
        "nonzero_fraction": 0.0}


def test_support_is_the_stack_any():
    rng = np.random.default_rng(4)
    members = _members(rng, zeros=0.97)
    expected = np.any(np.stack(members) != 0.0, axis=0)
    got = es.support(members, workers=6)
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)
