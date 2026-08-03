"""A larger card must buy a larger domain, and a bound that binds must say so.

`_MAX_SCALE` was an undocumented ceiling on the wizard's scale bisection.  Its
sibling `_MIN_SCALE` carries a full rationale in the source ("the smallest
layout that still hosts the deepest ladder with full Davies/blend clearance");
`_MAX_SCALE` carried none, and was introduced in the wizard's first commit with
no justification.  At 8.0 it bound for every single-domain budget at or above
64 GiB, so 64, 96 and 180 GiB all emitted the identical 880x704 root while
reporting a comfortable fit -- a 180 GiB card sized like a 64 GiB one, with the
difference silently unused.

The bug was invisible from any single invocation.  Nothing in the wizard's
output distinguished "this is the largest grid your budget affords" from "this
is the largest grid I was willing to consider", which is why it survived until
someone ran the wizard twice at different budgets and compared.

Both properties are tested here: budgets buy grids, and saturation is audible.
"""
from __future__ import annotations

import datetime

import pytest

from gpuwm import domain_wizard as dw

GIB = 1024 ** 3

PROJECTION = {
    "map_proj": "lambert", "ref_lat": 39.0, "ref_lon": -98.0,
    "truelat1": 33.0, "truelat2": 45.0, "stand_lon": -98.0,
}


def _fit(ladder: str, gib: float):
    return dw.fit_ladder(
        ladder=ladder, free_bytes=int(gib * GIB), vram_gib=float(gib),
        hours=6, start_time=datetime.datetime(2026, 8, 2, 0, 0),
        projection=PROJECTION, source="gfs", name="budget-monotonic")


@pytest.mark.parametrize("ladder", ["12", "12-3"])
def test_a_larger_budget_buys_a_strictly_larger_root_domain(ladder: str) -> None:
    """The property the cap violated, across the range real cards span."""
    budgets = (32, 48, 64, 96, 180)
    cells = {}
    for gib in budgets:
        dims, _ = _fit(ladder, gib)
        nx, ny = dims[0]
        cells[gib] = nx * ny
    pairs = list(zip(budgets, budgets[1:]))
    stalled = [(a, b) for a, b in pairs if cells[b] <= cells[a]]
    assert not stalled, (
        f"ladder {ladder}: a bigger budget bought no more grid at "
        f"{stalled}; cells by budget = {cells}")


def test_the_scale_ceiling_does_not_bind_at_any_real_card_size() -> None:
    """180 GiB is the largest accelerator this project has been asked about.

    If the ceiling ever binds again the wizard silently under-sizes, so this
    pins the ceiling above what the largest budget actually wants rather than
    pinning a particular number -- a future card may legitimately want more.
    """
    dims, _ = _fit("12", 180)
    nx, _ = dims[0]
    wanted = nx / 110.0
    assert wanted < dw._MAX_SCALE * 0.95, (
        f"a 180 GiB budget wants scale ~{wanted:.2f} against a ceiling of "
        f"{dw._MAX_SCALE}: the ceiling is binding and the wizard is "
        "under-sizing large cards")


def test_min_scale_still_bounds_the_search_from_below() -> None:
    """Raising the ceiling must not disturb the floor, which is load-bearing."""
    assert dw._MIN_SCALE == 0.55
    assert dw._dims_for_scale(dw._MIN_SCALE, ())[0] == (60, 48)
