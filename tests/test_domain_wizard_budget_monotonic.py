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


def _fit(ladder: str, gib: float, stop: dict | None = None):
    return dw.fit_ladder(
        ladder=ladder, free_bytes=int(gib * GIB), vram_gib=float(gib),
        hours=6, start_time=datetime.datetime(2026, 8, 2, 0, 0),
        projection=PROJECTION, source="gfs", name="budget-monotonic",
        stop_out=stop)


@pytest.mark.parametrize("ladder", ["12", "12-3"])
def test_a_larger_budget_buys_more_grid_until_a_named_bound_binds(
        ladder: str, capsys) -> None:
    """The property the cap violated, across the range real cards span.

    Restated in 2.5.0, because memory stopped being the only bound the
    fit answers to.  A 12 km root sized for a 64 GiB card spans more
    longitude than one crop of a global source can serve, and the fit
    now shrinks against that limit instead of emitting a layout its own
    emission would refuse.  Restated again in 2.7.3, which gave the
    POINT fit two bounds of its own -- the extent a centre with no
    extent is sized to, and the projection's polar envelope -- so the
    stall on this ladder is now request-shaped and arrives sooner.  So
    saturation is legitimate here -- what is not legitimate, and what
    this file exists for, is saturation nobody can see.  Every stall
    must therefore be REPORTED, and a bigger card must buy more grid up
    to the point where one does.

    Reported, not warned: the two request-shaped bounds are the DEFAULT
    sizing of a request that carries no extent, so they bind on the
    ordinary mid-latitude point on any card from about 16 GiB up, and a
    stderr ``warning:`` on the normal case is its own defect (it turned
    ``tests/test_go_chain.py`` red on the door's default emission).  The
    fit hands the bound to its caller in ``stop_out`` and the door
    states it as one line of plan summary; a source-shaped ceiling,
    which is not the normal case, keeps its warning.  Both are visible,
    which is the property this file is about.
    """
    budgets = (32, 48, 64, 96, 180)
    cells, spoke = {}, {}
    for gib in budgets:
        stop: dict = {}
        dims, _ = _fit(ladder, gib, stop)
        nx, ny = dims[0]
        cells[gib] = nx * ny
        spoke[gib] = ("not the card" in capsys.readouterr().err
                      or bool(stop))
    pairs = list(zip(budgets, budgets[1:]))
    silent_stall = [(a, b) for a, b in pairs
                    if cells[b] <= cells[a] and not spoke[b]]
    assert not silent_stall, (
        f"ladder {ladder}: a bigger budget bought no more grid at "
        f"{silent_stall} and the wizard said nothing about why; "
        f"cells by budget = {cells}")


def test_the_stall_announcement_is_not_simply_always_on(capsys) -> None:
    """The control the test above needs to mean anything.

    "Every stall is announced" is satisfied by a wizard that announces
    at every size and never grows, so one pair has to grow on memory
    alone, silently.  A two-level ladder is that pair on both platforms:
    its 3 km nest keeps the root well short of every non-memory bound,
    so the card is what binds there whether or not the peak envelope
    carries the Windows WDDM floor.

    The pair moved down a tier in 2.7.3, from 32 -> 64 to 16 -> 32 GiB.
    The point fit gained a maximum extent that a 64 GiB two-level ladder
    now reaches (its root wanted 7,800 km), so that end of the old pair
    is no longer a memory-bound one and would have proved the opposite
    of what this control is for.  Both budgets here are well inside
    every non-memory bound, which is the property the control needs.
    """
    small_stop, large_stop = {}, {}
    small, _ = _fit("12-3", 16, small_stop)
    assert "not the card" not in capsys.readouterr().err
    assert small_stop == {}
    large, _ = _fit("12-3", 32, large_stop)
    assert "not the card" not in capsys.readouterr().err
    assert large_stop == {}
    assert large[0][0] * large[0][1] > small[0][0] * small[0][1], (
        f"a 32 GiB budget bought no more grid than 16 GiB: "
        f"{large[0]} vs {small[0]}")


def test_the_scale_ceiling_does_not_bind_at_any_real_card_size() -> None:
    """180 GiB is the largest accelerator this project has been asked about.

    If the ceiling ever binds again the wizard silently under-sizes, so this
    pins the ceiling above what the largest budget actually wants rather than
    pinning a particular number -- a future card may legitimately want more.

    The bound that stops this ladder is now the point fit's maximum
    extent, which is reported; what must never stop it is the
    unreported bracket.
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
