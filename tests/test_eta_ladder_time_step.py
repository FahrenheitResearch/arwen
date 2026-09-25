"""A ladder is priced against the time step that has to run it.

A ladder is not a free choice of resolution.  The vertical Courant number
of an updraft through a layer is ``w * dt / dz``, so halving the layer
doubles it, and past 1 the engine's own vertical-velocity limiter
(``gpuwm/core/dycore.py::apply_w_damping``, WRF ``w_damping = 1``)
starts pushing against the motion.  That is a limiter, not physics, and
what it costs is the storm.

Measured on the card, the same storm at the same minute from the same
analysis: on the shipped 49-level ladder the convective layers are 643 m,
the strongest updraft is 33.9 m/s and the largest vertical Courant number
at the 15 s parent step is 0.79, with no cell over 1.  On an 80-level
ladder refined to about 200 m over the same heights the Courant number is
1.41 with 606 cells over 1, and the strongest updraft is 18.5 m/s --
half.  The 45 dBZ area lived 53 minutes on the first and 25 on the
second.

Nothing in the tool that BUILDS the ladder said a word about it.  Now it
does: the score carries the Courant number those layers will run at, and
the step that would hold it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "build_stretched_eta_ladder", _ROOT / "tools"
    / "build_stretched_eta_ladder.py")
ladder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ladder)

#: The part of the column a convective updraft occupies.
BAND = (4000.0, 9500.0)
P_TOP = 5000.0


def _fine(dz_max=200.0, nz=110):
    """A ladder whose layers through BAND are about ``dz_max`` metres."""
    depth = ladder.analytic_base_terrain_height(P_TOP, 290.0)
    thick = ladder.stretched_thicknesses(nz, 20.0, dz_max, depth)
    z_full = np.concatenate(([0.0], np.cumsum(thick)))
    return ladder.eta_from_heights(z_full, P_TOP, 290.0)


def _certified():
    from gpuwm.native_wrf_contract import CERTIFIED_ETA_LEVELS
    return np.asarray(CERTIFIED_ETA_LEVELS, dtype=np.float64)


def test_the_shipped_ladder_is_priced_below_one_at_the_run_step():
    price = ladder.price_time_step(
        _certified(), p_top=P_TOP, hybrid_opt=2, etac=0.2,
        dt=15.0, w_max=33.9, band=BAND)
    assert price["vertical_courant_in_band"] == pytest.approx(
        0.79, abs=0.06), (
        "the measured value on the card was 0.79; this is the tool's own "
        f"arithmetic on the same ladder: {price}")
    assert price["holds"] is True


def test_a_refined_ladder_is_priced_over_one_at_the_same_step():
    price = ladder.price_time_step(
        _fine(), p_top=P_TOP, hybrid_opt=2, etac=0.2,
        dt=15.0, w_max=33.9, band=BAND)
    assert price["vertical_courant_in_band"] > 1.0, (
        "200 m layers under a 15 s step run the Courant number past 1 and "
        f"the limiter fires: {price}")
    assert price["holds"] is False


def test_it_names_the_step_that_would_hold_the_target():
    price = ladder.price_time_step(
        _fine(), p_top=P_TOP, hybrid_opt=2, etac=0.2,
        dt=15.0, w_max=33.9, band=BAND, target_courant=0.8)
    dt_needed = price["dt_for_target_s"]
    assert 3.0 < dt_needed < 7.0, (
        f"a 200 m band under a 34 m/s updraft needs about 4.7 s: "
        f"{dt_needed}")
    again = ladder.price_time_step(
        _fine(), p_top=P_TOP, hybrid_opt=2, etac=0.2,
        dt=dt_needed, w_max=33.9, band=BAND, target_courant=0.8)
    assert again["holds"] is True, (
        "the step the tool names has to be a step the ladder holds")


def test_a_finer_ladder_is_priced_worse_not_better():
    coarse = ladder.price_time_step(
        _fine(dz_max=200.0), p_top=P_TOP, hybrid_opt=2, etac=0.2,
        dt=15.0, w_max=33.9, band=BAND)
    fine = ladder.price_time_step(
        _fine(dz_max=150.0, nz=130), p_top=P_TOP, hybrid_opt=2, etac=0.2,
        dt=15.0, w_max=33.9, band=BAND)
    assert (fine["vertical_courant_in_band"]
            > coarse["vertical_courant_in_band"])
    assert fine["dt_for_target_s"] < coarse["dt_for_target_s"]


def test_the_price_is_carried_in_the_score_the_tool_writes():
    score = ladder.score_ladder(
        _fine(), p_top=P_TOP, hybrid_opt=2, etac=0.2, bl_top=1700.0,
        dt=15.0, w_max=33.9, band=BAND)
    assert "time_step" in score
    assert score["time_step"]["holds"] is False
    assert score["time_step"]["dt_s"] == 15.0


def test_a_ladder_scored_without_a_step_says_nothing_about_one():
    score = ladder.score_ladder(
        _fine(), p_top=P_TOP, hybrid_opt=2, etac=0.2, bl_top=1700.0)
    assert score.get("time_step") is None, (
        "a caller that did not state a step gets no opinion about one, "
        "because the tool would have to invent the updraft to have one")
