"""The localize-pair receipts behind the strict dycore's round-3 claims.

``tools/wrf_exact_localize`` runs stock WRF 4.6.1 built with dump hooks (writes
only; the build reproduces the stock scalar recording of A088 in 208 of 208
fields) beside strict WOOF on the same wrfinput/wrfbdy, and compares every word
at every matched point of each step.  These tests pin what the receipts record,
so a later reading that claims the strict dycore matches WRF has to come with
a receipt that says so.  CPU only: the receipts are JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tools" / "wrf_exact_localize" / "receipts"
REASONS = {
    "intermediate of a different term sum",
    "WRF tile-local array, not readable by the memory crop",
    "WRF stores divergence-damped p''; WOOF the undamped",
    "in-loop ww is another quantity; ring values unused",
    "specified-ring values WRF never applies",
}


def _load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_physics_off_pair_matches_wrf_word_for_word_for_twenty_steps():
    r = _load("round3-physics-off-20-steps.json")
    assert r["schema"] == "gpuwm.wrf-exact-localize/v2"
    assert r["steps"] == 20 and len(r["per_step"]) == 20
    assert r["all_compared_identical"] is True
    for step in r["per_step"]:
        assert step["points_compared"] >= 260, step["step"]
        assert step["points_differing"] == [], step["step"]
    # Every exclusion is one of the named not-the-same-quantity reasons.
    assert {lab["excluded"] for lab in r["labels"]} - {None} <= REASONS
    compared = [i for i, lab in enumerate(r["labels"]) if lab["excluded"] is None]
    for step in r["per_step"]:
        assert all(step["differ_by_label"][i] in (0, None) for i in compared)
    # State and tendencies at every RK stage are among the compared points.
    points = {lab["point"] for lab in r["labels"] if lab["excluded"] is None}
    for needed in ("b u_2", "b p", "b ww", "e_addt ru_tend", "e_addt t_tend", "e_addt mu_tend",
                   "ss1 w_2", "ss1 ph_2", "finish u_2", "end u_2", "end qv",
                   "qv after update (WRF i_pphi)"):
        assert needed in points, needed


def test_physics_off_history_differs_only_in_the_physics_free_psfc():
    """With every physics scheme off WRF never refreshes PSFC (only phy_prep
    and the surface driver write it), so its history keeps the wrfinput value
    while WOOF diagnoses one; every other common field matches at every frame."""
    r = _load("round3-physics-off-20-steps.json")
    frames = r["history_frames_differing_fields"]
    assert len(frames) == 21
    assert all(fields == ["PSFC"] for fields in frames.values())


def test_physics_on_pair_leaves_wrf_only_where_the_physics_tendencies_enter():
    """A088 (YSU, revised MM5, Noah, RRTMG, WSM6): the step-1 state, the
    stage reference fluxes and the whole big-step tendency match; the first
    differing words are the coupled PBL and radiation tendencies entering the
    held *_tendf, which differ in the scheme outputs themselves
    (tools/wrf_exact_localize/cmpphys.py), not in the dycore's fold."""
    r = _load("round3-a088-physics-on-step1.json")
    step = r["per_step"][0]
    assert r["all_compared_identical"] is False
    first = step["points_differing"][0]
    assert first[1].startswith("held tendencies") and first[2].startswith("ru_tendf")
    by_label = {(lab["stage"], lab["point"]): step["differ_by_label"][i]
                for i, lab in enumerate(r["labels"])}
    for point in ("b u_2", "b w_2", "b t_2", "b mu_2", "b p", "b al", "b ww",
                  "d_rkte ru_tend", "d_rkte rv_tend", "d_rkte rw_tend", "d_rkte t_tend",
                  "d_rkte ph_tend"):
        assert by_label[("RK stage 1", point)] == 0, point
