"""The background A/B's staging logic, tested where it can lie.

A staged experiment is dangerous in one specific way: the machinery that
decides whether the answer is yes can be written so that it can only ever
say yes.  So the verdict function is exercised against constructed
numbers for every path it can take, INCLUDING the four that falsify, and
the pin resolver is exercised against the wrapper result a preparation
run without ``--wps-namelist`` actually leaves behind.
"""

from __future__ import annotations

import json

import pytest

from tools.da_background_ab import build_plan, run_arm
from tools.da_background_ab.score_background_ab import HALF_WIDTHS, verdict


def _curve(values):
    return [{"half_width": half_width,
             "neighborhood_km": (2 * half_width + 1) * 3.0,
             "published_rung": half_width == 4,
             "fss_mean_over_leads": value}
            for half_width, value in zip(HALF_WIDTHS, values)]


def _arm(*, ensemble, control, consistency, increment):
    return {
        "curve_ensemble_mean": _curve(ensemble),
        "curve_control": _curve(control),
        "cycles": [{"consistency_ratio": consistency,
                    "mean_increment_rms": {"u": increment, "v": increment}}],
    }


FLAT = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def _shift(values, delta):
    return [round(value + delta, 4) for value in values]


def test_no_rung_improves_is_falsified():
    gfs = _arm(ensemble=FLAT, control=_shift(FLAT, -0.4), consistency=10.0,
               increment=1.0)
    hrrr = _arm(ensemble=_shift(FLAT, -0.01), control=_shift(FLAT, -0.4),
                consistency=10.0, increment=0.5)
    assert verdict(gfs, hrrr)["answer"].startswith("FALSIFIED: no rung")


def test_a_control_that_gains_as_much_falsifies_the_wah_claim():
    # The ensemble improves by 0.02 everywhere and so does the control:
    # the background's own forecast got better and the filter added
    # nothing on top of it.
    gfs = _arm(ensemble=FLAT, control=_shift(FLAT, -0.4), consistency=10.0,
               increment=1.0)
    hrrr = _arm(ensemble=_shift(FLAT, 0.02),
                control=_shift(FLAT, -0.4 + 0.02),
                consistency=10.0, increment=0.5)
    answer = verdict(gfs, hrrr)["answer"]
    assert "FALSIFIED as a statement about WaH" in answer


def test_a_less_consistent_ensemble_is_named_not_credited():
    gfs = _arm(ensemble=FLAT, control=_shift(FLAT, -0.4), consistency=10.0,
               increment=1.0)
    hrrr = _arm(ensemble=_shift(FLAT, 0.05), control=_shift(FLAT, -0.4),
                consistency=18.0, increment=0.5)
    answer = verdict(gfs, hrrr)["answer"]
    assert answer.startswith("NOT SUPPORTED as stated")
    assert "under-dispersion" in answer


def test_skill_without_smaller_increments_names_the_wrong_mechanism():
    gfs = _arm(ensemble=FLAT, control=_shift(FLAT, -0.4), consistency=10.0,
               increment=1.0)
    hrrr = _arm(ensemble=_shift(FLAT, 0.05), control=_shift(FLAT, -0.4),
                consistency=9.0, increment=1.4)
    answer = verdict(gfs, hrrr)["answer"]
    assert answer.startswith("SUPPORTED, but NOT by the mechanism claimed")


def test_the_only_supporting_path_needs_all_three():
    gfs = _arm(ensemble=FLAT, control=_shift(FLAT, -0.4), consistency=10.0,
               increment=1.0)
    hrrr = _arm(ensemble=_shift(FLAT, 0.05), control=_shift(FLAT, -0.4),
                consistency=9.0, increment=0.6)
    result = verdict(gfs, hrrr)
    assert result["answer"].startswith("SUPPORTED:")
    assert result["delta_at_published_rung"] == 0.05
    assert result["clauses"]["analysis_increments_shrank"] is True


def test_every_cycling_arm_shares_the_flags_that_must_not_move(tmp_path):
    plan = build_plan.build(
        case_root=str(tmp_path / "case"), geog_root=str(tmp_path / "geog"),
        run_seconds_hrrr=14400.0, run_seconds_gfs=21600.0,
        gfs_pins={"proof_sha256": "a" * 64,
                  "source_manifest_sha256": "b" * 64,
                  "prepared_content_sha256": "c" * 64})
    cycling = [arm for arm in plan["arms"] if "members" in arm]
    assert [arm["name"] for arm in cycling] == ["G-gfs", "H-matched",
                                                "H-fresh"]

    def shared(arm):
        argv = arm["steps"][0]["argv"]
        return {flag: argv[argv.index(flag) + 1]
                for flag in ("--members", "--seed", "--leg-seconds",
                             "--free-legs", "--wind-sigma-ms",
                             "--length-scale-km", "--rtps-alpha",
                             "--physics-profile")}

    reference = shared(cycling[0])
    for arm in cycling[1:]:
        assert shared(arm) == reference, arm["name"]

    # And the one thing that MUST differ, does.
    sources = [arm["steps"][0]["argv"][
        arm["steps"][0]["argv"].index("--source") + 1] for arm in cycling]
    assert sources == ["gfs", "hrrr", "hrrr"]


def test_the_two_hrrr_arms_take_different_cycles():
    plan = build_plan.build(
        case_root="/case", geog_root="/geog", run_seconds_hrrr=14400.0,
        run_seconds_gfs=21600.0,
        gfs_pins={"proof_sha256": "a" * 64,
                  "source_manifest_sha256": "b" * 64,
                  "prepared_content_sha256": "c" * 64})
    prepares = [arm for arm in plan["arms"]
                if arm["name"].startswith("prepare-hrrr-")]
    cycles = []
    for arm in prepares:
        argv = arm["steps"][0]["argv"]
        cycles.append((argv[argv.index("--valid-time") + 1],
                       argv[argv.index("--forecast-start-hour") + 1]))
    assert cycles == [("2026-08-05_00:00:00", "4"),
                      ("2026-08-05_04:00:00", "0")]


def test_a_preparation_without_the_opt_in_refuses_at_the_pin(tmp_path):
    # This is exactly what a run WITHOUT --wps-namelist leaves behind:
    # a PASS result with no portable_bundle.  Binding a cycling arm to it
    # must fail loudly rather than with a KeyError three frames deep.
    path = tmp_path / "public-wrapper-result.json"
    path.write_text(json.dumps({"status": "PASS", "portable_bundle": None}),
                    encoding="utf-8")
    with pytest.raises(SystemExit) as refusal:
        run_arm.read_pins(path)
    assert "--wps-namelist" in str(refusal.value)


def test_a_non_digest_pin_is_refused(tmp_path):
    path = tmp_path / "public-wrapper-result.json"
    path.write_text(json.dumps({"portable_bundle": {
        "proof_sha256": "a" * 64,
        "source_manifest_sha256": "not-a-digest",
        "prepared_content_sha256": "c" * 64}}), encoding="utf-8")
    with pytest.raises(SystemExit) as refusal:
        run_arm.read_pins(path)
    assert "--source-manifest-sha256" in str(refusal.value)


def test_pins_given_twice_are_refused():
    pins = {"proof_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
            "prepared_content_sha256": "c" * 64}
    with pytest.raises(SystemExit):
        run_arm.assemble(["--proof-sha256", "d" * 64], pins)
