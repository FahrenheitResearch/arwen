"""The six scoring holes the adversarial check of the combo sweep found.

What breaks without these: `gpuwm verify-exact` printed PASS (or exit 0) for
runs that are not exact.  Each probe below is one the check ran through the
real command (`--score-only`, no card), rebuilt here on a synthetic
recording:

H1   the referee writes a field WOOF never writes          -> was PASS
H1b  WOOF stops writing a state field                       -> was PASS
H2   a field written in another shape on each side          -> was PASS
H3   the referee crashed early; WOOF matched what it wrote  -> was PASS
H4   the referee is flagged unsound and WOOF matches it     -> was PASS
H5   no WRF referee; WOOF wrote 3 of 20 steps, status OK    -> was WOOF-ONLY-OK
H6   no clean_reference entry for the run                   -> was treated as sound
exit a selection with no PASS, or a --combos typo           -> was exit 0
cover pair coverage was claimed for the list, not for what ran
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm.verify_exact.compare import (ALLOWED_ABSENT, FieldMeta, RunRecord, compare_runs,
                                        field_digest)
from gpuwm.verify_exact.recording import write_run_record
from gpuwm.verify_exact.scoring import decide

STEPS = 4


def _frames(seed: int, steps: int = STEPS, extra: dict | None = None):
    rng = np.random.default_rng(seed)
    base = {"U": rng.standard_normal((3, 4, 5)).astype(np.float32),
            "QRAIN": rng.standard_normal((3, 4, 4)).astype(np.float32),
            "HFX": rng.standard_normal((4, 4)).astype(np.float32)}
    frames = {}
    for step in range(steps):
        frame = {k: v + np.float32(step) for k, v in base.items()}
        for name, make in (extra or {}).items():
            frame[name] = make(step)
        frames[step] = frame
    return frames


def _record(frames, label=""):
    fields, digests = {}, {}
    for step, arrays in frames.items():
        row = {}
        for name, values in arrays.items():
            meta = fields.setdefault(name, FieldMeta(name, values.dtype.str, values.shape))
            row[name] = field_digest(values, meta.dtype, meta.shape)
        digests[step] = row
    return RunRecord(fields, digests, label=label)


def _changing(step):
    return np.full((4, 4), step, np.float32)


def _constant(step):
    return np.linspace(0, 1, 50).astype(np.float32)


# -- hole 1: a WRF field WOOF does not write, or writes in another shape ------

def test_h1_a_reference_field_woof_never_writes_is_not_a_match():
    ref = _record(_frames(1, extra={"SFROFF": _changing}))
    woof = _record(_frames(1))
    result = compare_runs(woof, ref)
    assert result.first_step is None  # every field both write agrees...
    assert not result.matched          # ...but SFROFF was never compared
    assert result.absent_unexcused == ("SFROFF",)
    assert "SFROFF" in result.summary() and result.summary() != "match"


def test_h1b_woof_dropping_a_state_field_is_not_a_match():
    ref = _record(_frames(2))
    woof = _record({s: {k: v for k, v in f.items() if k != "QRAIN"}
                    for s, f in _frames(2).items()})
    result = compare_runs(woof, ref)
    assert not result.matched
    assert result.absent_unexcused == ("QRAIN",)


def test_h2_a_field_written_in_another_shape_is_not_a_match():
    ref = _record(_frames(3))
    frames = _frames(3)
    for frame in frames.values():
        frame["HFX"] = np.zeros((5, 4), np.float32)
    result = compare_runs(_record(frames), ref)
    assert not result.matched
    assert result.reshaped == {"HFX": ([5, 4], [4, 4])}
    assert "another shape" in result.summary()


def test_a_listed_time_invariant_field_woof_does_not_write_is_excused_with_its_reason():
    assert "FNM" in ALLOWED_ABSENT and ALLOWED_ABSENT["FNM"]
    ref = _record(_frames(4, extra={"FNM": _constant}))
    result = compare_runs(_record(_frames(4)), ref)
    assert result.matched
    assert result.absent_unexcused == ()
    assert list(result.absent_allowed) == ["FNM"]
    assert result.as_json()["absent_allowed"]["FNM"] == ALLOWED_ABSENT["FNM"]


def test_a_listed_field_that_changes_in_the_reference_is_not_excused():
    ref = _record(_frames(5, extra={"FNM": lambda s: np.full(50, s, np.float32)}))
    result = compare_runs(_record(_frames(5)), ref)
    assert not result.matched
    assert result.absent_unexcused == ("FNM",)


def test_the_excusable_list_holds_no_prognostic_physics_or_accumulated_field():
    for name in ("U", "V", "W", "T", "PH", "MU", "P", "QVAPOR", "TKE_PBL", "THM", "P_HYD",
                 "RTHBLTEN", "RUBLTEN", "RTHRATEN", "RTHCUTEN", "SFROFF", "UDROFF", "ACHFX",
                 "ACLHF", "ACGRDFLX", "ACSNOM", "SWUPT", "ACSWDNB", "LWDNB", "SST", "ALBEDO",
                 "EMISS", "LAI", "SR", "PC", "COSZEN", "NOAHRES", "XLAND", "XLAT", "MAPFAC_M"):
        assert name not in ALLOWED_ABSENT, name


# -- hole 2: soundness before PASS; a crashed referee never gives PASS ------

def _cmp(matched=True):
    ref = _record(_frames(6))
    frames = _frames(6)
    if not matched:
        frames[1]["U"][0, 0, 0] += 1
    return compare_runs(_record(frames), ref)


def _decide(**kwargs):
    base = dict(woof_status="OK", nonfinite_step=None, referee="scalar",
                comparisons={"scalar": _cmp(kwargs.pop("matched", True))},
                referee_sound=True, designated=True, repeat=None)
    base.update(kwargs)
    return decide(**base)


def test_h4_a_match_against_an_unsound_referee_is_unsound_ref_not_pass():
    assert _decide(referee_sound=False) == "UNSOUND-REF"
    assert _decide(referee_sound=False, matched=False) == "UNSOUND-REF"
    assert _decide() == "PASS"


def test_a_match_that_skipped_a_reference_field_fails():
    ref = _record(_frames(7, extra={"SFROFF": _changing}))
    comparison = compare_runs(_record(_frames(7)), ref)
    assert _decide(comparisons={"scalar": comparison}) == "FAIL"
    assert _decide(comparisons={"scalar": comparison}, designated=False) == "INFO-MISS"


# -- hole 3: WOOF-only runs must reach every planned step -------------------

def test_h5_a_woof_only_run_that_stopped_early_is_not_ok():
    common = dict(woof_status="OK", nonfinite_step=None, referee=None, comparisons={},
                  referee_sound=True, designated=False, repeat=None)
    assert decide(**common, planned_last_step=20, woof_last_step=3) == "WOOF-CRASH"
    assert decide(**common, planned_last_step=20, woof_last_step=None) == "WOOF-CRASH"
    assert decide(**common, planned_last_step=20, woof_last_step=20) == "WOOF-ONLY-OK"


def test_a_repeat_the_combo_asks_for_must_exist():
    common = dict(woof_status="OK", nonfinite_step=None, referee=None, comparisons={},
                  referee_sound=True, designated=False, planned_last_step=3, woof_last_step=3)
    assert decide(**common, repeat=None, repeat_required=True) == "REPEAT-MISS"
    assert decide(**common, repeat=_cmp(True), repeat_required=True) == "WOOF-ONLY-OK"


# -- the real command, --score-only, on a synthetic recording ---------------

COLUMNS = ["build", "case", "combo", "stratum", "status", "steps_requested", "steps_per_frame",
           "last_good_step", "stock461_is_designated_reference", "raw_steps", "refusal", "crash"]


def _write_recording(root: Path, combos: dict, *, summary: dict | None):
    """``combos`` maps id -> dict(stratum, factors, builds={build: (status, last_good,
    frames|None)}, inputs=True)."""
    (root / "provenance" / "sweep").mkdir(parents=True)
    (root / "provenance" / "sweep" / "combos.json").write_text(json.dumps({"combos": [
        {"id": cid, "stratum": spec.get("stratum", "A"), "checks": spec.get("checks", ["finite"]),
         "woof_settings": {}, "factors": spec.get("factors", {})}
        for cid, spec in combos.items()]}))
    (root / "MANIFEST.json").write_text(json.dumps({"unreliable_fields": ["RQIBLTEN"]}))
    with (root / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, COLUMNS)
        writer.writeheader()
        for cid, spec in combos.items():
            if spec.get("inputs", True):
                (root / "inputs" / "c" / cid).mkdir(parents=True)
                (root / "inputs" / "c" / cid / "wrfinput_d01").write_bytes(b"x")
            for build, (status, last_good, frames) in spec["builds"].items():
                writer.writerow({"build": build, "case": "c", "combo": cid,
                                 "stratum": spec.get("stratum", "A"), "status": status,
                                 "steps_requested": spec.get("steps", STEPS - 1),
                                 "steps_per_frame": 1, "last_good_step": last_good,
                                 "stock461_is_designated_reference": "True", "raw_steps": "",
                                 "refusal": "", "crash": ""})
                if frames is not None:
                    record = _record(frames)
                    write_run_record(root / build / "c" / cid, fields=record.fields,
                                     frames=record.digests, steps_per_frame=1)
                    (root / build / "c" / cid / "namelist.input").write_text("&time_control\n/\n")
    if summary is not None:
        with (root / "per-run-summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, ["combo", "case", "clean_reference"])
            writer.writeheader()
            for cid, clean in summary.items():
                writer.writerow({"combo": cid, "case": "c", "clean_reference": clean})


def _write_woof(out: Path, cid: str, frames, status="OK", repeat=False):
    record = _record(frames)
    write_run_record(out / ("woof-repeat" if repeat else "woof") / "c" / cid,
                     fields=record.fields, frames=record.digests, steps_per_frame=1,
                     extra={"woof": {"status": status, "error": "", "wall_s": 1.0,
                                     "nonfinite_step": None, "phases": {}}})


def _run(tmp_path, *argv):
    from gpuwm.verify_exact.cli import register_cli
    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers())
    args = parser.parse_args(["verify-exact", "--recordings", str(tmp_path / "rec"),
                              "--out", str(tmp_path / "out"), "--score-only", *argv])
    code = args.func(args)
    rows = []
    for path in sorted((tmp_path / "out" / "results").glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return code, {r["combo"]: r for r in rows if "combo" in r}


def _scalar(frames, status="OK", last_good=STEPS - 1):
    return {"scalar": (status, last_good, frames)}


def test_cli_identical_against_a_sound_referee_passes_and_exits_zero(tmp_path):
    _write_recording(tmp_path / "rec", {"A001": {"builds": _scalar(_frames(10))}},
                     summary={"A001": "yes"})
    _write_woof(tmp_path / "out", "A001", _frames(10))
    code, rows = _run(tmp_path)
    assert rows["A001"]["outcome"] == "PASS" and code == 0


def test_cli_h1_skipped_reference_field_fails(tmp_path):
    _write_recording(tmp_path / "rec", {"A001": {"builds": _scalar(
        _frames(11, extra={"SFROFF": _changing}))}}, summary={"A001": "yes"})
    _write_woof(tmp_path / "out", "A001", _frames(11))
    code, rows = _run(tmp_path)
    assert rows["A001"]["outcome"] == "FAIL" and code == 1
    assert rows["A001"]["comparisons"]["scalar"]["absent_unexcused"] == ["SFROFF"]


def test_cli_h3_a_crashed_referee_never_gives_pass(tmp_path):
    frames = _frames(12)
    _write_recording(tmp_path / "rec", {"A001": {"builds": _scalar(
        {s: f for s, f in frames.items() if s <= 1}, status="CRASH", last_good=1)}},
        summary={"A001": "yes"})  # even a recording that forgot to mark it
    _write_woof(tmp_path / "out", "A001", frames)
    code, rows = _run(tmp_path)
    assert rows["A001"]["outcome"] == "UNSOUND-REF"
    assert rows["A001"]["referee_soundness"].startswith("incomplete")
    assert code != 0


def test_cli_h4_unsound_referee_match_is_not_pass_and_proves_nothing(tmp_path):
    _write_recording(tmp_path / "rec", {"A001": {"builds": _scalar(_frames(13))}},
                     summary={"A001": "no"})
    _write_woof(tmp_path / "out", "A001", _frames(13))
    code, rows = _run(tmp_path)
    assert rows["A001"]["outcome"] == "UNSOUND-REF"
    assert code == 3


def test_cli_h5_woof_only_run_must_reach_every_planned_step(tmp_path):
    _write_recording(tmp_path / "rec", {"D001": {
        "stratum": "D", "steps": 20, "builds": {"strict": ("WRF-REFUSED", 0, None)}}},
        summary={"D001": "no"})
    _write_woof(tmp_path / "out", "D001", _frames(14, steps=4))  # steps 0..3 of 20
    code, rows = _run(tmp_path)
    assert rows["D001"]["outcome"] == "WOOF-CRASH" and code == 1


def test_cli_h6_a_missing_clean_reference_entry_is_unknown_not_sound(tmp_path):
    _write_recording(tmp_path / "rec", {"A001": {"builds": _scalar(_frames(15))}},
                     summary={})
    _write_woof(tmp_path / "out", "A001", _frames(15))
    code, rows = _run(tmp_path)
    assert rows["A001"]["outcome"] == "UNSOUND-REF"
    assert rows["A001"]["referee_soundness"].startswith("unknown")
    assert code == 3


def test_cli_a_combos_typo_is_a_usage_error(tmp_path):
    _write_recording(tmp_path / "rec", {"A001": {"builds": _scalar(_frames(16))}},
                     summary={"A001": "yes"})
    _write_woof(tmp_path / "out", "A001", _frames(16))
    assert _run(tmp_path, "--combos", "NOSUCHCOMBO")[0] == 2
    assert _run(tmp_path, "--combos", "A001,NOSUCHCOMBO")[0] == 2


def test_cli_only_wrf_refusals_prove_nothing(tmp_path):
    _write_recording(tmp_path / "rec", {"A001": {
        "inputs": False, "builds": {"strict": ("WRF-REFUSED", 0, None)}}}, summary={"A001": "no"})
    code, rows = _run(tmp_path)
    assert rows["A001"]["outcome"] == "WRF-REFUSED"
    assert code == 3


# -- hole 6: coverage over what actually ran against a sound referee -------

def test_coverage_counts_pairs_only_in_runs_integrated_against_a_sound_referee():
    from gpuwm.verify_exact.coverage import pair_coverage
    combos = {"A001": {"stratum": "A", "factors": {"mp": 6, "pbl": 1, "lsm": 2}},
              "A002": {"stratum": "A", "factors": {"mp": 8, "pbl": 5, "lsm": 2}},
              "A003": {"stratum": "A", "factors": {"mp": 28, "pbl": 1, "lsm": 3}}}
    outcomes = {("A001", "c"): "FAIL", ("A002", "c"): "UNSOUND-REF",
                ("A003", "c"): "WRF-REFUSED"}
    (row,) = pair_coverage(combos, outcomes)
    assert row["stratum"] == "A"
    assert row["pairs_in_list"] == 9  # three combos, three disjoint pairs each
    assert row["pairs_integrated_sound"] == 3  # only A001's three pairs
    assert row["pairs_passed"] == 0
    assert row["values_never_integrated_sound"] == {"lsm": [3], "mp": [8, 28], "pbl": [5]}


def test_cli_reports_coverage_over_scored_runs(tmp_path, capsys):
    _write_recording(tmp_path / "rec", {
        "A001": {"factors": {"mp": 6, "pbl": 1}, "builds": _scalar(_frames(17))},
        "A002": {"factors": {"mp": 8, "pbl": 5}, "builds": _scalar(_frames(18))}},
        summary={"A001": "yes", "A002": "no"})
    _write_woof(tmp_path / "out", "A001", _frames(17))
    _write_woof(tmp_path / "out", "A002", _frames(18))
    code, rows = _run(tmp_path)
    assert (rows["A001"]["outcome"], rows["A002"]["outcome"]) == ("PASS", "UNSOUND-REF")
    text = capsys.readouterr().out
    assert "coverage A: pairs 2 in the list; 1 (50.0 %) integrated against a sound referee; " \
           "1 (50.0 %) passed" in text
    cover = json.loads((tmp_path / "out" / "results" / "1of1-score.coverage.json").read_text())
    assert cover[0]["pairs_integrated_sound"] == 1
