"""`gpuwm verify-exact` comparison logic, on CPU, with synthetic runs.

What breaks without these: the combo sweep's PASS lines are its only
evidence, so a digest that disagrees with the recording's own (a dtype or
byte-order slip), a first-divergence that picks the wrong step, or a ULP
that hides a NaN would turn every line of the sweep into a false claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime

import numpy as np
import pytest

from gpuwm.core.fp32_ulp import MISMATCH
from gpuwm.verify_exact.compare import (FieldMeta, RunRecord, compare_runs, field_class,
                                        field_digest, ulp_detail)
from gpuwm.verify_exact.recording import Recording, load_run_record, write_run_record
from gpuwm.verify_exact.scoring import FAIL_OUTCOMES, decide


def _recorded_sha(array) -> str:
    """The recording's own rule (combo-sweep scripts/extract.py)."""
    a = np.ascontiguousarray(array).astype(np.asarray(array).dtype.newbyteorder("<"), copy=False)
    return hashlib.sha256(a.tobytes()).hexdigest()


def test_digest_is_the_recordings_digest_for_float_and_int_fields():
    t = np.linspace(-3, 3, 24, dtype=np.float32).reshape(2, 3, 4)
    k = np.arange(12, dtype=np.int32).reshape(3, 4)
    assert field_digest(t, "<f4", (2, 3, 4)) == _recorded_sha(t)
    assert field_digest(k, "<i4", (3, 4)) == _recorded_sha(k)
    # a reader that widened float32 to float64 hashes back to the same words
    assert field_digest(t.astype(np.float64), "<f4", (2, 3, 4)) == _recorded_sha(t)
    assert field_digest(k.astype(np.float64), "<i4", (3, 4)) == _recorded_sha(k)
    # big-endian input is stored little-endian, as the recording stores it
    assert field_digest(t.astype(">f4"), "<f4", (2, 3, 4)) == _recorded_sha(t)


def test_digest_refuses_what_the_recorded_form_cannot_hold():
    t = np.zeros((2, 2), np.float32)
    assert field_digest(t, "<f4", (2, 3)) is None
    assert field_digest(np.array([[0.1, 0.0], [0.0, 0.0]]), "<f4", (2, 2)) is None
    assert field_digest(np.array([[0.5, 1.0], [2.0, 2.5]]), "<i4", (2, 2)) is None


def test_field_classes_follow_the_case_comparison_list():
    assert field_class("U", 3) == "state"
    assert field_class("QNRAIN", 3) == "state"
    assert field_class("QC_BL", 3) == "physics"
    assert field_class("Q2", 2) == "diagnostic"
    assert field_class("QFX", 2) == "physics"
    assert field_class("HFX", 2) == "physics"
    assert field_class("XLAT", 2) == "static"
    assert field_class("REFL_10CM", 3) == "diagnostic"


def _record(frames: dict[int, dict[str, np.ndarray]], label="", raw_steps=()):
    fields = {}
    digests = {}
    for step, arrays in frames.items():
        row = {}
        for name, values in arrays.items():
            meta = fields.setdefault(name, FieldMeta(name, values.dtype.str, values.shape))
            row[name] = field_digest(values, meta.dtype, meta.shape)
        digests[step] = row
    kept = {s: frames[s] for s in raw_steps}
    return RunRecord(fields, digests, kept.get, tuple(raw_steps), label)


def _state(seed: int, steps=range(0, 4)):
    rng = np.random.default_rng(seed)
    base = {"U": rng.standard_normal((3, 4, 5)).astype(np.float32),
            "HFX": rng.standard_normal((4, 4)).astype(np.float32),
            "T2": rng.standard_normal((4, 4)).astype(np.float32),
            "RQIBLTEN": rng.standard_normal((3, 4, 4)).astype(np.float32),
            "KPBL": rng.integers(0, 9, (4, 4)).astype(np.int32)}
    return {s: {k: v.copy() for k, v in base.items()} for s in steps}


def test_identical_runs_match_at_every_step():
    ref = _record(_state(1), "scalar")
    woof = _record(_state(1), "woof")
    result = compare_runs(woof, ref)
    assert result.matched and result.first_step is None
    assert result.steps_compared == (0, 1, 2, 3)
    assert result.summary() == "match"


def test_first_divergence_is_the_earliest_step_and_state_leads_the_report():
    ref_frames = _state(2)
    woof_frames = _state(2)
    woof_frames[2]["HFX"][1, 1] = np.nextafter(woof_frames[2]["HFX"][1, 1], np.float32(9))
    woof_frames[2]["U"][0, 0, 0] = np.nextafter(woof_frames[2]["U"][0, 0, 0], np.float32(9))
    woof_frames[3]["T2"][0, 0] += 1
    ref = _record(ref_frames, "scalar", raw_steps=(2,))
    woof = _record(woof_frames, "woof", raw_steps=(2,))
    result = compare_runs(woof, ref)
    assert result.first_step == 2
    assert result.first_fields == ("U", "HFX")
    assert result.lead_field == "U"
    detail = result.lead_detail()
    assert (detail.step, detail.max_ulp, detail.differing_words, detail.first[0]) == (2, 1, 1, (0, 0, 0))
    assert result.differing == {"U": 1, "HFX": 1, "T2": 1}
    assert "step 2 U (+1 more) max 1 ulp (|diff| <= " in result.summary()
    assert not result.diagnostics_only


def test_ulp_is_measured_at_the_next_kept_step_when_the_first_was_not_kept():
    ref_frames = _state(3)
    woof_frames = _state(3)
    for step in (1, 2, 3):
        woof_frames[step]["HFX"][0, 3] = np.float32(woof_frames[step]["HFX"][0, 3]) * np.float32(1.0001)
    ref = _record(ref_frames, "scalar", raw_steps=(3,))
    woof = _record(woof_frames, "woof", raw_steps=(1, 2, 3))
    result = compare_runs(woof, ref)
    assert result.first_step == 1
    assert result.lead_detail().step == 3
    assert "at step 3" in result.summary()


def test_without_reference_arrays_the_summary_says_so_rather_than_inventing_a_ulp():
    ref_frames, woof_frames = _state(4), _state(4)
    woof_frames[1]["U"][1, 1, 1] += 1
    result = compare_runs(_record(woof_frames, raw_steps=(1,)), _record(ref_frames))
    assert result.first_step == 1
    assert result.lead_detail() is None
    assert "needs the reference's raw arrays" in result.summary()


def test_excluded_fields_are_not_compared_and_absent_fields_block_a_match():
    ref_frames, woof_frames = _state(5), _state(5)
    woof_frames[1]["RQIBLTEN"][:] = np.nan
    result = compare_runs(_record(woof_frames), _record(ref_frames), exclude=("RQIBLTEN",))
    assert result.matched
    assert result.excluded == ("RQIBLTEN",)
    for frame in woof_frames.values():
        del frame["KPBL"]
    result = compare_runs(_record(woof_frames), _record(ref_frames), exclude=("RQIBLTEN",))
    assert not result.matched  # KPBL was never compared, so this is not exact
    assert result.absent_in_candidate == result.absent_unexcused == ("KPBL",)


def test_a_crashed_reference_is_compared_only_through_its_last_good_step():
    ref_frames, woof_frames = _state(6), _state(6)
    ref_frames[3]["U"][:] = np.nan  # WRF's blow-up frame
    result = compare_runs(_record(woof_frames), _record(ref_frames), through_step=2)
    assert result.matched and result.steps_compared == (0, 1, 2)


def test_a_candidate_that_stopped_early_is_not_a_match():
    ref_frames = _state(7)
    woof_frames = {s: v for s, v in _state(7).items() if s <= 1}
    result = compare_runs(_record(woof_frames), _record(ref_frames))
    assert not result.matched
    assert result.missing_steps == (2, 3)
    assert "candidate stopped before step 2" in result.summary()


def test_diagnostic_only_differences_are_labelled():
    ref_frames, woof_frames = _state(8), _state(8)
    woof_frames[1]["T2"][2, 2] += 1
    result = compare_runs(_record(woof_frames), _record(ref_frames))
    assert result.diagnostics_only
    assert "[diagnostics/static only]" in result.summary()


def test_ulp_detail_never_lets_a_nan_or_a_signed_zero_pass():
    want = np.array([1.0, 0.0, 2.0], np.float32)
    got = np.array([1.0, -0.0, np.nan], np.float32)
    detail = ulp_detail("X", 1, got, want)
    assert detail.differing_words == 2  # -0.0 is a different word
    assert detail.max_ulp == MISMATCH
    assert detail.worst[0] == (2,)
    assert detail.nonfinite
    assert "NaN mismatch" in detail.describe()


def test_a_sign_crossing_reports_its_absolute_size_beside_its_ulp():
    detail = ulp_detail("U", 1, np.array([-1e-4, 2.0], np.float32), np.array([7e-4, 2.0], np.float32))
    assert detail.max_ulp > 10**9
    assert abs(detail.max_abs - 8e-4) < 1e-9
    assert "|diff| <= 0.0008" in detail.describe()


def test_integer_fields_report_their_integer_gap():
    detail = ulp_detail("KPBL", 1, np.array([3, 4, 9], np.int32), np.array([3, 5, 6], np.int32))
    assert (detail.max_ulp, detail.differing_words, detail.first[0], detail.worst[0]) == (3, 2, (1,), (2,))


def test_a_written_record_reads_back_with_its_digests_and_kept_arrays(tmp_path):
    frames = _state(9)
    record = _record(frames, raw_steps=(1,))
    write_run_record(tmp_path / "run", fields=record.fields, frames=record.digests,
                     steps_per_frame=1, raw={1: {"U": frames[1]["U"]}},
                     extra={"woof": {"status": "OK"}})
    back = load_run_record(tmp_path / "run")
    assert back.digests == record.digests
    assert back.raw_steps == (1,)
    assert np.array_equal(back.raw(1)["U"], frames[1]["U"])
    assert back.raw(2) is None
    assert compare_runs(back, record).matched


def test_frames_every_two_steps_map_to_even_steps(tmp_path):
    frames = {0: {"U": np.zeros((2, 2), np.float32)}, 2: {"U": np.ones((2, 2), np.float32)}}
    record = _record(frames)
    write_run_record(tmp_path / "r", fields=record.fields, frames=record.digests, steps_per_frame=2)
    assert load_run_record(tmp_path / "r").steps == (0, 2)
    with pytest.raises(ValueError):
        write_run_record(tmp_path / "bad", fields=record.fields, frames={1: {}}, steps_per_frame=2)


def test_an_overlay_supplies_the_reference_arrays_the_copy_lacks(tmp_path):
    frames = _state(10)
    record = _record(frames)
    write_run_record(tmp_path / "rec", fields=record.fields, frames=record.digests, steps_per_frame=1)
    write_run_record(tmp_path / "overlay", fields=record.fields, frames={}, steps_per_frame=1,
                     raw={1: {"HFX": frames[1]["HFX"]}})
    (tmp_path / "overlay" / "hashes.json").unlink()
    back = load_run_record(tmp_path / "rec", overlays=[tmp_path / "overlay"])
    assert back.raw_steps == (1,)
    assert np.array_equal(back.raw(1)["HFX"], frames[1]["HFX"])


def _fake_recording(root, builds):
    """A recording root with one combo on one case and the given builds."""
    (root / "provenance" / "sweep").mkdir(parents=True)
    (root / "provenance" / "sweep" / "combos.json").write_text(json.dumps({"combos": [
        {"id": "A001", "stratum": "A", "checks": ["wrf_0ulp_every_step", "finite"],
         "woof_settings": {}, "factors": {"mp": 8}}]}))
    (root / "MANIFEST.json").write_text(json.dumps({"unreliable_fields": ["RQIBLTEN"]}))
    (root / "inputs" / "c" / "A001").mkdir(parents=True)
    (root / "inputs" / "c" / "A001" / "wrfinput_d01").write_bytes(b"x")
    columns = ["build", "case", "combo", "status", "steps_requested", "steps_per_frame",
               "last_good_step", "stock461_is_designated_reference", "raw_steps", "refusal", "crash"]
    with (root / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for build, status in builds.items():
            writer.writerow({"build": build, "case": "c", "combo": "A001", "status": status,
                             "steps_requested": 3, "steps_per_frame": 1, "last_good_step": 3,
                             "stock461_is_designated_reference": "True", "raw_steps": "1 3",
                             "refusal": "", "crash": ""})
            record = _record(_state(11))
            write_run_record(root / build / "c" / "A001", fields=record.fields,
                             frames=record.digests, steps_per_frame=1)
            (root / build / "c" / "A001" / "namelist.input").write_text("&time_control\n/\n")


def test_the_thompson_fixed_build_referees_where_it_was_recorded(tmp_path):
    _fake_recording(tmp_path, {"strict": "OK", "scalar": "OK", "strict-tfix": "OK",
                               "scalar-tfix": "OK"})
    recording = Recording(tmp_path)
    (run,) = recording.select()
    assert run.referee("scalar").build == "scalar-tfix"
    assert run.referee("strict").build == "strict-tfix"
    assert sorted(run.scorable()) == ["scalar", "scalar-tfix", "strict", "strict-tfix"]
    assert run.references["scalar"].raw_steps == (1, 3)
    assert recording.unreliable_fields == ("RQIBLTEN",)


def test_raw_steps_off_the_history_cadence_are_not_promised(tmp_path):
    _fake_recording(tmp_path, {"scalar": "OK"})
    table = (tmp_path / "runs.csv").read_text().replace(",1,3,True,1 3,", ",2,3,True,1 2 5 20,")
    (tmp_path / "runs.csv").write_text(table)
    (run,) = Recording(tmp_path).select()
    assert run.references["scalar"].raw_steps == (2, 20)


def test_without_a_fixed_build_the_stock_build_referees(tmp_path):
    _fake_recording(tmp_path, {"strict": "OK", "scalar": "OK"})
    (run,) = Recording(tmp_path).select(combos=["A001"])
    assert run.referee("scalar").build == "scalar"
    assert Recording(tmp_path).select(combos=["A002"]) == []


def _cmp(matched: bool, missing=()):
    ref = _record(_state(12))
    frames = _state(12)
    if not matched:
        frames[1]["U"][0, 0, 0] += 1
    frames = {s: v for s, v in frames.items() if s not in missing}
    return compare_runs(_record(frames), ref)


@pytest.mark.parametrize("kwargs, outcome", [
    (dict(woof_status="OK", referee="scalar", cmp=True), "PASS"),
    (dict(woof_status="OK", referee="scalar", cmp=False), "FAIL"),
    (dict(woof_status="REFUSED", referee="scalar", cmp=True), "WOOF-REFUSED"),
    (dict(woof_status="CRASH", referee="scalar", cmp=True, missing=(3,)), "WOOF-CRASH"),
    (dict(woof_status="OK", referee="scalar", cmp=False, sound=False), "UNSOUND-REF"),
    (dict(woof_status="OK", referee="scalar", cmp=False, designated=False), "INFO-MISS"),
    (dict(woof_status="OK", referee="scalar", cmp=True, designated=False), "INFO-MATCH"),
    (dict(woof_status="OK", referee=None), "WOOF-ONLY-OK"),
    (dict(woof_status="OK", referee=None, nonfinite=4), "WOOF-NONFINITE"),
    (dict(woof_status="OK", referee="scalar", cmp=True, repeat=False), "REPEAT-MISS"),
])
def test_outcomes(kwargs, outcome):
    comparisons = {}
    if kwargs.get("referee"):
        comparisons["scalar"] = _cmp(kwargs["cmp"], kwargs.get("missing", ()))
    repeat = None if "repeat" not in kwargs else _cmp(kwargs["repeat"])
    got = decide(woof_status=kwargs["woof_status"], nonfinite_step=kwargs.get("nonfinite"),
                 referee=kwargs.get("referee"), comparisons=comparisons,
                 referee_sound=kwargs.get("sound", True),
                 designated=kwargs.get("designated", True), repeat=repeat)
    assert got == outcome
    assert (got in FAIL_OUTCOMES) == (outcome in {"FAIL", "WOOF-REFUSED", "WOOF-CRASH",
                                                  "WOOF-NONFINITE", "REPEAT-MISS"})


def test_frame_files_map_valid_times_to_steps(tmp_path):
    from gpuwm.verify_exact.replay import frame_files
    for stamp in ("2024-05-21_21_00_00", "2024-05-21_21_00_15", "2024-05-21_21_00_30"):
        (tmp_path / f"wrfout_d01_{stamp}").write_bytes(b"")
    (tmp_path / "wrfout_d02_2024-05-21_21_00_00").write_bytes(b"")
    frames = frame_files(tmp_path, start=datetime(2024, 5, 21, 21), step_seconds=15.0)
    assert sorted(frames) == [0, 1, 2]
    halves = frame_files(tmp_path, start=datetime(2024, 5, 21, 21), step_seconds=7.5)
    assert sorted(halves) == [0, 2, 4]
    with pytest.raises(ValueError):
        frame_files(tmp_path, start=datetime(2024, 5, 21, 21), step_seconds=10.0)


def test_soundness_follows_the_recordings_clean_reference_column(tmp_path):
    import dataclasses
    _fake_recording(tmp_path, {"scalar": "OK", "scalar-tfix": "OK"})
    (run,) = Recording(tmp_path).select()
    stock, fixed = run.references["scalar"], run.references["scalar-tfix"]
    assert not run.referee_sound(stock)  # no summary: unknown is not sound
    assert run.referee_soundness(stock).startswith("unknown")
    assert run.referee_sound(dataclasses.replace(stock, clean="yes"))
    assert not run.referee_sound(dataclasses.replace(stock, clean="no"))
    thompson = "yes, with the Thompson fix build"
    assert not run.referee_sound(dataclasses.replace(stock, clean=thompson))
    assert run.referee_sound(dataclasses.replace(fixed, clean=thompson))


def test_the_raw_subset_tool_cuts_only_the_named_words(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    frames = _state(13)
    record = _record(frames)
    run_dir = tmp_path / "rec" / "scalar" / "c" / "A001"
    write_run_record(run_dir, fields=record.fields, frames=record.digests, steps_per_frame=1,
                     raw={1: {n: frames[1][n] for n in ("U", "HFX", "T2")}})
    needs = tmp_path / "needs.json"
    needs.write_text(json.dumps([{"build": "scalar", "case": "c", "combo": "A001", "step": 1,
                                  "fields": ["HFX", "U"]}]))
    tool = Path(__file__).resolve().parents[1] / "tools" / "verify_exact_raw_subset.py"
    subprocess.run([sys.executable, str(tool), str(tmp_path / "rec"), str(needs),
                    str(tmp_path / "overlay")], check=True, capture_output=True)
    cut = load_run_record(run_dir, overlays=[tmp_path / "overlay" / "scalar" / "c" / "A001"])
    overlay_only = cut.raw(1)
    assert set(overlay_only) == {"U", "HFX", "T2"}  # recording first, overlay fills gaps
    blob = (tmp_path / "overlay" / "scalar" / "c" / "A001" / "step0001.json").read_text()
    assert [f["name"] for f in json.loads(blob)["fields"]] == ["HFX", "U"]


def test_a_static_difference_at_the_initial_frame_does_not_hide_where_state_left():
    ref_frames, woof_frames = _state(14), _state(14)
    for frame in (*ref_frames.values(), *woof_frames.values()):
        frame["XLAT"] = np.zeros((4, 4), np.float32)
    for step in woof_frames:
        woof_frames[step]["XLAT"][0, 0] = np.float32(1e-7)
    for step in (2, 3):
        woof_frames[step]["U"][1, 2, 3] += np.float32(0.5)
    result = compare_runs(_record(woof_frames, raw_steps=(2,)), _record(ref_frames, raw_steps=(2,)))
    assert (result.first_step, result.lead_field) == (0, "XLAT")
    assert result.state_first == (2, ("U",))
    assert result.detail_for("U").step == 2
    assert "; state from step 2 U max" in result.summary()


def test_a_field_absent_at_one_frame_on_both_sides_is_not_a_difference(tmp_path):
    frames = _state(15)
    del frames[0]["T2"]  # neither run wrote T2 in its first frame
    record = _record(frames)
    write_run_record(tmp_path / "a", fields=record.fields, frames=record.digests, steps_per_frame=1)
    write_run_record(tmp_path / "b", fields=record.fields, frames=record.digests, steps_per_frame=1)
    back_a, back_b = load_run_record(tmp_path / "a"), load_run_record(tmp_path / "b")
    assert "T2" not in back_a.digests[0]
    assert compare_runs(back_a, back_b).matched
    lost = _record({s: {k: v for k, v in f.items() if not (s == 2 and k == "U")}
                    for s, f in _state(15).items()})
    assert compare_runs(lost, _record(_state(15))).first_step == 2


def test_a_field_written_in_another_shape_is_listed_and_blocks_a_match():
    ref = _record(_state(16))
    frames = _state(16)
    for frame in frames.values():
        frame["HFX"] = np.zeros((5, 4), np.float32)
    result = compare_runs(_record(frames), ref)
    assert not result.matched and result.first_step is None
    assert result.reshaped == {"HFX": ([5, 4], [4, 4])}
    assert "HFX" not in result.fields_compared


def test_a_named_refusal_before_the_first_frame_is_a_refusal_not_a_crash():
    from gpuwm.verify_exact.cli import initialisation_refusal
    empty = RunRecord({}, {})
    assert initialisation_refusal("CRASH", "ValueError: RUC smois must remain within 0..1", empty) == (
        "REFUSED", "at initialisation: ValueError: RUC smois must remain within 0..1")
    assert initialisation_refusal("CRASH", "RuntimeError: non-finite", empty)[0] == "CRASH"
    started = _record(_state(17))
    assert initialisation_refusal("CRASH", "ValueError: late", started)[0] == "CRASH"
