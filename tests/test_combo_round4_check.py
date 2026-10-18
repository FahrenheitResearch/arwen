import pytest

from tools.combo_round4_check import assert_equal_errors, control_signature, plan_pairs, validate_sweep


def combo(cid, cu, **extra):
    return {"id": cid, "factors": {"cu": cu, "mp": 6, **extra},
            "namelist": {"physics": {"cu_physics": cu, "cudt": 0}}}


def test_control_must_match_all_settings():
    assert control_signature(combo("k", 1)) == control_signature(combo("c", 0))
    assert control_signature(combo("k", 1)) != control_signature(combo("c", 0, epssm=0.1))
    other = combo("c", 0)
    other["namelist"]["physics"]["cudt"] = 5
    assert control_signature(combo("k", 1)) != control_signature(other)


def test_missing_control_fails_instead_of_choosing_nearest_family():
    rows = [{"combo": "k", "case": "fixture", "cu": "1", "outcome": "FAIL"}]
    pairs, errors = plan_pairs(rows, {"k": combo("k", 1), "c": combo("c", 0)})
    assert not pairs
    assert errors == ["no recorded identical no-cumulus control: k/fixture"]


def test_recorded_pair_is_selected_and_empty_selection_fails():
    rows = [{"combo": "k", "case": "fixture", "cu": "1", "outcome": "FAIL"},
            {"combo": "c", "case": "fixture", "cu": "0", "outcome": "FAIL"}]
    pairs, errors = plan_pairs(rows, {"k": combo("k", 1), "c": combo("c", 0)})
    assert pairs == [{"case": "fixture", "kf": "k", "control": "c"}]
    assert not errors
    assert plan_pairs([], {})[1]


@pytest.mark.parametrize("kf,control", [(0.0139, 0.0016), (float("nan"), 0), (0, float("inf"))])
def test_nonidentical_or_nonfinite_errors_fail(kf, control):
    with pytest.raises(AssertionError):
        assert_equal_errors(kf, control)


def test_exact_errors_pass():
    assert_equal_errors(0.001953125, 0.001953125)


def test_new_control_can_be_recorded_without_changing_the_frozen_target_list():
    rows = [{"combo": "k", "case": "fixture", "cu": "1", "outcome": "FAIL"}]
    pairs, errors = plan_pairs(rows, {"k": combo("k", 1), "c": combo("c", 0)},
                               available_runs={("fixture", "k"), ("fixture", "c")})
    assert pairs == [{"case": "fixture", "kf": "k", "control": "c"}]
    assert not errors


def test_acceptance_does_not_pass_without_the_full_sweep():
    with pytest.raises(AssertionError, match="every round-3 run"):
        validate_sweep([dict(combo="fixture", case="fixture")], [], "", {})


def test_above_floor_family_requires_named_cause_and_real_report_tables():
    from tools.combo_round4_summary import report
    previous = [dict(combo="fixture", case="fixture", outcome="FAIL", mp="6", pbl="1",
                     sfclay="91", lsm="2", cu="1", rad="off")]
    current = [dict(combo="fixture", case="fixture", outcome="FAIL", referee="scalar",
                    factors={"cu": 1}, comparisons={"scalar": {"details": [
                        {"field": "U", "step": 1, "max_abs_diff": 0.014}]}})]
    text = report(previous, current)
    with pytest.raises(AssertionError, match="above-floor family"):
        validate_sweep(previous, current, text, {})
    causes = {f"{family}/{value}": dict(cause="fixture cause", repro_command="fixture command",
                                        code_site="fixture.py:1", evidence="fixture.json")
              for family, value in (("cu", "1"), ("mp", "None"), ("pbl", "None"),
                                    ("sfclay", "None"), ("lsm", "None"), ("rad", "None"))}
    validate_sweep(previous, current, text, causes)
    with pytest.raises(AssertionError, match="tables"):
        validate_sweep(previous, current, "not a measured report", causes)


def wire_records(tmp_path, candidate_values, *, keep_raw=True, status="OK"):
    """Real serialized records for the measurement gate; no model claim."""
    from types import SimpleNamespace
    import numpy as np
    from gpuwm.verify_exact.compare import FieldMeta, field_digest
    from gpuwm.verify_exact.recording import load_run_record, write_run_record

    reference = np.array([1, 2, 3], dtype=np.float32)
    candidate = np.asarray(candidate_values, dtype=np.float32)
    meta = FieldMeta("U", "<f4", (3,))
    ref_dir = tmp_path / "reference"
    out = tmp_path / "out"
    cand_dir = out / "woof/fixture/fixture"
    for directory, values, raw, extra in (
        (ref_dir, reference, {1: {"U": reference}}, {}),
        (cand_dir, candidate, {1: {"U": candidate}} if keep_raw else {}, {"woof": {"status": status}}),
    ):
        write_run_record(directory, fields={"U": meta},
                         frames={1: {"U": field_digest(values, meta.dtype, meta.shape)}},
                         steps_per_frame=1, raw=raw, extra=extra)
    ref = SimpleNamespace(designated=True)
    run = SimpleNamespace(referee=lambda _: ref, referee_sound=lambda _: True)
    recording = SimpleNamespace(runs={("fixture", "fixture"): run},
                                reference_record=lambda _: load_run_record(ref_dir))
    return recording, out, cand_dir


def test_equal_digest_proves_zero_when_replay_did_not_keep_equal_raw(tmp_path):
    from tools.combo_round4_check import u_error
    recording, out, _ = wire_records(tmp_path, [1, 2, 3], keep_raw=False)
    assert u_error(recording, "fixture", "fixture", out) == 0


def test_measurement_uses_real_record_words_and_rejects_stale_raw(tmp_path):
    import numpy as np
    from tools.combo_round4_check import u_error
    recording, out, directory = wire_records(tmp_path, [1.25, 2, 3])
    assert u_error(recording, "fixture", "fixture", out) == 0.25
    (directory / "step0001.bin").write_bytes(np.array([1, 2, 3], np.float32).tobytes())
    with pytest.raises(AssertionError, match="does not match this replay"):
        u_error(recording, "fixture", "fixture", out)


def test_crashed_run_cannot_reuse_old_measurements(tmp_path):
    from tools.combo_round4_check import u_error
    recording, out, _ = wire_records(tmp_path, [1, 2, 3], status="CRASH")
    with pytest.raises(AssertionError, match="did not finish"):
        u_error(recording, "fixture", "fixture", out)
