"""``--run-seconds`` is the length the prepared door integrates.

The door used to read the flag, warn that the hash-bound experiment value
"is authoritative and is used", and run the whole prepared length: a
4200 s request on a 12 h preparation integrated 12 h (box S, 2026-10-06).
The prepared state never depended on the length (``run.run_seconds`` is a
non-trajectory identity field), so the requested length is executed, the
override is recorded in the execution plan, and the one real gate -- the
prepared boundary forcing must reach the run -- refuses by naming the
last prepared lead.
"""
import json

import pytest

from gpuwm import prepared_single_domain_forecast as runner
from gpuwm.experiment import load_experiment
from test_prepared_single_domain_forecast import (
    _bind_synthetic_preflight_geometry, _prepared_fixture, _sha256)


def _preflight(fixture, run_seconds):
    return runner.preflight_prepared_forecast(
        source=fixture.source, prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.RUC_PHYSICS_PROFILE,
        run_seconds=run_seconds,
        history_interval_seconds=load_experiment(
            fixture.experiment).root.history_interval_s)


def _hashes(fixture):
    return {"proof": _sha256(fixture.proof),
            "experiment": _sha256(fixture.experiment),
            "header": _sha256(fixture.prepared / "prepared-cache" / "header.json")}


def test_a_shorter_run_seconds_is_the_executed_length(tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "era5", physics_profile=runner.RUC_PHYSICS_PROFILE)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    prepared_seconds = float(fixture.run_seconds)
    requested = 3600.0
    assert requested < prepared_seconds
    before = _hashes(fixture)
    whole = _preflight(fixture, prepared_seconds)
    short = _preflight(fixture, requested)

    # Both timing authorities carry the requested length, together.
    assert float(short.experiment.run_seconds) == requested
    assert float(short.experiment.root.run.run_seconds) == requested
    assert float(whole.experiment.run_seconds) == prepared_seconds

    # The prepared state is the same state; nothing bound to it moved.
    assert short.cache_identity == whole.cache_identity
    assert short.file_sha256 == whole.file_sha256
    assert _hashes(fixture) == before

    plan = short.execution_plan
    assert plan["source_experiment"]["run_seconds"] == prepared_seconds
    assert plan["executed_d01"]["run_seconds"] == requested
    rows = [row for row in plan["execution_overrides"] if row["kind"] == "run-length"]
    assert rows == [{
        "kind": "run-length",
        "source_experiment_seconds": prepared_seconds,
        "executed_seconds": requested,
        "mechanism": runner.RUN_LENGTH_OVERRIDE_MECHANISM,
        "prepared_state_changed": False,
        "model_state_or_physics_changed": False,
    }]
    assert plan["physics_overrides"] == []
    # An equal request records no override.
    assert not [row for row in whole.execution_plan["execution_overrides"]
                if row["kind"] == "run-length"]

    # The history schedule is planned on the executed length: the start
    # frame and the one cadence that lands at or before 3600 s.
    cadence = load_experiment(fixture.experiment).root.history_interval_s
    receipt = runner._validate_hash_bound_history_cadence(short.experiment, cadence)
    assert receipt["run_end_offset_seconds"] == requested
    assert receipt["expected_frame_count"] == 1 + int(requested // cadence)


def test_a_run_past_the_prepared_forcing_refuses_naming_the_last_lead(tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "era5", physics_profile=runner.RUC_PHYSICS_PROFILE)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    forcing_hours = json.loads(fixture.proof.read_text(encoding="utf-8"))["forcing_hours"]
    last = int(forcing_hours[-1])
    with pytest.raises(ValueError) as refusal:
        _preflight(fixture, float((last + 1) * 3600))
    text = str(refusal.value)
    assert f"ends at lead {last} h" in text
    assert "step past its last prepared boundary tendency" in text
    assert f"--run-seconds {last * 3600:g}" in text


def test_validate_physics_no_longer_silently_substitutes_the_experiment_length(tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "era5", physics_profile=runner.RUC_PHYSICS_PROFILE)
    exp = load_experiment(fixture.experiment)
    with pytest.raises(ValueError, match="apply the run length to the experiment"):
        runner._validate_physics(exp, None, float(exp.run_seconds) / 2, 3600, source="era5")


def test_runner_capabilities_say_the_requested_length_is_executed():
    window = runner.runner_capabilities()["window"]["run_seconds"]
    assert window["must_equal_hash_bound_experiment"] is False
    assert window["requested_length_executed"] is True
    assert window["override_mechanism"] == runner.RUN_LENGTH_OVERRIDE_MECHANISM
