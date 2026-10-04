"""Posted handoffs defer future files while retaining native member work."""
from datetime import datetime, timezone
import hashlib
import json

import pytest

from gpuwm import prep_handoff, member_prep
from gpuwm.ensemble.recipes import SourceTrajectory
from gpuwm.fetch_routes import PREP_ARGUMENTS_SCHEMA
from gpuwm.source_adapters import get_source_adapter
from gpuwm.source_posting import SCHEDULE_SCHEMA


def _handoff(tmp_path):
    ready = tmp_path / "p01.f000.grib2"
    future = tmp_path / "p01.f003.grib2"
    ready.write_bytes(b"ready analytical native member fixture")
    inputs = tmp_path / "input-list.txt"
    inputs.write_text(f"{ready}\n{future}\n")
    posting = tmp_path / "posting"
    posting.mkdir()
    schedule = {"schema": SCHEDULE_SCHEMA, "source": "gefs", "member": "p01",
                "cycle": "2024-05-21T12:00:00Z",
                "leads": [{"lead": 0}, {"lead": 3}]}
    (posting / "schedule.json").write_text(json.dumps(schedule))
    grammar = get_source_adapter("gefs").member_set
    document = {"schema": PREP_ARGUMENTS_SCHEMA, "source": "gefs", "prep_source": "gefs",
        "cycle": "2024-05-21T12", "member": "p01", "member_set": grammar,
        "member_verification": {"set": grammar, "member": "p01"},
        "as_posted": True, "posting": str(posting),
        "argv": ["--source", "gefs", "--input-list", str(inputs),
                 "--author-input-manifest", str(tmp_path / "inputs.json")]}
    (tmp_path / "prep-arguments.json").write_text(json.dumps(document))
    return document, ready, future


def test_posted_head_does_not_open_the_absent_future_member(tmp_path, monkeypatch):
    document, ready, future = _handoff(tmp_path)
    monkeypatch.setattr(prep_handoff, "_verify_selected_inputs", lambda *_: pytest.fail("whole-window verification"))
    monkeypatch.setattr(member_prep, "verify_member_file", lambda *_: pytest.fail("payload read at head"))
    argv = prep_handoff.posted_preparation_arguments_from_directory(tmp_path)
    assert argv == document["argv"] + ["--as-posted", str(tmp_path / "posting")]
    assert ready.is_file() and not future.exists()


def test_ready_batch_verifies_only_its_actual_native_member_bytes(tmp_path, monkeypatch):
    _, ready, future = _handoff(tmp_path)
    calls = []
    monkeypatch.setattr(member_prep, "verify_member_file", lambda grammar, member, path:
                        calls.append((member, path)))
    receipt = prep_handoff.verify_posted_member_batch(tmp_path, leads=(0,), primary_files=(ready,))
    assert calls == [("p01", ready)]
    assert receipt["files"] == {str(ready): hashlib.sha256(ready.read_bytes()).hexdigest()}
    assert not future.exists()


def test_wrong_member_ready_batch_propagates_native_refusal(tmp_path, monkeypatch):
    _, ready, future = _handoff(tmp_path)
    def refuse(*_):
        raise ValueError("actual bytes carry p02 instead of p01")
    monkeypatch.setattr(member_prep, "verify_member_file", refuse)
    with pytest.raises(ValueError, match="actual bytes carry p02"):
        prep_handoff.verify_posted_member_batch(tmp_path, leads=(0,), primary_files=(ready,))
    assert not future.exists()


def test_posted_helper_refuses_another_recipe_trajectory(tmp_path):
    _handoff(tmp_path)
    trajectory = SourceTrajectory("gefs", datetime(2024, 5, 21, 12, tzinfo=timezone.utc), "p02")
    with pytest.raises(ValueError, match="requested source trajectory"):
        prep_handoff.posted_preparation_arguments_from_directory(tmp_path, trajectory=trajectory)


def test_posted_helper_refuses_changed_member_grammar(tmp_path):
    document, _, _ = _handoff(tmp_path)
    document["member_verification"]["set"] = "different"
    (tmp_path / "prep-arguments.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="another grammar or member"):
        prep_handoff.posted_preparation_arguments_from_directory(tmp_path)


def test_required_member_staging_runs_for_only_the_ready_leads(tmp_path, monkeypatch):
    document, ready, future = _handoff(tmp_path)
    document["member_prep"] = {"set": document["member_set"], "member": "p01",
        "cycle": document["cycle"], "steps": [0, 3], "inputs": str(tmp_path / "upstream"),
        "output": str(tmp_path / "selected"), "input_list_after": str(tmp_path / "selected-inputs.txt")}
    (tmp_path / "prep-arguments.json").write_text(json.dumps(document))
    staged = tmp_path / "selected-ready.grib2"
    staged.write_bytes(ready.read_bytes())
    selected_list = tmp_path / "selected-ready.txt"
    selected_list.write_text(str(staged) + "\n")
    calls = []
    def stage(narrowed):
        calls.append(narrowed["member_prep"]["steps"])
        return ["--source", "gefs", "--input-list", str(selected_list)]
    monkeypatch.setattr(prep_handoff, "preparation_arguments", stage)
    receipt = prep_handoff.verify_posted_member_batch(tmp_path, leads=(0,), primary_files=(ready,))
    assert calls == [[0]]
    assert receipt["selected_files"] == {str(staged): hashlib.sha256(staged.read_bytes()).hexdigest()}
    assert not future.exists()


def test_selected_member_batch_cannot_substitute_other_bytes(tmp_path, monkeypatch):
    document, ready, _ = _handoff(tmp_path)
    document["member_prep"] = {"set": document["member_set"], "member": "p01",
        "cycle": document["cycle"], "steps": [0, 3], "inputs": str(tmp_path / "upstream"),
        "output": str(tmp_path / "selected"), "input_list_after": str(tmp_path / "selected-inputs.txt")}
    (tmp_path / "prep-arguments.json").write_text(json.dumps(document))
    selected = tmp_path / "selected.grib2"
    selected.write_bytes(b"other bytes")
    selected_list = tmp_path / "selected.txt"
    selected_list.write_text(str(selected) + "\n")
    monkeypatch.setattr(prep_handoff, "preparation_arguments",
                        lambda _: ["--input-list", str(selected_list)])
    with pytest.raises(ValueError, match="differ from the ordinary selected"):
        prep_handoff.verify_posted_member_batch(tmp_path, leads=(0,), primary_files=(ready,))
