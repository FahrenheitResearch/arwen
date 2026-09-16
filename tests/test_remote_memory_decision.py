"""One sizing statement, both remote doors, and no card is ever refused for it.

A remote memory figure is an estimate made before the run exists. Every door
states it, prices an unmeasured card from recorded capacity, and launches the
configuration the reader asked for. CPU-only: no card is opened, no forecast is
planned and no probe subprocess runs.
"""
from datetime import datetime

import pytest

from gpuwm import remote_plan as rp, remote_worker as rw


MEASURED_TOO_SMALL = {"measured": True, "refuse": True, "warn": False, "verdict": "native refusal",
                      "peak_envelope_bytes": 9_000_000_000, "free_bytes": 8_000_000_000}
UNMEASURED = {"measured": False, "refuse": False, "warn": True, "free_bytes": None,
              "verdict": "no card here", "probe_reason": "the device probe is not installed",
              "peak_envelope_bytes": 1_000_000}
PROBE = {"devices": [{"memory_free_bytes": 8_000_000_000}, {"memory_free_bytes": 12_000_000_000}],
         "device_query_basis": "NVML device rows read through nvidia-smi"}


def _advised(memory, probe=None):
    value = dict(memory)
    value.update(rp.memory_advice(value, probe))
    return value


def _saved_config(tmp_path, name="measured.toml"):
    from gpuwm import domain_wizard as dw
    source = tmp_path / name
    source.write_text(dw.render_config(name="memory-decision", start_time=datetime(2026, 9, 5),
        hours=3, projection=dw._projection_entries(40, -100, "auto"), dims=dw._dims_for_scale(1, ()),
        ratios=(), fetch_hints=dict(source="gfs", cycle="2026-09-05T00", hours=3,
                                    out="data/cache", cadence=3), case_data=None), encoding="utf-8")
    return source


def _launch(monkeypatch, tmp_path, memory):
    review = {"memory": memory, "plan_sha256": "a" * 64, "config_sha256": "b" * 64,
              "input_sha256": "c" * 64, "geog_root": None,
              "entry": {"door": "run-plan", "document": str(tmp_path / "plan.json"), "flags": []}}
    monkeypatch.setattr(rp, "review", lambda *_: (review, {"files": [], "geog_root": None}, tmp_path))
    launched = []
    monkeypatch.setattr(rw, "_launch_review", lambda *args, **kwargs: launched.append(args[2]) or {"job": {}})
    request = {"expected_plan_sha256": "a" * 64, "expected_config_sha256": "b" * 64,
               "expected_input_sha256": "c" * 64}
    return request, launched


def test_an_unprobeable_node_is_priced_and_warned_rather_than_blocked(monkeypatch, tmp_path):
    memory = _advised(UNMEASURED, PROBE)
    assert memory["advice"] is None and memory["warn"] is True
    assert memory["basis"] == "NVML device rows read through nvidia-smi"
    # The most conservative recorded capacity is the basis, and it is stated.
    assert memory["priced_free_bytes"] == 8_000_000_000
    assert "8000000000" in memory["priced_note"]
    assert "1000000" in memory["priced_note"]
    assert "the device probe is not installed" in memory["priced_note"]
    request, launched = _launch(monkeypatch, tmp_path, memory)
    assert rp.launch(request, tmp_path) == {"job": {}}
    assert len(launched) == 1


def test_a_node_with_no_recorded_device_capacity_still_launches(monkeypatch, tmp_path):
    memory = _advised(UNMEASURED, {"devices": []})
    assert memory["advice"] is None and memory["warn"] is True
    assert memory["basis"] == "no recorded device capacity"
    assert "no recorded basis" in memory["priced_note"]
    request, launched = _launch(monkeypatch, tmp_path, memory)
    assert rp.launch(request, tmp_path) == {"job": {}}
    assert len(launched) == 1


def test_a_measured_card_that_looks_too_small_is_advised_with_both_byte_counts_and_launched(monkeypatch, tmp_path):
    """The estimate is stated; the configuration the reader asked for still runs."""
    memory = _advised(MEASURED_TOO_SMALL, PROBE)
    assert memory["basis"] == "measured device probe" and memory["warn"] is True
    advice = memory["advice"]
    assert "9000000000" in advice and "8000000000" in advice
    assert "not launch-ready" not in advice
    assert "Free memory on that card" in advice and "launched as requested" in advice
    request, launched = _launch(monkeypatch, tmp_path, memory)
    assert rp.launch(request, tmp_path) == {"job": {}}
    assert launched[0]["memory"]["advice"] == advice


def test_a_memory_review_that_cannot_be_computed_is_stated_not_a_silent_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "memory_review", lambda *_a, **_k:
        (_ for _ in ()).throw(RuntimeError("planner report failed")))
    memory = rp.memory_decision("/node/case.toml", probe={"devices": []})
    assert memory["measured"] is False and memory["refuse"] is False and memory["advisory"] is True
    assert "planner report failed" in memory["advice"]
    assert memory["basis"] == "memory review failed"
    request, launched = _launch(monkeypatch, tmp_path, memory)
    assert rp.launch(request, tmp_path) == {"job": {}}
    assert len(launched) == 1


def test_both_doors_state_a_too_small_card_with_the_same_sentence(monkeypatch, tmp_path):
    """One function, both doors: the staged and node-configuration doors agree."""
    monkeypatch.setattr(rp, "memory_review", lambda *_a, **_k: dict(MEASURED_TOO_SMALL))
    node_side = rp.memory_decision("/node/case.toml", probe=PROBE)
    staged_side = _advised(MEASURED_TOO_SMALL, PROBE)
    assert node_side["advice"] == staged_side["advice"]
    assert node_side["advice"] is not None


def test_the_node_configuration_door_launches_a_measured_too_small_card_with_the_advice_on_its_record(monkeypatch, tmp_path):
    source = _saved_config(tmp_path)
    monkeypatch.setattr(rp, "memory_review", lambda *_a, **_k: dict(MEASURED_TOO_SMALL))
    monkeypatch.setattr(rp, "hardware_probe", lambda **_k: dict(PROBE))
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(tmp_path / "new output"), "products": "none"}
    captured = []
    monkeypatch.setattr(rw, "_launch_review", lambda *args, **kwargs: captured.append(args[2]) or {"job": {}})
    assert rw._launch(request, tmp_path) == {"job": {}}
    memory = captured[0]["memory"]
    assert "9000000000" in memory["advice"] and "8000000000" in memory["advice"]
    # The composed command carries the same policy: go is told not to turn
    # the same estimate into a refusal of its own.
    assert "--no-memory-gate" in captured[0]["argv"]


def test_the_dry_run_door_reports_the_advice(monkeypatch, tmp_path):
    source = _saved_config(tmp_path)
    monkeypatch.setattr(rp, "memory_review", lambda *_a, **_k: dict(MEASURED_TOO_SMALL))
    monkeypatch.setattr(rp, "hardware_probe", lambda **_k: dict(PROBE))
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(tmp_path / "new output"), "products": "none",
               "dry_run": True}
    value = rw._launch(request, tmp_path)
    assert value["dry_run"] is True
    assert "9000000000" in value["review"]["memory"]["advice"]
    assert not (tmp_path / "new output").exists()


def test_the_node_configuration_door_prices_an_unmeasured_card_from_this_node_s_own_probe(monkeypatch, tmp_path):
    """The door a saved configuration comes through reads the probe, not nothing.

    A node whose own probe recorded devices must not be told it records no
    device capacity at all, and the two doors must name one basis.
    """
    source = _saved_config(tmp_path)
    monkeypatch.setattr(rp, "memory_review", lambda *_a, **_k: dict(UNMEASURED))
    probes = []

    def probe(**kwargs):
        probes.append(kwargs)
        return dict(PROBE)

    monkeypatch.setattr(rp, "hardware_probe", probe)
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(tmp_path / "new output"), "products": "none",
               "dry_run": True}
    memory = rw._launch(request, tmp_path)["review"]["memory"]
    assert memory["advice"] is None and memory["warn"] is True
    assert memory["basis"] == PROBE["device_query_basis"]
    assert memory["priced_free_bytes"] == 8_000_000_000
    assert "8000000000" in memory["priced_note"]
    assert "no recorded basis" not in memory["priced_note"]
    assert probes and probes[0]["measure_sizing"] is False, "the card is not measured a second time"
    staged = _advised(UNMEASURED, PROBE)
    assert (memory["basis"], memory["priced_free_bytes"], memory["priced_note"]) == (
        staged["basis"], staged["priced_free_bytes"], staged["priced_note"]), "one basis, both doors"


def test_a_node_configuration_door_whose_probe_cannot_be_read_still_launches(monkeypatch, tmp_path):
    """An unreadable probe is stated by the pricing, never a refusal of its own."""
    source = _saved_config(tmp_path, "unprobeable.toml")
    monkeypatch.setattr(rp, "memory_review", lambda *_a, **_k: dict(UNMEASURED))
    monkeypatch.setattr(rp, "hardware_probe", lambda **_k:
        (_ for _ in ()).throw(OSError("nvidia-smi is not installed")))
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(tmp_path / "new output"), "products": "none",
               "dry_run": True}
    memory = rw._launch(request, tmp_path)["review"]["memory"]
    assert memory["advice"] is None and memory["warn"] is True
    assert memory["basis"] == "no recorded device capacity"
    assert "no recorded basis" in memory["priced_note"]


def test_no_remote_door_carries_a_memory_refusal_any_more():
    """The retired refusal arm has no reader left: a sizing figure advises only."""
    import inspect
    for function in (rp.launch, rw._launch, rw._review, rp.review):
        source = inspect.getsource(function)
        assert "not launch-ready" not in source
        assert '"refusal"' not in source
