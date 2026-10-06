"""Explicit per-card concurrency with unchanged memory and default receipts."""
import json

import numpy as np
import pytest

from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
from gpuwm.ensemble.packing import CardBudget, pack_members
from gpuwm.ensemble.request import EnsembleRequest
from gpuwm.toml_document import emit_experiment_toml


def _ordinary():
    return EnsembleMemoryModel((MemoryComponent("ordinary", "state", fixed_bytes=12 << 30,
        basis="envelope", evidence="whole ordinary forecast inventory"),))


def test_four_large_cards_run_eight_original_members_two_per_card():
    model = _ordinary()
    cards = tuple(CardBudget(device, 96 << 30, 96 << 30) for device in range(4))
    plan = pack_members(8, cards, model, batched=False, concurrent_ordinary=True,
                        max_ordinary_members_per_device=2)
    assert plan.waves == 1 and plan.capacities == (2, 2, 2, 2)
    assert [(batch.device_id, batch.member_indices) for batch in plan.batches] == [
        (0, (0, 1)), (1, (2, 3)), (2, (4, 5)), (3, (6, 7))]
    assert all(batch.execution_mode == "ordinary_concurrent_members" for batch in plan.batches)
    assert all(batch.required_bytes == 2 * model.required_bytes(1) for batch in plan.batches)
    assert plan.receipt()["max_ordinary_members_per_device"] == 2
    assert [card["memory_member_capacity"] for card in plan.receipt()["cards"]] == [8, 8, 8, 8]


def test_eight_cards_with_cap_one_keep_one_original_member_per_card():
    cards = tuple(CardBudget(device, 96 << 30, 96 << 30) for device in range(8))
    plan = pack_members(8, cards, _ordinary(), batched=False, concurrent_ordinary=True,
                        max_ordinary_members_per_device=1)
    assert plan.waves == 1
    assert [(batch.device_id, batch.member_indices) for batch in plan.batches] == [
        (device, (device,)) for device in range(8)]


def test_ten_member_template_cap_schedules_eight_then_two_without_dropping_members():
    cards = tuple(CardBudget(device, 96 << 30) for device in range(4))
    plan = pack_members(10, cards, _ordinary(), batched=False, concurrent_ordinary=True,
                        max_ordinary_members_per_device=2)
    assert plan.waves == 2
    assert sum(batch.members for batch in plan.batches_in_wave(0)) == 8
    assert sum(batch.members for batch in plan.batches_in_wave(1)) == 2
    assert [member for batch in plan.batches for member in batch.member_indices] == list(range(10))


def test_actual_memory_fit_remains_stricter_than_requested_concurrency():
    model = _ordinary()
    cards = (CardBudget(0, model.required_bytes(1)), CardBudget(1, model.required_bytes(1) - 1))
    plan = pack_members(4, cards, model, batched=False, concurrent_ordinary=True,
                        max_ordinary_members_per_device=2)
    assert plan.capacities == (1, 0)
    assert all(batch.members == 1 for batch in plan.batches)
    assert [batch.execution_mode for batch in plan.batches] == [
        "ordinary_member", "ordinary_streamed_member", "ordinary_member", "ordinary_streamed_member"]


@pytest.mark.parametrize("value", [True, False, np.bool_(True), 0, -1, 1.5, "2"])
def test_invalid_card_caps_are_rejected_at_both_request_and_packing(value):
    with pytest.raises((TypeError, ValueError)):
        EnsembleRequest(8, max_ordinary_members_per_device=value)
    with pytest.raises((TypeError, ValueError)):
        pack_members(8, (CardBudget(0, 96 << 30),), _ordinary(),
                     batched=False, concurrent_ordinary=True, max_ordinary_members_per_device=value)


def test_default_none_retains_receipt_and_emitted_config_bytes():
    request = EnsembleRequest.from_mapping({"members": 8, "base_seed": 20261004,
                                           "member_device_ids": [0, 1, 2, 3]})
    explicit_none = EnsembleRequest.from_mapping({**request.receipt(), "max_ordinary_members_per_device": None})
    assert "max_ordinary_members_per_device" not in request.receipt()
    assert json.dumps(request.receipt(), separators=(",", ":")) == json.dumps(explicit_none.receipt(), separators=(",", ":"))
    # The baseline receipt is the pre-cap public schema, including key order.
    baseline = {"members": 8, "keep_member_files": False, "retain_member_diagnostics": False,
        "stochastic": None, "thresholds": {}, "sources": [], "perturbation": None,
        "base_seed": 20261004, "member_device_ids": [0, 1, 2, 3]}
    assert json.dumps(request.receipt(), separators=(",", ":")) == json.dumps(baseline, separators=(",", ":"))
    raw = {"ensemble": {name: value for name, value in request.receipt().items() if value is not None}}
    original = {"ensemble": {name: value for name, value in baseline.items() if value is not None}}
    assert emit_experiment_toml(raw).encode() == emit_experiment_toml(original).encode()
    cards = tuple(CardBudget(device, 96 << 30) for device in range(4))
    default = pack_members(8, cards, _ordinary(), batched=False, concurrent_ordinary=True)
    none = pack_members(8, cards, _ordinary(), batched=False, concurrent_ordinary=True,
                        max_ordinary_members_per_device=None)
    assert default.receipt() == none.receipt()
    assert "max_ordinary_members_per_device" not in default.receipt()
    assert default.batches[0].member_indices == tuple(range(8))


def test_cap_roundtrips_in_request_receipt_and_configuration_only_when_active():
    request = EnsembleRequest(8, max_ordinary_members_per_device=2)
    assert EnsembleRequest.from_mapping(request.receipt()) == request
    assert request.receipt()["max_ordinary_members_per_device"] == 2
    raw = {"ensemble": {name: value for name, value in request.receipt().items() if value is not None}}
    assert "max_ordinary_members_per_device = 2\n" in emit_experiment_toml(raw)


def test_production_session_records_ordinary_policy_without_limiting_native_batches(tmp_path):
    from contextlib import nullcontext
    from gpuwm.ensemble.production import PreparedEnsembleSession
    from test_ensemble_production_execution import COPIES, Collector, inputs
    seen = []
    def native(**kwargs):
        batch = kwargs["batch"]
        seen.append((batch.device_id, batch.member_indices))
        assert kwargs["request"].max_ordinary_members_per_device == 2
        return {"status": "PASS", "wrfout_count": 0}
    session = PreparedEnsembleSession({"members": 8, "max_ordinary_members_per_device": 2},
        output_directory=tmp_path, identical_members=COPIES, collector=Collector(),
        cards=tuple(CardBudget(device, 96 << 30) for device in range(4)),
        memory_model=_ordinary(), device_scope=lambda device: nullcontext(), native_executor=native)
    receipt = session.run_prepared(lambda *args, **kwargs: pytest.fail("ordinary path selected"), inputs())
    assert seen == [(0, tuple(range(8)))]
    assert receipt["packing"]["max_ordinary_members_per_device"] == 2
    assert receipt["request"]["max_ordinary_members_per_device"] == 2
    assert receipt["members_completed"] == list(range(8))


def test_ordinary_device_cap_does_not_change_qualified_native_packing():
    cards = tuple(CardBudget(device, 96 << 30) for device in range(4))
    original = pack_members(8, cards, _ordinary(), batched=True)
    capped = pack_members(8, cards, _ordinary(), batched=True,
                          max_ordinary_members_per_device=1)
    assert capped.capacities == original.capacities
    assert capped.batches == original.batches
    assert capped.batches[0].execution_mode == "member_batched"
