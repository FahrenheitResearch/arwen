"""Named surface options validate before the recipe acquires data or CUDA."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.surface_controls import validate_surface_recipe
from gpuwm.ensemble.request import EnsembleRequest


@pytest.mark.parametrize("options", [
    {"soil_moisture_scale": 0.8}, {"sst_offset_k": -1.5},
    {"soil_moisture_scale": [0.8, 1.2], "sst_offset_k": [-1.5, 1.5]},
])
def test_surface_options_are_named_and_bound_to_source_recipe(options):
    selected = {"kind": "surface-state", **options}
    request = EnsembleRequest(4, recipe="time-lagged", perturbation=selected)
    assert request.receipt()["perturbation"] == validate_surface_recipe(selected)
    selected[next(iter(options))] = "changed after admission"
    assert request.receipt()["perturbation"] != selected


@pytest.mark.parametrize("options, words", [
    ({}, "needs soil_moisture_scale"),
    ({"soil_moisture_scale": 0}, "must be positive"),
    ({"soil_moisture_scale": [-1, 1]}, "must be positive"),
    ({"soil_moisture_scale": [1, 0.8]}, "maximum is below"),
    ({"sst_offset_k": [1]}, "needs \\[minimum, maximum\\]"),
    ({"sst_offset_k": True}, "finite FP32"),
    ({"sst_offset_k": "1"}, "finite FP32"),
    ({"sst_offset_k": float("nan")}, "finite FP32"),
    ({"sst_offset_k": 1e40}, "finite FP32"),
    ({"soil_moisture_scale": 1, "sst": 1}, "unknown surface-state options"),
])
def test_invalid_surface_options_refuse_before_numerical_provider(options, words, monkeypatch):
    from gpuwm.ensemble import surface_recipe
    monkeypatch.setattr(surface_recipe, "_module", lambda *args: pytest.fail("GPU entered"))
    with pytest.raises(ValueError, match=words):
        EnsembleRequest(4, perturbation={"kind": "surface-state", **options})


def test_surface_options_are_revalidated_when_an_admitted_request_is_mutated():
    request = EnsembleRequest(2, perturbation={"kind": "surface-state", "sst_offset_k": [-1, 1]})
    request.perturbation["sst_offset_k"][0] = float("nan")
    with pytest.raises(ValueError, match="finite FP32"):
        EnsembleRequest.from_mapping(request)


def test_surface_recipe_reuses_one_source_and_preserves_selected_member_seed():
    from gpuwm.ensemble.recipes import build_recipe
    from gpuwm.ensemble.seeds import member_seed
    cycle = datetime(2026, 8, 20, tzinfo=timezone.utc)
    options = {"kind": "surface-state", "soil_moisture_scale": [0.8, 1.2]}
    plan = build_recipe(source="hrrr", cycle=cycle, start=cycle,
        end=cycle + timedelta(hours=29), count=8, base_seed=73,
        kind="surface-state", perturbation=options)
    assert len(plan.members) == 8
    assert len(plan.acquisitions()) == 1
    assert {member.trajectory.identity for member in plan.members} == {plan.base.identity}
    assert [member.seed for member in plan.members] == [member_seed(73, index) for index in range(8)]
    assert plan.describe()["perturbation"] == validate_surface_recipe(options)
    assert plan.describe()["calibration"] == "not calibrated"
    replay = plan.select_members((6,))
    assert replay.members[0].index == 6 and replay.members[0].seed == member_seed(73, 6)
    assert replay.perturbation == plan.perturbation


@pytest.mark.parametrize("options", [
    None, {"kind": "surface-state", "soil_moisture_scale": 1.2},
    {"kind": "surface-state", "sst_offset_k": [1, 1]},
])
def test_shared_source_surface_recipe_does_not_fabricate_distinct_members(options):
    with pytest.raises(ValueError, match="surface perturbation needs|identical copies"):
        EnsembleRequest(8, recipe="surface-state", perturbation=options)


def test_unperturbed_source_receipt_keeps_its_schema():
    from gpuwm.ensemble.recipes import build_recipe
    cycle = datetime(2026, 8, 20, tzinfo=timezone.utc)
    plan = build_recipe(source="hrrr", cycle=cycle, start=cycle,
        end=cycle + timedelta(hours=1), count=1, base_seed=73)
    assert "perturbation" not in plan.describe()


def test_surface_callback_composes_existing_owner_and_every_started_domain(monkeypatch):
    from gpuwm.ensemble import surface_recipe
    request = SimpleNamespace(perturbation={"kind": "surface-state", "sst_offset_k": [-1, 1]})
    root = SimpleNamespace(cfg=SimpleNamespace(grid_id=1), state=SimpleNamespace(elapsed_seconds=0), _started=True)
    child = SimpleNamespace(cfg=SimpleNamespace(grid_id=2), state=SimpleNamespace(elapsed_seconds=0), _started=True)
    delayed = SimpleNamespace(cfg=SimpleNamespace(grid_id=3), state=SimpleNamespace(elapsed_seconds=0), _started=False)
    model = SimpleNamespace(walk_parent_first=lambda: iter((root, child, delayed)))
    order, draws, receipts = [], [], []
    def realization(*args, **kwargs):
        draws.append(kwargs["seed"])
        return "device words", {"realized_fp32_hex": "same words"}
    def apply(state, **kwargs):
        order.append(kwargs["domain_id"])
        assert kwargs["realization"][0] == "device words"
        return {"domain_id": kwargs["domain_id"], "seed": kwargs["seed"]}
    monkeypatch.setattr(surface_recipe, "realize_surface_recipe", realization)
    monkeypatch.setattr(surface_recipe, "apply_surface_recipe", apply)
    callback = surface_recipe.surface_initialization_callback(request, member_id=4, seed=13,
        previous=lambda **kwargs: order.append("existing owner"), record=receipts.append)
    callback(model=model)
    assert order == ["existing owner", 1, 2]
    assert draws == [13]
    assert receipts[0]["member_id"] == 4 and receipts[0]["seed"] == 13
    assert [row["domain_id"] for row in receipts[0]["domains"]] == [1, 2]


def test_surface_recipe_inactive_callback_does_not_import_or_call_cuda():
    from gpuwm.ensemble.surface_recipe import surface_initialization_callback
    previous = lambda **kwargs: None
    assert surface_initialization_callback(SimpleNamespace(perturbation=None),
        member_id=0, seed=0, previous=previous) is previous


def test_advanced_domain_does_not_get_an_initial_surface_perturbation():
    from gpuwm.ensemble.surface_recipe import apply_surface_recipe
    state = SimpleNamespace(elapsed_seconds=60)
    receipt = apply_surface_recipe(state, value={"kind": "surface-state", "sst_offset_k": 1},
                                   member_id=3, seed=13, domain_id=2)
    assert not receipt["applied"] and receipt["domain_id"] == 2
    assert "retain their advanced surface state" in receipt["reason"]


def test_realized_surface_inventory_reports_duplicate_words_without_claiming_independent_forecasts():
    from gpuwm.ensemble.surface_recipe import surface_realization_inventory
    inventory = surface_realization_inventory([
        {"member_id": 4, "domains": [{"realized_fp32_hex": "first"}, {"realized_fp32_hex": "first"}]},
        {"member_id": 5, "domains": [{"realized_fp32_hex": "second"}]},
        {"member_id": 6, "domains": [{"realized_fp32_hex": "first"}]},
    ])
    assert inventory["member_ids"] == [4, 5, 6]
    assert inventory["realized_surface_arms"] == 2
    assert inventory["duplicate_surface_realizations"] == [[4, 6]]
    assert inventory["calibration"] == "not calibrated"


def test_surface_recipe_round_trip_keeps_seeded_options_in_posted_source_authority():
    from gpuwm.ensemble.recipes import build_recipe
    from gpuwm.ensemble.posted_physical import _recipe_from_document
    cycle = datetime(2026, 8, 20, tzinfo=timezone.utc)
    recipe = build_recipe(source="hrrr", cycle=cycle, start=cycle, end=cycle + timedelta(hours=1),
        count=4, base_seed=73, kind="surface-state",
        perturbation={"kind": "surface-state", "sst_offset_k": [-1, 1]})
    restored = _recipe_from_document(recipe.describe())
    assert restored.describe() == recipe.describe()
    assert restored.sha256 == recipe.sha256


def test_delayed_domain_at_activation_is_fresh_at_its_own_epoch():
    from gpuwm.ensemble.surface_recipe import _advanced
    assert not _advanced(SimpleNamespace(elapsed_seconds=900, domain_start_offset=900))
    assert _advanced(SimpleNamespace(elapsed_seconds=910, domain_start_offset=900))
    assert _advanced(SimpleNamespace(elapsed_seconds=60, domain_start_offset=0))


def test_absent_surface_recipe_keeps_declared_configuration_and_source_bytes_unchanged():
    from copy import deepcopy
    from gpuwm.ensemble.door import request_for_payload
    from gpuwm.ensemble.surface_recipe import surface_initialization_callback
    payload = b'[experiment]\nname = "ordinary"\nrun_seconds = 120.0\nfeedback = 0\n'
    original = bytes(payload)
    assert request_for_payload(payload) is None
    request = EnsembleRequest(1)
    receipt = deepcopy(request.receipt())
    state = SimpleNamespace(elapsed_seconds=0, physics=None)
    before = dict(vars(state))
    callback = surface_initialization_callback(request, member_id=0, seed=0)
    assert callback is None
    assert vars(state) == before
    assert request.receipt() == receipt
    assert payload == original
