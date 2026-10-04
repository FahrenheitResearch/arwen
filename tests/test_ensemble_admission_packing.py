"""CPU admission contracts: allocator rounding and complete member schedules."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ensemble.admission import (
    AllocatorMargin, EnsembleMemoryModel, MemoryCalibration, MemoryComponent,
    ordinary_memory_model_from_estimate,
)
from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
from gpuwm.ensemble.packing import (
    CardBudget, MemberBatch, MemberPackingPlan, card_budgets_from_readings, pack_members,
)


def small_model():
    state = BatchMemoryPlan((BatchArraySpec("terrain", (100,), "shared"),
                             BatchArraySpec("theta", (100,), "member")), reserved_bytes=0)
    return EnsembleMemoryModel.from_plans({"state": ("state", state)},
        reservations=(MemoryComponent("context", "context", fixed_bytes=100,
                                      basis="measured", evidence="context-receipt"),),
        inventory_id="small-v1")


def test_contiguous_and_independent_rounding_match_allocator_blocks():
    plan = BatchMemoryPlan((BatchArraySpec("private", (129,), "member"),
                            BatchArraySpec("shared", (1,), "shared")), reserved_bytes=19)
    contiguous = MemoryComponent("state", "state", plan)
    independent = replace(contiguous, independent_members=True)
    assert contiguous.inventory(3)["required_bytes"] == 2048 + 512 + 19
    assert independent.inventory(3)["required_bytes"] == 3 * 1024 + 512 + 3 * 19
    arrays = contiguous.inventory(3)["arrays"]
    assert arrays[0]["payload_bytes"] == np.empty((3, 129), np.float32).nbytes
    assert arrays[1]["payload_bytes"] == np.empty((1,), np.float32).nbytes


def test_margin_is_a_named_reservation_with_exact_integer_ceiling():
    model = replace(small_model(), allocator_margin=AllocatorMargin.from_fraction(
        "0.13", categories=("state",), minimum_bytes=1, evidence="pool-peak"))
    receipt = model.inventory(1)
    assert receipt["allocator_margin_bytes"] == 134
    assert receipt["category_bytes"] == {"state": 1024, "context": 100, "allocator_margin": 134}
    assert receipt["required_bytes"] == 1258
    assert receipt["allocator_margin_evidence"] == "pool-peak"


def test_member_dependent_products_price_paintball_word_boundary():
    def products(members):
        return BatchMemoryPlan((BatchArraySpec("paintball", ((members + 63) // 64, 65), "shared", "uint64"),), 0)
    model = EnsembleMemoryModel((MemoryComponent("products", "products", plan_for_members=products),))
    assert model.required_bytes(64) == 1024
    assert model.required_bytes(65) == 1536
    assert model.largest_that_fits(1024, max_members=100) == 64


@pytest.mark.parametrize("available", (0, 1123, 1124, 1636, 10000))
def test_capacity_is_largest_allocation_that_fits(available):
    model = small_model()
    fits = [members for members in range(1, 31) if model.required_bytes(members) <= available]
    assert model.largest_that_fits(available, max_members=30) == max(fits, default=0)


def test_multi_card_packing_then_waves_preserves_all_global_member_ids():
    model = small_model()
    cards = (CardBudget(3, model.required_bytes(4), 32 << 30),
             CardBudget(7, model.required_bytes(2), 96 << 30))
    plan = pack_members(13, cards, model)
    assert plan.capacities == (5, 2)  # allocation rounding admits a fifth slab
    assert [batch.member_indices for batch in plan.batches] == [tuple(range(5)), (5, 6), tuple(range(7, 12)), (12,)]
    assert plan.waves == 2
    assert plan.batches[-1].execution_mode == "ordinary_member"
    assert plan.receipt()["physics_changed"] is False
    assert plan.receipt()["halo_exchange"] is False


def test_insufficient_resident_card_uses_streamed_member_without_dropping_request():
    model = small_model()
    plan = pack_members(4, (CardBudget(0, 17),), model)
    assert plan.capacities == (0,)
    assert plan.waves == 4
    assert all(batch.execution_mode == "ordinary_streamed_member" for batch in plan.batches)
    assert all(batch.required_bytes is None for batch in plan.batches)
    assert tuple(batch.member_indices[0] for batch in plan.batches) == (0, 1, 2, 3)
    assert "without changing its physics" in plan.batches[0].reason


def test_resident_card_receives_members_before_unnecessary_tile_fallback():
    model = small_model()
    cards = (CardBudget(0, 17), CardBudget(1, model.required_bytes(4)))
    plan = pack_members(4, cards, model)
    assert plan.cards == cards and plan.capacities[0] == 0
    assert len(plan.batches) == 1
    assert plan.batches[0].device_id == 1
    assert plan.batches[0].member_indices == (0, 1, 2, 3)
    assert plan.batches[0].execution_mode == "member_batched"


def test_ordinary_path_reuses_one_member_inventory_and_its_independent_clock():
    model = small_model()
    plan = pack_members(5, (CardBudget(0, 100000), CardBudget(1, 100000)), model,
                        batched=False, reason="independent adaptive clocks")
    assert plan.capacities == (1, 1)
    assert plan.waves == 3
    assert all(batch.members == 1 and batch.execution_mode == "ordinary_member" for batch in plan.batches)
    assert all(batch.reason == "independent adaptive clocks" for batch in plan.batches)


def test_existing_device_ids_resolve_to_distinct_physical_cards_without_cuda():
    options = SimpleNamespace(device_ids=lambda: (4, 2, 4))
    cards = card_budgets_from_readings(options, {
        4: {"available_bytes": 123, "total_bytes": 456}, 2: CardBudget(2, 789, 999)})
    assert tuple(card.device_id for card in cards) == (4, 2)
    with pytest.raises(ValueError, match="no sampled memory"):
        card_budgets_from_readings(options, {4: {"available_bytes": 123}})


def test_card_bound_models_price_different_context_and_workspace_reservations():
    first = small_model()
    second = replace(first, components=(first.components[0],
        replace(first.components[1], fixed_bytes=600)), inventory_id="larger-context-v1")
    available = first.required_bytes(3)
    cards = (CardBudget(0, available), CardBudget(2, available))
    plan = pack_members(8, cards, {0: first, 2: second})
    assert plan.capacities == (3, 2)
    assert [batch.member_indices for batch in plan.batches] == [(0, 1, 2), (3, 4), (5, 6, 7)]
    for batch in plan.batches:
        assert batch.required_bytes == {0: first, 2: second}[batch.device_id].required_bytes(batch.members)
    assert [card["inventory_id"] for card in plan.receipt()["cards"]] == ["small-v1", "larger-context-v1"]
    assert "inventory_id" not in pack_members(1, cards, first).receipt()["cards"][0]


@pytest.mark.parametrize("models", ({}, {0: small_model()}, {0: small_model(), 1: small_model(), 3: small_model()}))
def test_card_bound_model_mapping_cannot_omit_or_invent_a_card(models):
    with pytest.raises(ValueError, match="every sampled card exactly"):
        pack_members(2, (CardBudget(0, 10000), CardBudget(1, 10000)), models)


def test_card_bound_streamed_model_records_the_correct_card_envelope():
    cards = (CardBudget(0, 1), CardBudget(1, 1))
    models = {0: small_model(), 1: replace(small_model(), inventory_id="second")}
    streamed = {0: EnsembleMemoryModel((MemoryComponent("tile", "state", fixed_bytes=5),), inventory_id="tile-0"),
                1: EnsembleMemoryModel((MemoryComponent("tile", "state", fixed_bytes=7),), inventory_id="tile-1")}
    plan = pack_members(2, cards, models, streamed_model=streamed)
    assert [batch.required_bytes for batch in plan.batches] == [5, 7]
    assert all(batch.execution_mode == "ordinary_streamed_member" for batch in plan.batches)


def test_preflight_envelope_carries_whole_ordinary_inventory_once():
    class Domain:
        items = tuple(SimpleNamespace(category=category) for category in
                      ("state", "scratch", "lbc", "nest", "physics", "diagnostic"))
        def category_bytes(self, category):
            return {"state": 100, "scratch": 30, "lbc": 20, "nest": 10,
                    "physics": 60, "diagnostic": 4}.get(category, 0)
    estimate = SimpleNamespace(domains=(Domain(), Domain()), dycore_state_saved_bytes=20,
        scratch_arena_saved_bytes=10, k_tables_bytes=40, workspace_bytes=50,
        transient_peak_bytes=70, subtotal_bytes=578, peak_envelope_bytes=800,
        envelope_basis="ordinary measured envelope")
    model = ordinary_memory_model_from_estimate(estimate)
    assert model.required_bytes(1) == model.required_bytes(99) == 800
    assert model.inventory(1)["category_bytes"]["state"] == 180
    assert all(row["basis"] == "envelope" for row in model.inventory(1)["components"])
    estimate.subtotal_bytes += 4
    with pytest.raises(ValueError, match="unclassified"):
        ordinary_memory_model_from_estimate(estimate)


def test_future_preflight_memory_category_is_inherited_without_suite_specific_code():
    from gpuwm.core.preflight import DomainMemoryEstimate, MemoryItem
    domain = DomainMemoryEstimate(1, (MemoryItem("field", "new_physics_bank", (10,), 4),))
    estimate = SimpleNamespace(domains=(domain,), dycore_state_saved_bytes=0,
        scratch_arena_saved_bytes=0, k_tables_bytes=0, workspace_bytes=0,
        transient_peak_bytes=0, subtotal_bytes=40, peak_envelope_bytes=60,
        envelope_basis="ordinary envelope")
    model = ordinary_memory_model_from_estimate(estimate)
    assert model.inventory(1)["category_bytes"]["new_physics_bank"] == 40


def test_calibration_requires_matching_inventory_and_separates_missing_components():
    model = small_model()
    calibration = MemoryCalibration("small-v1", 1, "sm120-nvrtc13", 1024, 1100, 1200, 1500, "peak.json")
    receipt = calibration.receipt(model)
    assert receipt["unpriced_live_bytes"] == 76
    assert receipt["allocator_retention_bytes"] == 100
    assert receipt["non_pool_bytes"] == 300
    assert receipt["peak_within_plan"] is False
    with pytest.raises(ValueError, match="inventory differs"):
        calibration.receipt(replace(model, inventory_id="different"))
    with pytest.raises(ValueError, match="array bytes differ"):
        replace(calibration, planned_array_bytes=1).receipt(model)


@pytest.mark.parametrize("value", (True, np.bool_(True), -1, 1.5))
def test_invalid_count_cannot_be_silently_coerced(value):
    with pytest.raises((ValueError, TypeError)):
        pack_members(value, (CardBudget(0, 10000),), small_model())


def test_duplicate_cards_or_member_ids_cannot_overbook_a_wave():
    with pytest.raises(ValueError, match="physical card once"):
        pack_members(2, (CardBudget(0, 10000), CardBudget(0, 10000)), small_model())
    batch = MemberBatch(0, 0, (0,), "ordinary_member", 1, 10)
    with pytest.raises(ValueError, match="every requested member exactly once"):
        MemberPackingPlan(2, (CardBudget(0, 10),), (1,), (batch, batch))


@pytest.mark.parametrize("mp", (0, 6, 8, 9, 10, 16, 18, 28, 50))
def test_real_preflight_categories_are_retained_for_each_microphysics_inventory(mp):
    from datetime import datetime
    from gpuwm.config import RunConfig
    from gpuwm.core.preflight import estimate_experiment
    from gpuwm.experiment import experiment_from_run_config
    cfg = RunConfig(nx=24, ny=20, nz=12, dx=3000., dy=3000., ztop=12000.,
                    dt=6., run_seconds=60., moist=True, mp_physics=mp)
    estimate = estimate_experiment(experiment_from_run_config(cfg, datetime(2024, 1, 1)))
    model = ordinary_memory_model_from_estimate(estimate)
    assert model.required_bytes(1) == estimate.peak_envelope_bytes
    assert model.inventory(1)["category_bytes"]["physics"] == estimate.domains[0].category_bytes("physics")
