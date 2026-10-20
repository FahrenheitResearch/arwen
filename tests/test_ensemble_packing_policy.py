from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
from gpuwm.ensemble.packing import CardBudget, pack_members
from gpuwm.ensemble.packing_policy import DEFAULT_ORDINARY_MEMBER_CAP, ordinary_member_policy
from gpuwm.ensemble.request import EnsembleRequest


def test_bare_ordinary_request_uses_the_measured_cap_without_rewriting_authority():
    request = EnsembleRequest(9)
    original = request.receipt()
    cap, policy = ordinary_member_policy(request, concurrent=True)
    assert cap == DEFAULT_ORDINARY_MEMBER_CAP and cap > 0
    assert policy["source"] == "measured_default"
    assert request.receipt() == original
    assert "max_ordinary_members_per_device" not in original


def test_explicit_caps_win_for_experiments_and_sequential_routes():
    for value in (1, 2, 4, 6):
        request = EnsembleRequest(9, max_ordinary_members_per_device=value)
        for concurrent in (False, True):
            cap, policy = ordinary_member_policy(request, concurrent=concurrent)
            assert cap == value and policy["source"] == "explicit_request"


def test_native_checkpoint_and_single_member_routes_keep_their_existing_policy():
    for request, concurrent in ((EnsembleRequest(9), False), (EnsembleRequest(1), True)):
        cap, policy = ordinary_member_policy(request, concurrent=concurrent)
        assert cap is None and policy["source"] == "existing_memory_admission"


def test_live_memory_fit_remains_stricter_and_no_member_is_dropped():
    request = EnsembleRequest(9)
    cap, _ = ordinary_member_policy(request, concurrent=True)
    model = EnsembleMemoryModel((MemoryComponent("model", "state", fixed_bytes=10,
        basis="envelope", evidence="complete symbolic ordinary model"),))
    plan = pack_members(9, (CardBudget(0, 10), CardBudget(1, 1000)), model,
                        batched=False, concurrent_ordinary=True,
                        max_ordinary_members_per_device=cap)
    assert plan.capacities == (1, cap)
    assert [member for batch in plan.batches for member in batch.member_indices] == list(range(9))
    assert all(batch.required_bytes <= batch.available_bytes for batch in plan.batches)
    assert plan.receipt()["max_ordinary_members_per_device"] == cap


def test_wofs_uses_live_memory_capacity_and_retains_explicit_cap():
    request = EnsembleRequest(36, recipe="wofs", member_variants=[{"name": f"member-{i}"} for i in range(36)])
    cap, policy = ordinary_member_policy(request, concurrent=True)
    assert cap is None and policy["source"] == "live_memory_admission"
    model = EnsembleMemoryModel((MemoryComponent("model", "state", fixed_bytes=6),))
    plan = pack_members(36, (CardBudget(0, 84),), model, batched=False,
                        concurrent_ordinary=True, max_ordinary_members_per_device=cap)
    assert plan.capacities == (14,)
    assert [batch.members for batch in plan.batches] == [14, 14, 8]
    assert ordinary_member_policy(EnsembleRequest(36, recipe="wofs",
        member_variants=[{"name": f"member-{i}"} for i in range(36)], max_ordinary_members_per_device=3), concurrent=True)[0] == 3
