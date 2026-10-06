from contextlib import contextmanager
from dataclasses import replace
from threading import Barrier
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
from gpuwm.ensemble.execution import (MemberRunControl, execute_concurrent_members,
    failed_member_rows, member_run_scope, current_run_control)
from gpuwm.ensemble.packing import CardBudget, pack_members


def test_original_concurrency_prices_each_complete_model_and_uses_waves():
    model = EnsembleMemoryModel((MemoryComponent("all original allocations", "ordinary", fixed_bytes=40,
                                               per_member_bytes=60),))
    plan = pack_members(8, (CardBudget(0, 450),), model, batched=False, concurrent_ordinary=True)
    assert plan.capacities == (4,)
    assert [batch.member_indices for batch in plan.batches] == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert all(batch.required_bytes == 400 for batch in plan.batches)
    assert all(batch.execution_mode == "ordinary_concurrent_members" for batch in plan.batches)
    small = pack_members(2, (CardBudget(0, 99),), model, batched=False, concurrent_ordinary=True)
    assert all(batch.execution_mode == "ordinary_streamed_member" for batch in small.batches)


def test_concurrent_members_overlap_keep_context_and_return_roster_order():
    model = EnsembleMemoryModel((MemoryComponent("all", "ordinary", fixed_bytes=100),))
    batch = pack_members(2, (CardBudget(0, 1000),), model, batched=False, concurrent_ordinary=True).batches[0]
    barrier = Barrier(2)
    scopes = []
    @contextmanager
    def scope(**kwargs):
        scopes.append(kwargs)
        yield SimpleNamespace(receipt=lambda: kwargs)
    control = MemberRunControl()
    def member(single):
        assert current_run_control() is control
        assert single.execution_mode == "ordinary_member"
        barrier.wait(timeout=5)
        return {"status": "PASS", "clock": [1, 3, 7], "member": single.member_indices[0]}
    with member_run_scope(control):
        receipt = execute_concurrent_members(batch, member, member_scope=scope)
    assert [row["member_id"] for row in receipt["members"]] == [0, 1]
    assert sorted(row["member_id"] for row in scopes) == [0, 1]
    assert all(row["result"]["clock"] == [1, 3, 7] for row in receipt["members"])


def test_concurrent_failure_names_only_failed_member_and_joins_other():
    model = EnsembleMemoryModel((MemoryComponent("all", "ordinary", fixed_bytes=100),))
    batch = pack_members(2, (CardBudget(0, 1000),), model, batched=False, concurrent_ordinary=True).batches[0]
    batch = replace(batch, member_indices=(17, 29))
    completed = []
    barrier = Barrier(2)
    @contextmanager
    def scope(**kwargs):
        yield SimpleNamespace(receipt=lambda: kwargs)
    def member(single):
        barrier.wait(timeout=5)
        if single.member_indices == (17,):
            raise ValueError("forcing rejected")
        completed.append(29)
        return {"status": "PASS"}
    with pytest.raises(ValueError, match="member 17: forcing rejected") as caught:
        execute_concurrent_members(batch, member, member_scope=scope)
    assert completed == [29]
    assert [row["member_ids"] for row in failed_member_rows(caught.value)] == [[17]]
