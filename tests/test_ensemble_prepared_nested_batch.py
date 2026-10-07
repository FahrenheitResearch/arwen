"""Genuine executor closure capture, state ownership and hybrid accounting."""
from dataclasses import replace
from types import SimpleNamespace
from contextlib import nullcontext
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from gpuwm.core.clock import execute_schedule
from gpuwm.core.model import execute_experiment
from gpuwm.core.state import DomainState
from gpuwm.ensemble.batch_state import BatchStateUnsupported, BatchedDomainState
from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
from gpuwm.ensemble.prepared_nested_batch import (
    PreparedHybridNestedBatch, ScheduledDomainBanks, _domain_bank_plan,
    bootstrap_prepared_tree_members, capture_original_member_schedule,
    _require_identity_transition,
    ProductionNestedPackFactory,
)
from test_model import _model
from test_ensemble_batch_state import config


def _actual_capture(member, *, dispatch=True):
    exp, model = _model()
    for node in model.walk_parent_first():
        node.state.words = np.array([member + 37, member + 41], np.uint32)
    trace = []
    def step(state, cfg, **unused):
        state.words ^= np.uint32(cfg.grid_id + 5)
        trace.append(("step", cfg.grid_id, state.elapsed_seconds))
    options = {"validate_state": False, "steppers": {1: step, 2: step},
               "history_handler": lambda model, node, ticks: trace.append(("history", node.cfg.grid_id, ticks)),
               "pool_trim_per_period": False, "experiment": exp}
    if dispatch:
        binding = capture_original_member_schedule(member, model,
            operation_authority=lambda **unused: "original-model-fixture-layout", execution_options=options)
        assert all(node.clock.step_count == 0 for node in model.walk_parent_first())
        return model, trace, binding
    report = execute_experiment(model, **options)
    return model, trace, report


def _state(model):
    return [(node.state.words.tobytes(), node.state.elapsed_seconds, node.clock.ticks,
             node.clock.step_count, node.clock.dt_fp32.tobytes(), node.clock.dtbc_fp32.tobytes())
            for node in model.walk_parent_first()]


def test_actual_execute_experiment_closures_match_original_step_history_and_clock_words():
    references = [_actual_capture(member, dispatch=False) for member in range(2)]
    captures = [_actual_capture(member) for member in range(2)]
    batch = PreparedHybridNestedBatch([row[2] for row in captures],
        ordinary_forecast_bytes={0: 4096, 1: 8192}, available_bytes=20000,
        ordinary_memory_evidence="test explicit ordinary envelopes", array_module=np,
        completion_wait=lambda: None)
    receipt = batch.execute()
    for (original, trace, report), (model, captured_trace, _) in zip(references, captures, strict=True):
        assert _state(model) == _state(original)
        assert captured_trace == trace
    assert receipt["memory"]["ordinary_forecasts_retained_bytes"] == 12288
    assert receipt["memory"]["bank_bytes"] == 0
    assert receipt["complete_native_forecast"] is False
    assert not any(row["packed"] for row in receipt["operations"])


def test_schedule_dispatch_off_uses_original_executor_directly(monkeypatch):
    import gpuwm.core.clock as clocks
    calls = []
    original = clocks.execute_schedule
    def actual(schedule, **kwargs):
        calls.append((schedule, kwargs["on_period_steps"], kwargs["on_step"]))
        return original(schedule, **kwargs)
    monkeypatch.setattr(clocks, "execute_schedule", actual)
    _actual_capture(0, dispatch=False)
    assert len(calls) == 1 and calls[0][1] is None
    assert calls[0][2].__qualname__.endswith("execute_experiment.<locals>.on_step")


def _bank_nodes(*, adaptive=False):
    cfg = config(use_adaptive_time_step=adaptive)
    states = [DomainState(cfg, array_module=np) for _ in range(2)]
    for member, state in enumerate(states):
        state.u.fill(member + 1)
    return tuple(SimpleNamespace(cfg=SimpleNamespace(run=cfg), state=state) for state in states)


def test_adaptive_resident_banks_preserve_configs_and_retained_original_words():
    nodes = _bank_nodes(adaptive=True)
    configs = [node.cfg.run for node in nodes]
    original = [node.state.u.tobytes() for node in nodes]
    plan = _domain_bank_plan(nodes, shared_fields=("ht",), array_module=np)
    bank = ScheduledDomainBanks(nodes, plan, array_module=np, available_bytes=plan.required_bytes(2))
    assert isinstance(bank, BatchedDomainState) and bank.cfg.use_adaptive_time_step
    assert [node.cfg.run for node in nodes] == configs
    assert bank.clock is None
    bank.u[0].fill(10)
    assert [node.state.u.tobytes() for node in nodes] == original
    bank.copy_mutations_to_originals(("u",))
    assert np.all(nodes[0].state.u == 10) and np.all(nodes[1].state.u == 2)
    assert bank.copy_out_bytes == sum(node.state.u.nbytes for node in nodes)
    assert not np.shares_memory(bank.u, nodes[0].state.u)


@pytest.mark.parametrize("chem_sets", ["smoke", "gocart_primary,smoke", "cams_aq"])
@pytest.mark.parametrize("field_names", [None, ("u",)])
def test_active_chemistry_cannot_enter_full_or_edge_only_component_banks(
        monkeypatch, chem_sets, field_names):
    from gpuwm.ensemble import prepared_nested_batch
    nodes = _bank_nodes()
    nodes[1].cfg.run = replace(nodes[1].cfg.run, chem_sets=chem_sets)
    configs = [vars(node.cfg.run).copy() for node in nodes]
    original = [node.state.u.tobytes() for node in nodes]
    monkeypatch.setattr(prepared_nested_batch, "state_array_specs",
                        lambda *args, **kwargs: pytest.fail("active chemistry reached bank inventory"))
    with pytest.raises(BatchStateUnsupported, match="species arrays, source-hour caches and mass ledger"):
        _domain_bank_plan(nodes, shared_fields=(), array_module=np, field_names=field_names)
    assert [vars(node.cfg.run) for node in nodes] == configs
    assert [node.state.u.tobytes() for node in nodes] == original


@pytest.mark.parametrize("chem_sets", ["smoke", "gocart_primary,smoke", "cams_aq"])
def test_active_chemistry_component_fallback_executes_original_callbacks_and_owners(chem_sets):
    references = [_actual_capture(member, dispatch=False) for member in range(2)]
    captures = [_actual_capture(member) for member in range(2)]
    owners = []
    for model, _, _ in captures:
        for node in model.walk_parent_first():
            node.cfg = replace(node.cfg, run=replace(node.cfg.run, chem_sets=chem_sets))
            species = np.array([17, 29], dtype=np.uint32)
            source_hours, ledger = {"hour": object()}, object()
            node.state.chem = species
            node.state._chem_source_hour_cache = source_hours
            node.state._chem_mass_ledger = ledger
            owners.append((node.state, species, species.tobytes(), source_hours, ledger))
    class ForbiddenFactory:
        def memory_plans(self, *args):
            pytest.fail("active chemistry reached packed component pricing")
        def prepare(self, *args, **kwargs):
            pytest.fail("active chemistry constructed packed component state")
    batch = PreparedHybridNestedBatch([row[2] for row in captures],
        ordinary_forecast_bytes={0: 4096, 1: 8192}, available_bytes=20000,
        ordinary_memory_evidence="complete original CPU callback fixture envelopes",
        pack_factory=ForbiddenFactory(), array_module=np, completion_wait=lambda: None)
    assert not batch.banks and not batch.plans and batch.packed is None
    result = batch.execute()
    assert all(not row["packed"] for row in result["operations"])
    assert any("active chemistry" in reason for reason in result["fallback_reasons"])
    for (original, trace, _), (model, actual_trace, _) in zip(references, captures, strict=True):
        assert _state(model) == _state(original) and actual_trace == trace
    for state, species, words, source_hours, ledger in owners:
        assert state.chem is species and species.tobytes() == words
        assert state._chem_source_hour_cache is source_hours and state._chem_mass_ledger is ledger


@pytest.mark.parametrize("name,moved", [("v_sca_adv_order", 5), ("v_mom_adv_order", 5),
                                       ("h_mom_adv_order", 3)])
def test_advection_order_changes_refuse_before_live_bank_copy(name, moved):
    nodes = _bank_nodes()
    plan = _domain_bank_plan(nodes, shared_fields=(), array_module=np)
    bank = ScheduledDomainBanks(nodes, plan, array_module=np, available_bytes=plan.required_bytes(2))
    before = bank.u.tobytes()
    nodes[1].cfg.run = replace(nodes[1].cfg.run, **{name: moved})
    nodes[0].state.u.fill(99)
    with pytest.raises(BatchStateUnsupported, match="different domain configuration"):
        _domain_bank_plan(nodes, shared_fields=(), array_module=np)
    with pytest.raises(BatchStateUnsupported, match="diagnostic configuration"):
        bank.synchronize_from_originals()
    assert bank.u.tobytes() == before


def test_shared_word_identity_keeps_signed_zero_and_nan_payloads_distinct():
    nodes = _bank_nodes()
    nodes[0].state.ht.view(np.uint32).flat[0] = 0x80000000
    with pytest.raises(BatchStateUnsupported, match="different member words"):
        _domain_bank_plan(nodes, shared_fields=("ht",), array_module=np)
    nodes[1].state.ht.view(np.uint32).flat[0] = 0x80000000
    nodes[0].state.ht.view(np.uint32).flat[1] = 0x7fc00123
    nodes[1].state.ht.view(np.uint32).flat[1] = 0x7fc00456
    with pytest.raises(BatchStateUnsupported, match="different member words"):
        _domain_bank_plan(nodes, shared_fields=("ht",), array_module=np)


def test_live_bank_scalar_mismatch_refuses_before_copying_member_words():
    nodes = _bank_nodes()
    plan = _domain_bank_plan(nodes, shared_fields=(), array_module=np)
    bank = ScheduledDomainBanks(nodes, plan, array_module=np, available_bytes=plan.required_bytes(2))
    before = bank.u.tobytes()
    nodes[1].state.elapsed_seconds = 1.
    nodes[0].state.u.fill(99)
    with pytest.raises(BatchStateUnsupported, match="scalar words differ"):
        bank.synchronize_from_originals()
    assert bank.u.tobytes() == before


@pytest.mark.parametrize("change", ["broadcast", "dtype", "strided", "streamed"])
def test_live_endpoint_rebinding_refuses_before_any_bank_copy(change):
    nodes = _bank_nodes()
    plan = _domain_bank_plan(nodes, shared_fields=(), array_module=np)
    bank = ScheduledDomainBanks(nodes, plan, array_module=np, available_bytes=plan.required_bytes(2))
    before = bank.u.tobytes()
    if change == "broadcast":
        nodes[1].state.u = np.ones((1, 1, 1), np.float32)
    elif change == "dtype":
        nodes[1].state.u = nodes[1].state.u.astype(np.float64)
    elif change == "strided":
        nodes[1].state.u = nodes[1].state.u[..., ::-1]
    else:
        nodes[1].state._streamed_domain = object()
    with pytest.raises(BatchStateUnsupported, match="resident|streamed"):
        bank.synchronize_from_originals()
    assert bank.u.tobytes() == before


def test_hybrid_budget_prices_both_complete_ordinary_forecasts_before_allocation():
    captures = [_actual_capture(member) for member in range(2)]
    with pytest.raises(MemoryError, match="retain 12288 ordinary forecast bytes"):
        PreparedHybridNestedBatch([row[2] for row in captures],
            ordinary_forecast_bytes={0: 4096, 1: 8192}, available_bytes=12287,
            ordinary_memory_evidence="full forecast envelope", array_module=np, completion_wait=lambda: None)
    with pytest.raises(ValueError, match="every retained member"):
        PreparedHybridNestedBatch([row[2] for row in captures],
            ordinary_forecast_bytes={0: 4096}, available_bytes=12288,
            ordinary_memory_evidence="full forecast envelope", array_module=np, completion_wait=lambda: None)


def test_ordinary_prepared_bootstrap_reuses_input_handle_but_retains_distinct_trees():
    shared_inputs, outputs = object(), []
    def runner(inputs, *, output_directory, ensemble_bootstrap):
        _, model = _model()
        outputs.append(output_directory)
        return ensemble_bootstrap(inputs=inputs, model=model, node=model.root)
    members = bootstrap_prepared_tree_members((3, 7), inputs_for=lambda member: shared_inputs,
        runner=runner, output_for=lambda member: f"member-{member}")
    assert all(member.inputs is shared_inputs for member in members)
    assert members[0].model is not members[1].model
    assert outputs == ["member-3", "member-7"]
    assert all(member.receipt["forecast_steps"] == 0 and len(member.receipt["domain_ids"]) == 2 for member in members)


def test_packed_bank_refusal_keeps_the_complete_ordinary_member_walk():
    captures = [_actual_capture(member) for member in range(2)]
    class UnsupportedFactory:
        def memory_plans(self, bindings, descriptors):
            raise AssertionError("field inventory must refuse before a factory can allocate")
    batch = PreparedHybridNestedBatch([row[2] for row in captures],
        ordinary_forecast_bytes={0: 4096, 1: 8192}, available_bytes=12288,
        ordinary_memory_evidence="complete forecasts", array_module=np,
        pack_factory=UnsupportedFactory(), completion_wait=lambda: None)
    receipt = batch.execute()
    assert receipt["fallback_reasons"]
    assert receipt["memory"]["bank_bytes"] == 0
    assert all(model.root.clock.step_count > 0 for model, _, _ in captures)


def test_factory_bank_and_edge_plans_are_charged_together_before_any_allocation():
    from gpuwm.ensemble.batch_storage import BatchStorage
    from gpuwm.core.device_inventory import state_array_shapes
    captures = [_actual_capture(member) for member in range(2)]
    for model, _, _ in captures:
        for node in model.walk_parent_first():
            node.cfg = replace(node.cfg, run=replace(node.cfg.run, nx=4, ny=3, nz=2))
            words = node.state.words
            node.state = DomainState(node.cfg.run, array_module=np)
            node.state.words = words
    class DeclaredFactory:
        calls = []
        callbacks = {}
        plan = BatchMemoryPlan((BatchArraySpec("edge-proof", (257,), "member"),), reserved_bytes=3072)
        def memory_plans(self, bindings, descriptors):
            self.calls.append("plans")
            return {"edge": self.plan}
        def prepare(self, bindings, banks, *, available_bytes, array_module):
            self.calls.append("allocation")
            self.storage = BatchStorage(self.plan, len(bindings), array_module=array_module, available_bytes=available_bytes)
            self.banks = banks
            return self
        def admit(self, group):
            raise AssertionError("no numerical callback was requested")
        def receipt(self):
            return {"test_declared_plan": True}
    bindings = [row[2] for row in captures]
    reserve = {0: 50000, 1: 50000}
    bank_bytes = sum(_domain_bank_plan([binding.model.node(gid) for binding in bindings],
        shared_fields=(), array_module=np).required_bytes(2) for gid in (1, 2))
    factory = DeclaredFactory()
    total = sum(reserve.values()) + bank_bytes + factory.plan.required_bytes(2)
    with pytest.raises(MemoryError):
        PreparedHybridNestedBatch(bindings, ordinary_forecast_bytes=reserve,
            available_bytes=total - 1, ordinary_memory_evidence="test complete envelopes",
            pack_factory=factory, shared_fields=(), array_module=np, completion_wait=lambda: None)
    assert factory.calls == ["plans"]
    factory.calls.clear()
    batch = PreparedHybridNestedBatch(bindings, ordinary_forecast_bytes=reserve,
        available_bytes=total, ordinary_memory_evidence="test complete envelopes",
        pack_factory=factory, shared_fields=(), array_module=np, completion_wait=lambda: None)
    assert factory.calls == ["plans", "allocation"]
    assert batch.required_bytes == total
    assert sum(bank.plan.required_bytes(2) for bank in factory.banks.values()) == bank_bytes


def test_stock_same_scheme_transition_object_is_admitted_without_transforming_fields():
    from gpuwm.core.microphysics_transition import resolve_microphysics_transition, resolve_reverse_microphysics_transition
    from gpuwm.core.preflight import nest_field_kinds
    cfg = config(moist=True, moist_cq=True, mp_physics=8)
    forward = resolve_microphysics_transition(cfg, cfg)
    reverse = resolve_reverse_microphysics_transition(cfg, cfg)
    assert forward is not None and reverse is not None
    assert not forward.mixed and not reverse.mixed
    _require_identity_transition(forward, cfg, cfg, nest_field_kinds(cfg))
    _require_identity_transition(reverse, cfg, cfg, nest_field_kinds(cfg))


def test_mixed_scheme_transition_keeps_its_ordinary_diagnosed_species_refusal():
    from gpuwm.core.microphysics_transition import resolve_microphysics_transition, resolve_reverse_microphysics_transition
    from gpuwm.core.preflight import nest_field_kinds
    parent = config(moist=True, moist_cq=True, mp_physics=8)
    child = config(moist=True, moist_cq=True, mp_physics=18)
    for contract, source, target in ((resolve_microphysics_transition(parent, child), parent, child),
            (resolve_reverse_microphysics_transition(parent, child), child, parent)):
        assert contract.mixed
        with pytest.raises(BatchStateUnsupported, match="original diagnosed-species"):
            _require_identity_transition(contract, source, target, nest_field_kinds(target))


def _step_lifecycle_fixture(phase):
    """Use the genuine entry rendezvous and private callback contexts."""
    from gpuwm.ensemble.scheduled_production_physics import ScheduledProductionPhysics
    from test_ensemble_scheduled_production_physics import _Owner
    owner = object.__new__(ScheduledProductionPhysics)
    owner.owner = _Owner(members=3)
    owner.owner.receipt = {"test_original_owners": True}
    owner.plan = BatchMemoryPlan((BatchArraySpec("coordination-fixture", (1,), "member", "uint8"),), reserved_bytes=0)
    owner._initialize_coordination()
    error = RuntimeError(f"injected {phase} failure")
    entered, cancelled_queued, trace = threading.Event(), threading.Event(), []
    original_wait = owner._condition.wait
    def observe_wait(timeout=None):
        entered.set()
        return original_wait(timeout)
    owner._condition.wait = observe_wait

    class Stream:
        def __init__(self, slot=None):
            self.slot = slot
            self.synchronizations = 0
        def __enter__(self):
            return self
        def __exit__(self, *unused):
            pass
        def wait_event(self, unused):
            pass
        def synchronize(self):
            if phase == "queued known then fail" and self.slot == 0:
                assert cancelled_queued.wait(timeout=2), "known queued Future was not cancelled by shutdown"
            self.synchronizations += 1
    streams = tuple(Stream(slot) for slot in range(3))
    central = Stream()
    class Event:
        def __init__(self):
            if phase == "event allocation":
                raise error
        def record(self, unused):
            if phase == "event record":
                raise error
    actual_pool = ThreadPoolExecutor(max_workers=1 if phase in {"enqueue then fail", "queued known then fail"} else 3)
    class Pool:
        futures, submissions = [], []
        def submit(self, work, slot, request):
            self.submissions.append(slot)
            if (phase == "first submit" and slot == 0
                    or phase in {"partial submit", "enqueue then fail"} and slot == 1
                    or phase == "queued known then fail" and slot == 2):
                if slot:
                    assert entered.wait(timeout=2), "first member never reached the actual entry wait"
                if phase == "enqueue then fail":
                    self.futures.append(actual_pool.submit(work, slot, request))
                raise error
            if phase == "queued known then fail" and slot == 1:
                assert entered.wait(timeout=2), "first member never reached the actual entry wait"
            future = actual_pool.submit(work, slot, request)
            if phase == "queued known then fail" and slot == 1:
                original_cancel = future.cancel
                def cancelled_before_worker_observed():
                    result = original_cancel()
                    if result:
                        cancelled_queued.set()
                    return result
                future.cancel = cancelled_before_worker_observed
            self.futures.append(future)
            return future
        def shutdown(self, **kwargs):
            return actual_pool.shutdown(**kwargs)
    factory = ProductionNestedPackFactory(qualification_receipt={"test_lifecycle": True})
    factory.physics, factory._streams, factory._device = {1: owner}, streams, 0
    factory.xp = SimpleNamespace(cuda=SimpleNamespace(Event=Event, Device=lambda unused: nullcontext(),
        get_current_stream=lambda: central))
    factory._step_pool, factory._step_groups = Pool(), 0
    requests = []
    for member, driver in enumerate(owner.owner.drivers):
        def ordinary(state, cfg):
            return state.physics.compute(state, cfg)
        step = owner.wrap_step(member, ordinary)
        def callback(state, cfg, member=member, step=step, driver=driver):
            trace.append(("before", member))
            result = step(state, cfg)
            assert state.physics is driver
            trace.append(("after", member))
            return result
        binding = SimpleNamespace(context=copy_context(), callbacks={"step": callback})
        requests.append(SimpleNamespace(grid_id=1, binding=binding, args=(driver.state, f"cfg {member}")))
    factory.bindings = tuple(request.binding for request in requests)
    return factory, SimpleNamespace(requests=tuple(requests)), owner, error, trace


@pytest.mark.parametrize("phase", ["event allocation", "event record", "first submit", "partial submit",
                                   "enqueue then fail", "queued known then fail"])
def test_armed_step_launch_failure_aborts_wakes_and_joins_even_unscheduled_members(phase):
    factory, group, owner, error, trace = _step_lifecycle_fixture(phase)
    step_pool = factory._step_pool
    outcome, finished = [], threading.Event()
    def execute():
        try:
            outcome.append(factory.execute_step(group))
        except BaseException as caught:
            outcome.append(caught)
        finally:
            finished.set()
    invocation = threading.Thread(target=execute, daemon=True)
    invocation.start()
    try:
        assert finished.wait(timeout=3), "partial launch left a rendezvous or cancelled Future waiting"
        invocation.join(timeout=2)
        assert not invocation.is_alive() and outcome == [error]
        assert owner._callbacks_finished == {0, 1, 2}
        assert not owner._armed and not owner.owner._active
        assert all(future.done() for future in factory._step_pool.futures)
        assert all(driver.state.physics is driver for driver in owner.owner.drivers)
        assert not any(event == "after" for event, _ in trace)
        assert factory._step_groups == 0
        if phase in {"partial submit", "enqueue then fail", "queued known then fail"}:
            assert factory._step_pool.submissions == ([0, 1, 2] if phase == "queued known then fail" else [0, 1])
            assert trace == [("before", 0)]
            assert factory._streams[0].synchronizations == 1
            if phase == "queued known then fail":
                # This is a genuine Future cancelled inside the real pool's
                # shutdown while its only worker still owns member zero.
                queued = factory._step_pool.futures[1]
                assert queued.cancelled() and queued._state == "CANCELLED"
        else:
            assert trace == []
        factory.close()
        assert owner.owner.trace[-1] == ("close",)
    finally:
        # Even the regressed implementation must fail this test promptly.
        # Retire its blocked first callback before executor shutdown at exit.
        owner.abort(error)
        for member in range(3):
            owner.callback_finished(member)
        step_pool.shutdown(wait=True, cancel_futures=True)
        for future in step_pool.futures:
            if future._state == "CANCELLED":
                future.set_running_or_notify_cancel()
        invocation.join(timeout=2)
        assert not invocation.is_alive(), "failure cleanup left the test invocation parked"
        if owner._armed:
            with pytest.raises(RuntimeError):
                owner.finish()
        factory.close()


def test_successful_original_step_entry_preserves_each_callback_once_and_can_close():
    factory, group, owner, error, trace = _step_lifecycle_fixture("success")
    receipt = None
    try:
        assert factory.execute_step(group) == (0, 1, 2)
        assert factory._step_groups == 1 and owner.receipt["entry_groups"] == 1
        assert owner._callbacks_finished == {0, 1, 2} and not owner._armed
        assert owner.owner.trace[0] == ("begin", ("cfg 0", "cfg 1", "cfg 2"), (None, None, None))
        for member in range(3):
            assert [event for event, slot in trace if slot == member] == ["before", "after"]
        assert all(stream.synchronizations == 1 for stream in factory._streams)
        assert all(driver.state.physics is driver for driver in owner.owner.drivers)
        receipt = factory.receipt()
    finally:
        factory.close()
    assert factory.receipt() == receipt
    assert not factory.physics and not factory.edges and not factory.banks and not factory.bindings
    assert factory._streams == () and factory._step_pool is None


def test_resident_factory_retirement_drops_banks_and_models_but_preserves_receipt():
    import gc
    import weakref
    from gpuwm.ensemble.prepared_nested_batch import ResidentNestedEdgeFactory
    class Handle:
        pass
    bank, edge, model, binding = (Handle() for _ in range(4))
    edge.bank, binding.model = bank, model
    edge.receipt = lambda: {"force_count": 3, "feedback_count": 3}
    observed = weakref.ref(bank), weakref.ref(edge), weakref.ref(binding)
    factory = ResidentNestedEdgeFactory(qualification_receipt={"test_retirement": True})
    factory.edges, factory.banks = {2: edge}, {1: bank}
    factory.bindings, factory.nodes = (binding,), {2: (model,)}
    before = factory.receipt()
    factory.close()
    assert factory.receipt() == before and factory._closed
    factory.close()
    del bank, edge, binding
    gc.collect()
    assert all(reference() is None for reference in observed)
    assert model is not None
    assert not factory.bindings and not factory.nodes and not factory.banks and not factory.edges


def test_hybrid_retirement_preserves_first_cleanup_error_then_releases_on_retry():
    captures = [_actual_capture(member) for member in range(2)]
    original_models = tuple(row[0] for row in captures)
    joined, error = [], RuntimeError("original queue join failed")
    def complete():
        joined.append("caller")
        if len(joined) == 1:
            raise error
    batch = PreparedHybridNestedBatch([row[2] for row in captures],
        ordinary_forecast_bytes={0: 4096, 1: 8192}, available_bytes=20000,
        ordinary_memory_evidence="test retained original envelopes", array_module=np,
        completion_wait=complete)
    with pytest.raises(RuntimeError) as caught:
        batch.close()
    assert caught.value is error and batch.bindings and not batch._closed
    with pytest.raises(RuntimeError, match="retirement started"):
        batch.execute()
    batch.close()
    assert batch._closed and batch.bindings == () and batch._retired_copies == {"into_banks_bytes": 0, "to_originals_bytes": 0}
    assert all(model.root.clock.step_count == 0 for model in original_models)


def test_hybrid_retirement_attempts_all_cleanup_owners_and_keeps_primary_error():
    captures = [_actual_capture(member) for member in range(2)]
    calls = []
    errors = [RuntimeError("packed queue failed"), RuntimeError("original queue failed"), RuntimeError("component close failed")]
    def fail(index):
        calls.append(index)
        raise errors[index]
    batch = PreparedHybridNestedBatch([row[2] for row in captures],
        ordinary_forecast_bytes={0: 4096, 1: 8192}, available_bytes=20000,
        ordinary_memory_evidence="test retained original envelopes", array_module=np,
        completion_wait=lambda: fail(1))
    batch.packed = SimpleNamespace(completion_wait=lambda: fail(0), close=lambda: fail(2))
    with pytest.raises(RuntimeError) as caught:
        batch.close()
    assert caught.value is errors[0] and calls == [0, 1, 2]
    assert "original queue failed" in str(errors[0].__notes__)
    assert "component close failed" in str(errors[0].__notes__)
    assert batch.bindings and not batch._closed
    batch.packed, batch.completion_wait = None, lambda: None
    batch.close()
    assert batch._closed
