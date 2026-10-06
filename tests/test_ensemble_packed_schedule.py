"""Original adaptive recursion, headers and carried words survive dispatch."""
from dataclasses import replace
from threading import get_ident, Barrier
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.adaptive_clock import AdaptiveClockDriver
from gpuwm.core.cfl_member import current_cfl_member, member_cfl_scope
from gpuwm.core.clock import build_schedule, execute_schedule, resolve_clock
from gpuwm.ensemble.packed_schedule import (
    MemberScheduleBinding, ScheduleBatchAdmission, execute_packed_member_schedules,
    ReadyMemberOperation, ready_operation_groups, _operation_key,
)
from test_clock import _chain_experiment


def test_optional_original_group_dispatch_matches_standalone_and_records_unpacked_reason():
    references = [_build(member, adaptive=False) for member in (0, 2)]
    for row in references:
        _standalone(row[0])
    actual = [_build(member, adaptive=False) for member in (0, 2)]
    calls = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        def dispatch(group):
            overlap = Barrier(len(group.requests))
            def run(request):
                overlap.wait(timeout=2)
                return request.binding.context.run(request.binding.callbacks[request.kind], *request.args)
            futures = [pool.submit(run, request) for request in group.requests]
            calls.append((group.kind, group.member_ids))
            return tuple(future.result(timeout=2) for future in futures)
        result = execute_packed_member_schedules([row[0] for row in actual],
            admitted_callbacks={"step": lambda group: pytest.fail("refused STEP became packed")},
            batch_admission=lambda group: ScheduleBatchAdmission(False, "original scheme owns its unbound field ABI"),
            original_group_dispatch=dispatch, completion_wait=lambda: None)
    for expected, observed in zip(references, actual, strict=True):
        assert _final(expected[0]) == _final(observed[0])
        assert expected[1] == observed[1]
    assert calls and all(not row["packed"] for row in result["operations"])
    routed = [row for row in result["operations"] if "original_dispatch" in row]
    assert all(row["admission_reason"] for row in routed)
    assert all(row["admission_reason"] == "original scheme owns its unbound field ABI"
               for row in routed if row["kind"] == "step")
    assert all("original_dispatch" not in row for row in result["operations"]
               if row["kind"] not in {"step", "force", "feedback_commit", "feedback_finalize"})


@pytest.mark.parametrize("outcome", ["incomplete", "failed"])
def test_original_group_dispatch_failure_joins_schedule_workers_without_retry(outcome):
    actual = [_build(member, adaptive=False) for member in (0, 2)]
    completed, calls = [], []
    error = RuntimeError("original callback group failed")
    def dispatch(group):
        calls.append(group.kind)
        if outcome == "failed":
            raise error
        return ()
    with pytest.raises(RuntimeError if outcome == "failed" else ValueError) as caught:
        execute_packed_member_schedules([row[0] for row in actual], original_group_dispatch=dispatch,
            completion_wait=lambda: completed.append("joined"))
    if outcome == "failed":
        assert caught.value is error
    else:
        assert "incomplete member roster" in str(caught.value)
    assert calls == ["step"] and completed == ["joined"]
    assert all(row[0].model.root.clock.step_count == 0 for row in actual)


def _snapshot(clock):
    return (clock.ticks, clock.step_ticks, clock.tick_den, clock.step_count,
            clock.dt_fp32.tobytes(), clock.dtbc_fp32.tobytes(), clock.adaptive_state)


def _build(member, *, adaptive=True, signature="common"):
    exp = _chain_experiment((1, 3), run_seconds=120., history_s=30.)
    domains = tuple(replace(domain, run=replace(domain.run,
        use_adaptive_time_step=adaptive, min_time_step=1, max_time_step=30 if domain.grid_id == 1 else 10,
        starting_time_step=12 if domain.grid_id == 1 else 4, max_step_increase_pct=20,
        step_to_output_time=True)) for domain in exp.domains)
    if not adaptive:
        domains = tuple(replace(domain, history_interval_s=60., run=replace(domain.run,
            output_interval_s=60.)) for domain in domains)
    exp = replace(exp, domains=domains, restart_interval_s=60.)
    clock = resolve_clock(exp, lbc_interval_s=60.)
    schedule = build_schedule(exp, clock)
    clocks = clock.clocks()
    nodes = {}
    for domain in domains:
        parent = None if domain.parent_id == 0 else nodes[domain.parent_id]
        nodes[domain.grid_id] = SimpleNamespace(cfg=domain, clock=clocks[domain.grid_id],
            state=SimpleNamespace(words=np.array([member + 17, member + 53], dtype=np.uint32),
                elapsed_seconds=0., domain_start_offset=0., physics=None),
            parent=parent, children=[], coupler=None)
        if parent is not None:
            parent.children.append(nodes[domain.grid_id])
    model = SimpleNamespace(root=nodes[1], schedule=schedule,
        walk_parent_first=lambda: iter(nodes.values()), node=lambda gid: nodes[gid])
    trace, threads = [], []
    pattern = ((.01, .01), (.3, .2), (.01, .01), (.7, .3)) if member != 1 else (
        (.01, .01), (2.4, 1.7), (.2, .1), (1.8, .9))
    def cfl(gid):
        owner = current_cfl_member()
        assert owner is not None
        return owner.banks["_WRF_CFL_LAST"].get(gid, (0., 0.))
    driver = AdaptiveClockDriver(model, cfl_source=cfl, tick_den=clock.tick_den,
        map_factor_source=lambda gid: 1.) if adaptive else None
    def record(kind, args):
        threads.append(get_ident())
        if kind in {"step", "before_step", "domain_start"}:
            payload = (args[0], _snapshot(args[1]))
        elif kind in {"force", "feedback_prepare", "feedback_commit", "feedback_finalize"}:
            payload = (args[0], args[1], _snapshot(args[2]), _snapshot(args[3]))
        elif kind.startswith("period_"):
            payload = (args[0], tuple((gid, _snapshot(dom)) for gid, dom in args[1].items()))
        else:
            payload = args
        trace.append((kind, payload))
    def stamp(gid, dom):
        return np.uint32((dom.ticks ^ dom.step_ticks ^ (gid << 16)) & 0xffffffff)
    def step_metadata(gid, dom):
        node = model.node(gid)
        node.state.elapsed_seconds = dom.elapsed_seconds
        if adaptive:
            current_cfl_member().banks["_WRF_CFL_LAST"][gid] = pattern[dom.step_count % len(pattern)]
        record("step", (gid, dom))
    def step(gid, dom):
        node = model.node(gid)
        node.state.words ^= stamp(gid, dom)
        step_metadata(gid, dom)
    def force(gid, pid, child, parent):
        model.node(gid).state.words[:] = model.node(pid).state.words ^ stamp(gid, parent)
        record("force", (gid, pid, child, parent))
    def feedback_commit(gid, pid, child, parent):
        model.node(pid).state.words ^= model.node(gid).state.words ^ stamp(pid, child)
        record("feedback_commit", (gid, pid, child, parent))
    callbacks = {kind: lambda *args, kind=kind: record(kind, args) for kind in (
        "before_step", "feedback_prepare", "feedback_finalize", "history", "restart", "lbc_reset",
        "period_begin", "period_end", "period_commit", "domain_start")}
    callbacks.update(step=step, force=force, feedback_commit=feedback_commit)
    binding = MemberScheduleBinding(member, model, callbacks,
        operation_authority=lambda **unused: {"complete_component_layout": "two-domain-carried-word-test",
            "source_layout": signature, "geometry": "same-original-chain"}, adaptive_driver=driver)
    return binding, trace, threads, stamp, step_metadata


def _standalone(binding):
    callbacks = {"on_" + name: callback for name, callback in binding.callbacks.items() if name != "before_step"}
    original_step = binding.callbacks["step"]
    def step(gid, clock):
        if binding.adaptive_driver is not None:
            binding.adaptive_driver.before_step(gid)
        binding.callbacks["before_step"](gid, clock)
        original_step(gid, clock)
    callbacks["on_step"] = step
    with member_cfl_scope():
        return execute_schedule(binding.model.schedule, clocks=binding.clocks,
            on_period_steps=binding.adaptive_driver, **callbacks)


def _final(binding):
    return {gid: (node.state.words.tobytes(), _snapshot(node.clock)) for gid, node in
            ((gid, binding.model.node(gid)) for gid in binding.clocks)}


def test_divergent_original_adaptive_schedules_match_standalone_order_and_carried_words():
    references = [_build(member) for member in range(3)]
    reference_reports = [_standalone(binding) for binding, *_ in references]
    together = [_build(member) for member in range(3)]
    caller = get_ident()
    numerical_calls = []
    def packed_step(group):
        # A real vector operation on the test's carried word buffers. The
        # test qualifies dispatch, not atmospheric kernels or a forecast.
        values = np.stack([request.binding.model.node(request.grid_id).state.words for request in group.requests])
        words = np.array([together[request.member_id][3](request.grid_id, request.args[1])
                          for request in group.requests], dtype=np.uint32)
        np.bitwise_xor(values, words[:, None], out=values)
        for request, value in zip(group.requests, values, strict=True):
            request.binding.model.node(request.grid_id).state.words[:] = value
            request.binding.context.run(together[request.member_id][4], *request.args)
        numerical_calls.append(group.member_ids)
        return (None,) * len(group.requests)
    receipt = execute_packed_member_schedules([row[0] for row in together],
        admitted_callbacks={"step": packed_step},
        batch_admission=lambda group: ScheduleBatchAdmission(True, "test carried-word banks", "member_owned_rings"),
        completion_wait=lambda: None)
    assert numerical_calls and any(len(ids) > 1 for ids in numerical_calls)
    assert any(not row["packed"] and row["kind"] == "step" for row in receipt["operations"])
    for (reference, reference_trace, _, _, _), (binding, trace, threads, _, _) in zip(references, together, strict=True):
        assert trace == reference_trace
        assert _final(binding) == _final(reference)
        assert set(threads) == {caller}, "metadata workers invoked an original numerical/controller callback"
        assert all(not bank for bank in binding.cfl_owner.banks.values())
        observed = receipt["reports"][binding.member_id]
        expected = reference_reports[binding.member_id]
        for field in ("steps", "forces", "feedback_calls", "histories", "restarts", "lbc_resets"):
            assert getattr(observed, field) == getattr(expected, field)
    assert receipt["reports"][0].steps != receipt["reports"][1].steps
    assert receipt["forecast_graph_admission_changed"] is False


def test_fixed_nested_walk_preserves_original_terminal_feedback_and_deadlines():
    references = [_build(member, adaptive=False) for member in range(2)]
    for binding, *_ in references:
        _standalone(binding)
    together = [_build(member, adaptive=False) for member in range(2)]
    execute_packed_member_schedules([row[0] for row in together], completion_wait=lambda: None)
    for (reference, old_trace, *_), (binding, trace, *_) in zip(references, together, strict=True):
        # The dispatcher only inserts before_step when its owner supplies
        # that hook; this fixture supplies it in both original paths.
        assert trace == old_trace
        assert _final(binding) == _final(reference)


def test_member_clock_lattice_mismatch_fails_without_leaving_a_worker_parked():
    bindings = [_build(member)[0] for member in range(2)]
    bindings[1].model.node(2).clock.tick_den += 1
    completed = []
    with pytest.raises(ValueError, match="clock lattice"):
        execute_packed_member_schedules(bindings, completion_wait=lambda: completed.append(True))
    assert completed == [True]


def test_callback_failure_joins_parked_workers_and_preserves_first_error():
    bindings = [_build(member)[0] for member in range(4)]
    bindings[0].callbacks["step"] = lambda *args: (_ for _ in ()).throw(ValueError("step failed"))
    def failed_completion():
        raise RuntimeError("queue wait also failed")
    with pytest.raises(ValueError, match="step failed") as caught:
        execute_packed_member_schedules(bindings, completion_wait=failed_completion)
    assert any("queue wait also failed" in note for note in caught.value.__notes__)
    assert caught.value.ensemble_member_ids == (0,)


def test_adaptive_packed_step_requires_member_cfl_binding_before_mutation():
    bindings = [_build(member)[0] for member in (0, 2)]
    calls = []
    with pytest.raises(ValueError, match="private member CFL-ring"):
        execute_packed_member_schedules(bindings, admitted_callbacks={"step": lambda group: calls.append(group)},
            batch_admission=lambda group: ScheduleBatchAdmission(True, "missing ring owner"), completion_wait=lambda: None)
    assert calls == []


def test_source_authority_and_original_physics_cadence_partition_step_groups():
    first, second = [_build(member, adaptive=False)[0] for member in range(2)]
    args1, args2 = (1, first.model.root.clock), (1, second.model.root.clock)
    def request(binding, args):
        return ReadyMemberOperation(binding, "step", args, 0, _operation_key(binding, "step", args))
    assert len(ready_operation_groups((request(first, args1), request(second, args2)))) == 1
    second.operation_authority = lambda **unused: "different source field layout"
    assert len(ready_operation_groups((request(first, args1), request(second, args2)))) == 2
    second.operation_authority = first.operation_authority
    first.model.root.state.physics = SimpleNamespace(bldt_seconds=2., radt_minutes=0., cudt_minutes=0.)
    second.model.root.state.physics = SimpleNamespace(bldt_seconds=3., radt_minutes=0., cudt_minutes=0.)
    assert len(ready_operation_groups((request(first, args1), request(second, args2)))) == 2


def test_schedule_refuses_shared_models_and_missing_numerical_step_owners():
    first, second = [_build(member, adaptive=False)[0] for member in range(2)]
    second.model = first.model
    with pytest.raises(ValueError, match="cannot share an original model"):
        execute_packed_member_schedules((first, second), completion_wait=lambda: None)
    with pytest.raises(ValueError, match="STEP callback"):
        MemberScheduleBinding(9, first.model, {}, operation_authority=lambda **unused: "test")


def test_replaced_member_clock_identity_is_diagnosed_before_callback():
    binding = _build(0, adaptive=False)[0]
    old_clock = binding.model.root.clock
    binding.model.root.clock = binding.model.schedule.clock.domain_clock(1)
    with pytest.raises(ValueError, match="clock identity"):
        _operation_key(binding, "step", (1, old_clock))


def test_equal_current_step_groups_preserve_cadence_dependencies_but_not_prior_cfl_equality():
    first, second = [_build(member, adaptive=False)[0] for member in range(2)]
    history = {"last_dt_num": 12, "last_dt_den": 1, "radiation_seen": 30.,
               "radiation_actual": 30., "cumulus_fired": 20., "last_max_vert_cfl": .2,
               "last_max_horiz_cfl": .3, "started": True, "stepping_to_time": False}
    first.model.root.clock.adaptive_state = dict(history)
    second.model.root.clock.adaptive_state = dict(history, last_max_vert_cfl=.9, last_max_horiz_cfl=.8)
    def request(binding):
        args = (1, binding.model.root.clock)
        return ReadyMemberOperation(binding, "step", args, 0, _operation_key(binding, "step", args))
    assert len(ready_operation_groups((request(first), request(second)))) == 1
    second.model.root.clock.adaptive_state["radiation_seen"] = 45.
    assert len(ready_operation_groups((request(first), request(second)))) == 2
    second.model.root.clock.adaptive_state["radiation_seen"] = 30.
    second.model.root.clock.adaptive_state["last_dt_num"] = 15
    assert len(ready_operation_groups((request(first), request(second)))) == 2
