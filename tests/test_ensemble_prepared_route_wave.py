"""Original route TLS/owners, complete preprice and failure join controls."""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
import ast
import inspect
import threading

import numpy as np
import pytest

from gpuwm.core.cfl_member import current_cfl_member, member_cfl_scope
from gpuwm.core.model import execute_experiment
from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
from gpuwm.ensemble.prepared_route_wave import PreparedRouteWave, RouteMember, RouteWaveReservation
from gpuwm.ensemble.runtime_context import MemberOutputCapture, current_capture, member_output_scope
from test_model import _model


def reservation(*, plans=None, available=20000):
    return RouteWaveReservation({0: 4096, 1: 4096}, plans or {},
        {"collector": 256, "stochastic": 0, "cuda_owners": 128, "allocator_margin": 256},
        {"ordinary": "complete fixture forecast envelopes", "native": "named fixture component plans",
         "collector": "bounded fixture writer reserve", "stochastic": "inactive",
         "cuda_owners": "fixture queue owner reserve", "allocator_margin": "fixture allocator reserve"}, available)


def fixture(*, barrier=None, callback_error=None, initialization_error=None, finalization_error=None):
    tls, trace, models, owners = threading.local(), [], {}, {}
    shared = SimpleNamespace(authority="one unchanged shared prepared bank")
    @contextmanager
    def scope(*, member_id, device_id):
        assert device_id == 0
        tls.member, tls.thread = member_id, threading.get_ident()
        trace.append(("scope-enter", member_id))
        with member_cfl_scope() as cfl:
            owners[member_id] = cfl
            try:
                yield SimpleNamespace(stream=None, pool=object())
            finally:
                trace.append(("scope-close", member_id))
        del tls.member
    def runner(inputs, *, output_directory, schedule_dispatch=None):
        member = current_capture().member_id
        assert inputs is shared
        if initialization_error is not None and member == 1:
            raise initialization_error
        exp, model = _model()
        models[member] = model
        for node in model.walk_parent_first():
            node.state.words = np.array([member + 17, member + 31], np.uint32)
        writer = SimpleNamespace(thread=threading.get_ident(), member=member, closed=False)
        trace.append(("writers-ready", member))
        seen = set()
        def require_owners():
            assert threading.get_ident() == writer.thread == tls.thread
            assert tls.member == member == current_capture().member_id
            assert current_cfl_member() is owners[member]
        def step(state, cfg, **unused):
            require_owners()
            if barrier is not None and not seen:
                barrier.wait(timeout=3)
            seen.add(cfg.grid_id)
            trace.append(("step", member, cfg.grid_id))
            if callback_error is not None and member == 0:
                raise callback_error
            state.words ^= np.uint32(cfg.grid_id + 5)
        def history(tree, node, ticks):
            require_owners()
            current_capture().callback(member_id=member, grid_id=node.cfg.grid_id, ticks=ticks,
                                       words=node.state.words.tobytes())
        def dispatch(**kwargs):
            require_owners()
            assert all(kwargs["step_bindings"][gid] is value for gid, value in steppers.items())
            trace.append(("dispatch", member))
            return schedule_dispatch(**kwargs)
        steppers = {1: step, 2: step}
        try:
            report = execute_experiment(model, validate_state=False, experiment=exp,
                steppers=steppers, history_handler=history, pool_trim_per_period=False,
                schedule_dispatch=None if schedule_dispatch is None else dispatch)
            require_owners()
            if finalization_error is not None and member == 0:
                raise finalization_error
            return {"status": "PASS", "report": report}
        finally:
            writer.closed = True
            trace.append(("writers-close", member))
    frames = []
    members = tuple(RouteMember(member, shared, f"unused-{member}",
        MemberOutputCapture(lambda **frame: frames.append(frame), member),
        lambda **unused: {"bank": "one unchanged shared prepared bank"}) for member in range(2))
    return members, runner, scope, trace, models, owners, frames


def words(model):
    return [(node.state.words.tobytes(), node.state.elapsed_seconds, node.clock.ticks,
             node.clock.step_count, node.clock.dt_fp32.tobytes(), node.clock.dtbc_fp32.tobytes(),
             tuple(node.coupler.calls) if node.coupler is not None else ())
            for node in model.walk_parent_first()]


def test_parked_actual_executor_preserves_route_tls_cfl_writers_history_and_standalone_words():
    members, runner, scope, trace, models, owners, frames = fixture(barrier=threading.Barrier(2))
    wave = PreparedRouteWave(members, reservation=reservation(), runner=runner, member_scope=scope)
    result = wave.run()
    expected_members, expected_runner, expected_scope, _, expected_models, _, expected_frames = fixture()
    for member in expected_members:
        with expected_scope(member_id=member.member_id, device_id=0), member_output_scope(member.capture):
            expected_runner(member.inputs, output_directory=member.output_directory)
    assert [words(models[member]) for member in range(2)] == [words(expected_models[member]) for member in range(2)]
    key = lambda row: (row["member_id"], row["grid_id"], row["ticks"])
    assert sorted(frames, key=key) == sorted(expected_frames, key=key)
    assert result["route_finalization"] == "original_runner"
    assert result["execution"]["packed_components"] is False
    assert result["default_door_enabled"] is False
    for member, endpoint in wave.endpoints.items():
        assert endpoint.retired_owner_identity["cfl"] == id(owners[member])
        assert endpoint.binding is None and endpoint.adaptive_driver is None
        assert not endpoint.callbacks and endpoint.stream is None
        assert trace.index(("writers-ready", member)) < trace.index(("dispatch", member)) < trace.index(("writers-close", member))
        assert ("scope-close", member) in trace
        assert endpoint._terminated


def test_complete_ordinary_preprice_refuses_before_any_runner_or_scope_starts():
    members, runner, scope, trace, *_ = fixture()
    wave = PreparedRouteWave(members, reservation=reservation(available=1), runner=runner, member_scope=scope)
    with pytest.raises(MemoryError, match="ordinary route wave requires"):
        wave.run()
    assert not trace


def test_native_memory_refusal_before_threads_uses_the_admitted_concurrent_original_wave():
    members, runner, scope, trace, *_ = fixture(barrier=threading.Barrier(2))
    plan = BatchMemoryPlan((BatchArraySpec("large", (20000,), "member"),), 0)
    class UnusedFactory:
        def memory_plans(self, *unused):
            raise AssertionError("native refusal must precede component inspection")
    result = PreparedRouteWave(members, reservation=reservation(plans={"physics:1": plan}),
        runner=runner, member_scope=scope, pack_factory=UnusedFactory()).run()
    assert "native component wave needs" in result["execution"]["fallback_reason"]
    assert not result["native_admitted_before_threads"]
    assert len([row for row in trace if row[0] == "scope-close"]) == 2


@pytest.mark.parametrize("failure", ["initialization", "callback", "finalization"])
def test_original_route_failure_wakes_and_joins_every_worker_without_retry(failure):
    error = RuntimeError(f"injected {failure}")
    options = {failure + "_error": error}
    if failure == "callback":
        options["barrier"] = threading.Barrier(2)
    members, runner, scope, trace, *_ = fixture(**options)
    wave = PreparedRouteWave(members, reservation=reservation(), runner=runner, member_scope=scope)
    with pytest.raises(RuntimeError) as outcome:
        wave.run()
    assert outcome.value is error
    assert len([row for row in trace if row[0] == "scope-enter"]) == 2
    assert len([row for row in trace if row[0] == "scope-close"]) == 2
    assert len([row for row in trace if row[0] == "writers-ready"]) == (1 if failure == "initialization" else 2)
    if failure == "callback":
        assert len([row for row in trace if row[0] == "step" and row[1] == 0]) == 1


def test_runner_that_ignores_dispatch_aborts_parked_peers():
    members, original, scope, trace, *_ = fixture()
    def broken(inputs, *, schedule_dispatch, **kwargs):
        if current_capture().member_id == 1:
            return {"status": "PASS"}
        return original(inputs, schedule_dispatch=schedule_dispatch, **kwargs)
    with pytest.raises(RuntimeError, match="returned without its route schedule handoff"):
        PreparedRouteWave(members, reservation=reservation(), runner=broken, member_scope=scope).run()
    assert len([row for row in trace if row[0] == "scope-close"]) == 2


def test_wave_cannot_reinitialize_or_reuse_completed_member_owners():
    members, runner, scope, *_ = fixture()
    wave = PreparedRouteWave(members, reservation=reservation(), runner=runner, member_scope=scope)
    wave.run()
    with pytest.raises(RuntimeError, match="single-use"):
        wave.run()


@pytest.mark.parametrize("change", ["name", "shape", "dtype", "transient"])
def test_live_allocation_changes_cannot_escape_prethread_reservation(change):
    source = BatchMemoryPlan((BatchArraySpec("a", (8,), "member"),), 0)
    changed = BatchMemoryPlan((BatchArraySpec("b" if change == "name" else "a",
        (16,) if change == "shape" else (8,), "member", "float64" if change == "dtype" else "float32"),),
        512 if change == "transient" else 0)
    from gpuwm.ensemble.batch_state import BatchStateUnsupported
    with pytest.raises(BatchStateUnsupported, match="live"):
        reservation(plans={"bank:1": source}).validate_live_plans({"bank:1": changed})


@pytest.mark.parametrize("edge_only", [False, True])
@pytest.mark.parametrize("changed", [False, True])
def test_parked_route_live_bank_inventory_matches_factory_field_policy_without_weakening_preprice(monkeypatch, edge_only, changed):
    from gpuwm.core.state import DomainState
    from gpuwm.ensemble.batch_state import BatchStateUnsupported
    from gpuwm.ensemble.prepared_nested_batch import ResidentNestedEdgeFactory, _domain_bank_plan
    models = [_model()[1] for _ in range(2)]
    for model in models:
        for node in model.walk_parent_first():
            node.cfg = replace(node.cfg, run=replace(node.cfg.run, nx=9, ny=9, nz=8))
            node.state = DomainState(node.cfg.run, array_module=np)
    bindings = tuple(SimpleNamespace(clocks={1: model.root.clock, 2: model.node(2).clock},
        model=model) for model in models)
    native = ResidentNestedEdgeFactory(qualification_receipt={"scope": "actual CPU bank inventory"},
        edge_state_only=edge_only)
    cold = {f"bank:{gid}": _domain_bank_plan(tuple(model.node(gid) for model in models),
        shared_fields=(), array_module=np, field_names=native.state_fields_for_grid(bindings, gid))
        for gid in (1, 2)}
    expected = {gid: {spec.name for spec in cold[f"bank:{gid}"].arrays} for gid in (1, 2)}
    if edge_only:
        assert "u" in expected[1] and "u0" not in expected[1]
    if changed:
        cold["bank:1"] = BatchMemoryPlan(tuple(spec for spec in cold["bank:1"].arrays if spec.name != "u"), 0)
    inspected, allocated = [], []
    def inspect_components(actual, descriptors):
        assert actual is bindings
        inspected.append({gid: set(descriptors[gid].storage.specs) for gid in (1, 2)})
        return {}
    monkeypatch.setattr(native, "memory_plans", inspect_components)
    class Batch:
        def __init__(self, actual, **kwargs):
            allocated.append(actual)
        def execute(self, *, original_group_dispatch):
            raise AssertionError("inventory control stops before any forecast operation")
    members, runner, scope, *_ = fixture()
    wave = PreparedRouteWave(members, reservation=reservation(plans=cold, available=1 << 40),
        runner=runner, member_scope=scope, array_module=np, pack_factory=native, batch_factory=Batch,
        shared_fields=())
    if changed:
        with pytest.raises(BatchStateUnsupported, match="live allocation bank:1/u differs from its pre-thread reservation"):
            wave._native_batch(bindings)
        assert not allocated
    else:
        assert isinstance(wave._native_batch(bindings), Batch)
        assert inspected == [expected] and allocated == [bindings]


def test_missing_fixed_reservation_category_is_refused():
    value = reservation()
    with pytest.raises(ValueError, match="explicit collector"):
        replace(value, fixed_bytes={"collector": 0})


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_boolean_memory_reservation_is_not_an_integer_byte_budget(value):
    with pytest.raises(TypeError, match="integer"):
        replace(reservation(), available_bytes=value)


def test_prepared_runner_default_is_none_and_both_real_executor_calls_forward_opt_in_dispatch():
    """Both real executor paths (no writers, and writers) forward an opt-in dispatch.

    Until 1ec83bf01 each path spelled out its own ``execute_experiment`` call,
    and this test counted two.  Bounded nested health recovery folded both
    into ONE leg, ``execute_leg``, that each path hands to ``recovery.run`` so
    a retry leg re-enters the same executor with the same arguments.  The
    property is unchanged and is pinned here at least as tightly: exactly one
    executor call, inside ``execute_leg``, forwarding the dispatch only when
    one was given; exactly two ``recovery.run(execute_leg)`` calls, one on
    each side of ``if writers is None``; and no path that reaches the leg
    any other way.
    """
    from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
    assert inspect.signature(run_prepared_tree).parameters["schedule_dispatch"].default is None
    tree = ast.parse(inspect.getsource(run_prepared_tree))

    def is_executor(node):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "execute_experiment")

    calls = [node for node in ast.walk(tree) if is_executor(node)]
    assert len(calls) == 1
    legs = [node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "execute_leg"]
    assert len(legs) == 1
    assert [node for node in ast.walk(legs[0]) if is_executor(node)] == calls
    call = calls[0]
    dispatch = [arg.value for arg in call.keywords if arg.arg is None]
    assert len(dispatch) == 1
    assert isinstance(dispatch[0], ast.IfExp)
    assert ast.unparse(dispatch[0].test) == "schedule_dispatch is None"
    assert ast.literal_eval(dispatch[0].body) == {}
    assert dispatch[0].orelse.keys[0].value == "schedule_dispatch"
    assert dispatch[0].orelse.values[0].id == "schedule_dispatch"
    history = [arg.value for arg in call.keywords if arg.arg == "history_handler"]
    assert [ast.unparse(value) for value in history] == [
        "None if writers is None else history_handler"]

    runs = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "recovery.run"]
    assert [ast.unparse(node) for node in runs] == ["recovery.run(execute_leg)"] * 2
    branch = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "writers is None"
              and any(run in ast.walk(node) for run in runs)]
    assert len(branch) == 1
    in_body = [run for run in runs if any(run in ast.walk(stmt) for stmt in branch[0].body)]
    in_else = [run for run in runs if any(run in ast.walk(stmt) for stmt in branch[0].orelse)]
    assert len(in_body) == 1 and len(in_else) == 1
    uses = [node for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == "execute_leg"]
    assert len(uses) == 2 and all(any(use is run.args[0] for run in runs) for use in uses)


def test_a_dispatched_member_leg_is_never_re_entered_by_health_recovery():
    """The shared leg above may be re-run by health recovery; a member never is.

    A route member runs under its own output capture (``RouteMember`` refuses
    one without it) and its dispatch hook is a live endpoint of the wave.
    Re-entering the leg would hand that hook a second schedule after the
    first had failed and leave the wave's other members waiting on a member
    that already reported.  Recovery must refuse under a capture, by name,
    and run the leg exactly once.
    """
    from gpuwm.core.health import HealthCheckError, validate_fields_cpu
    from gpuwm.stability_recovery import NestedHealthRecovery
    import tempfile
    report = validate_fields_cpu({"w": np.array([239.79], np.float32)}, phase="post-d01-sync.d02")
    assert not report.ok
    model = SimpleNamespace(root=SimpleNamespace(clock=SimpleNamespace(elapsed_seconds=3600)),
                            _last_checkpoint=None)
    experiment = SimpleNamespace(relocation=SimpleNamespace(enabled=False), domains=())
    legs = []
    def leg(active):
        legs.append(active)
        raise HealthCheckError(report)
    with tempfile.TemporaryDirectory() as outdir:
        recovery = NestedHealthRecovery(model=model, experiment=experiment, output_directory=outdir)
        with member_output_scope(SimpleNamespace(member_id="m01")):
            with pytest.raises(HealthCheckError):
                recovery.run(leg)
    assert legs == [experiment]
    assert recovery.receipt["status"] == "REFUSED"
    assert "ensemble capture" in recovery.receipt["refusal"]
    assert recovery.receipt["attempts"] == []


@pytest.mark.parametrize("enabled", [False, True])
def test_prepared_session_recursion_preserves_the_explicit_dispatch_only_when_active(monkeypatch, enabled):
    import gpuwm.ensemble.runtime_context as context
    from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
    calls = []
    session = SimpleNamespace(run_prepared=lambda *args, **kwargs: calls.append((args, kwargs)) or "handled")
    monkeypatch.setattr(context, "current_session", lambda: session)
    dispatch = (lambda **unused: None) if enabled else None
    assert run_prepared_tree(SimpleNamespace(), output_directory="unused", io_mode="none", schedule_dispatch=dispatch) == "handled"
    assert ("schedule_dispatch" in calls[0][1]) is enabled
    if enabled:
        assert calls[0][1]["schedule_dispatch"] is dispatch


@pytest.mark.parametrize("phase", ["success", "mutated callback failure", "component close failure", "partial RPC launch failure"])
def test_native_rpc_waits_ready_runs_on_route_tls_drains_real_route_streams_and_never_retries(monkeypatch, phase):
    trace, tls = [], threading.local()
    class Stream:
        def __init__(self, owner):
            self.owner = owner
        def wait_event(self, event):
            trace.append(("wait", self.owner, event.stream.owner))
        def synchronize(self):
            trace.append(("synchronize", self.owner))
    manager = Stream("manager")
    error = RuntimeError(phase)
    manager_calls = [0]
    class Event:
        def __init__(self):
            if phase == "partial RPC launch failure" and ("component-inspection",) in trace:
                manager_calls[0] += 1
                if manager_calls[0] == 2:
                    raise error
        def record(self, stream):
            self.stream = stream
            trace.append(("record", stream.owner))
    cuda = SimpleNamespace(Event=Event, Device=lambda unused: nullcontext(),
        get_current_stream=lambda: getattr(tls, "stream", manager))
    members, runner, ordinary_scope, route_trace, models, *_ = fixture(
        barrier=None if phase == "partial RPC launch failure" else threading.Barrier(2))
    @contextmanager
    def scope(**kwargs):
        tls.stream = Stream(kwargs["member_id"])
        with ordinary_scope(**kwargs) as value:
            yield value
        del tls.stream
    class Batch:
        def execute(self, *, original_group_dispatch):
            requests = tuple(SimpleNamespace(member_id=member.member_id, kind="step",
                args=(1, models[member.member_id].root.clock)) for member in members)
            assert original_group_dispatch(SimpleNamespace(requests=requests)) == (None, None)
            if phase == "mutated callback failure":
                raise error
            return {"reports": {member.member_id: "genuine schedule result" for member in members}}
        def close(self):
            trace.append(("component-close",))
            if phase == "component close failure":
                raise error
    def prepared(wave, bindings):
        trace.append(("component-inspection",))
        return Batch()
    monkeypatch.setattr(PreparedRouteWave, "_native_batch", prepared)
    wave = PreparedRouteWave(members, reservation=reservation(), runner=runner, member_scope=scope,
        array_module=SimpleNamespace(cuda=cuda), pack_factory=object())
    if phase == "success":
        result = wave.run()
        assert all(record["report"] == "genuine schedule result" for record in result["member_results"].values())
    else:
        with pytest.raises(RuntimeError) as outcome:
            wave.run()
        assert outcome.value is error
    inspection = trace.index(("component-inspection",))
    for member in range(2):
        assert trace.index(("record", member)) < trace.index(("wait", "manager", member)) < inspection
        if phase != "partial RPC launch failure":
            assert ("wait", member, "manager") in trace
        assert ("synchronize", member) in trace
        assert len([row for row in route_trace if row[0] == "step" and row[1] == member]) <= 1
        assert ("writers-close", member) in route_trace and ("scope-close", member) in route_trace


def test_partial_runner_submission_failure_joins_a_worker_that_already_parked(monkeypatch):
    import gpuwm.ensemble.prepared_route_wave as module
    from concurrent.futures import ThreadPoolExecutor
    error = RuntimeError("injected thread submit failure")
    class Pool:
        def __init__(self, **kwargs):
            self.pool, self.calls = ThreadPoolExecutor(**kwargs), 0
        def submit(self, *args):
            self.calls += 1
            if self.calls == 2:
                raise error
            return self.pool.submit(*args)
        def shutdown(self, **kwargs):
            self.pool.shutdown(**kwargs)
    monkeypatch.setattr(module, "ThreadPoolExecutor", Pool)
    members, runner, scope, trace, *_ = fixture()
    wave = PreparedRouteWave(members, reservation=reservation(), runner=runner, member_scope=scope)
    with pytest.raises(RuntimeError) as outcome:
        wave.run()
    assert outcome.value is error
    assert ("scope-close", 0) in trace
    assert not any(row[0] == "step" for row in trace)
