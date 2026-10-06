"""Default session selection and original finalization of component routes."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.cfl_member import member_cfl_scope
from gpuwm.core.model import execute_experiment
from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
from gpuwm.ensemble.packing import CardBudget
from gpuwm.ensemble.prepared_component_reservation import ComponentReservationDecision
from gpuwm.ensemble.prepared_route_wave import RouteWaveReservation
from gpuwm.ensemble.production import PreparedEnsembleSession
from gpuwm.ensemble.runtime_context import current_capture
from test_model import _model


class Collector:
    def __init__(self):
        self.frames = []
        self.finished = False

    def submit(self, **frame):
        self.frames.append(frame)

    def finish_run(self):
        self.finished = True
        return {"frames": len(self.frames)}

    def require_complete(self):
        assert self.finished


@dataclass
class Report:
    status: str
    member: int


def fixture(tmp_path, monkeypatch, *, ids=(0, 1), decline=None, failure=None, native_executor=None, source_kind=None,
            model_created=None):
    from gpuwm.ensemble import production, production_component_route, suite_capabilities
    exp, _ = _model()
    shared = SimpleNamespace(experiment=exp, boundary_interval_seconds=3600,
                             authority_sha256={"prepared": "one unchanged authority"})
    collector, events, returned = Collector(), [], {}
    xp = SimpleNamespace(cuda=SimpleNamespace(Stream=lambda **unused: None,
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)),
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None))
    @contextmanager
    def scope(*, member_id, device_id, **unused):
        events.append(("open", member_id))
        try:
            with member_cfl_scope():
                yield SimpleNamespace(receipt=lambda: {"member": member_id})
        finally:
            events.append(("close", member_id))
    monkeypatch.setattr(suite_capabilities, "plan_suite", lambda *args, **kwargs:
        SimpleNamespace(native_fallback_reasons=("legacy fixture has no complete native RK binding",)))
    monkeypatch.setattr("gpuwm.ensemble.ordinary_concurrency.plan_ordinary_concurrency", lambda exp:
        SimpleNamespace(eligible=True, reasons=(), receipt=lambda: {"eligible": True}))
    monkeypatch.setattr("gpuwm.ensemble.member_stream.member_cuda_scope", scope)
    reservation = RouteWaveReservation({member: 100 for member in ids}, {},
        {"collector": 0, "stochastic": 0, "cuda_owners": 0, "allocator_margin": 0},
        {key: "complete bounded CPU owner fixture" for key in
            ("ordinary", "native", "collector", "stochastic", "cuda_owners", "allocator_margin")}, 1000)
    def planning(values, **kwargs):
        events.append(("plan", tuple(values)))
        selected = replace(reservation, ordinary_forecast_bytes={member: 100 for member in values})
        return (None, decline) if decline else (ComponentReservationDecision(selected, None), None)
    monkeypatch.setattr(production, "_prepared_component_route_plan", planning)
    original = production_component_route.run_production_component_wave
    def actual_adapter(session, runner, **kwargs):
        events.append(("component", tuple(kwargs["member_inputs"])))
        # The actual route/capture/clock adapter runs. The fixture has no
        # forecast array inventories, so its live native planner retains
        # original callbacks before a numerical packed operation.
        kwargs.update(array_module=np, member_scope=scope)
        return original(session, runner, **kwargs)
    monkeypatch.setattr(production_component_route, "run_production_component_wave", actual_adapter)
    roster = SimpleNamespace(members=tuple(SimpleNamespace(member_id=member) for member in ids),
        receipt=lambda: {"member_order": list(ids)}, select=lambda requested:
        tuple(SimpleNamespace(inputs=shared, seed=member + 17) for member in requested)) if ids != (0, 1) else None
    source = None
    if source_kind is not None:
        from gpuwm.ensemble.ordinary_execution import OrdinaryRecipeExecution
        from gpuwm.ensemble.posted_execution import PostedRecipeExecution
        source = object.__new__(OrdinaryRecipeExecution if source_kind == "ordinary" else PostedRecipeExecution)
        source.member_order = ids
        source.recipe = SimpleNamespace(sha256="CPU source protocol fixture")
        source._members = {member: SimpleNamespace(index=member, seed=member + 17,
            trajectory=SimpleNamespace(identity="unchanged fixture trajectory", source="gfs",
                cycle=exp.start_time, member=None)) for member in ids}
        source.planning_inputs = lambda member: shared
        source.stochastic_member_binding = lambda member: None
        source.receipt = lambda: {"member_order": list(ids), "source_kind": source_kind}
        source.require_complete = lambda: {"status": "complete", **source.receipt()}
        def source_member(member, *, forecast, observer):
            events.append(("source", member))
            assert current_capture().member_id == member
            return forecast(shared)
        source.run_member = source_member
    session = PreparedEnsembleSession({"members": len(ids)}, output_directory=tmp_path, collector=collector,
        input_provider=(None if roster or source is not None else lambda **unused: shared),
        member_roster=roster, source_execution=source,
        cards=(CardBudget(0, 1000, 1000),), array_module=xp, device_scope=lambda unused: nullcontext(),
        memory_model=EnsembleMemoryModel((MemoryComponent("complete original fixture", "forecast",
            per_member_bytes=100),)), native_executor=native_executor)
    def runner(inputs, *, output_directory, observer=None, first_products=None, schedule_dispatch=None):
        assert inputs is shared and first_products is None
        member = current_capture().member_id
        events.append(("runner", member))
        actual_exp, model = _model()
        if model_created is not None:
            model_created(member, model)
        if current_capture().initialize_callback is not None:
            current_capture().initialize_callback(model=model)
        def step(state, cfg, **unused):
            if failure is not None and member == ids[0]:
                raise failure
            if cfg.grid_id == 1 and observer is not None:
                observer(model_elapsed_seconds=state.elapsed_seconds, outer_step=model.root.clock.step_count)
        execute_experiment(model, experiment=actual_exp, validate_state=False,
            steppers={1: step, 2: step}, pool_trim_per_period=False,
            schedule_dispatch=schedule_dispatch,
            history_handler=lambda tree, node, ticks: collector.submit(member_id=member, grid_id=node.cfg.grid_id))
        events.append(("writer-close", member))
        result = returned[member] = Report("PASS", member)
        return result
    return session, runner, shared, collector, events, returned


def test_actual_constructor_selects_component_route_and_keeps_original_finalization(tmp_path, monkeypatch):
    session, runner, inputs, collector, events, returned = fixture(tmp_path, monkeypatch)
    result = session.run_prepared(runner, inputs)
    assert result["status"] == "PASS" and result["members_completed"] == [0, 1]
    assert events.index(("plan", (0, 1))) < events.index(("open", 0))
    assert ("component", (0, 1)) in events
    assert collector.finished and set(returned) == {0, 1}
    assert result["prepared_component_admission"][0]["eligible"]
    wave = result["prepared_component_waves"][0]
    assert not wave["complete_native_rk_graph"] and not wave["native_component_operations"]
    assert wave["route"]["route_finalization"] == "original_runner"
    assert "original per-member adaptive clocks" in result["time_step_policy"]
    assert [row["member_id"] for row in result["member_results"][0]["result"]["members"]] == [0, 1]
    for member in (0, 1):
        assert events.index(("writer-close", member)) < events.index(("close", member))
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "PASS"
    assert current_capture() is None


def test_checkpointed_members_keep_original_restart_writers_and_disable_component_wave(tmp_path, monkeypatch):
    from gpuwm.ensemble.restart_roster import atomic_json
    session, runner, inputs, collector, events, returned = fixture(tmp_path, monkeypatch)
    inputs.experiment = replace(inputs.experiment, restart_interval_s=12.)
    collector.save_resume = lambda: atomic_json(tmp_path / ".ensemble-resume" / "collector.json", {"files": []})
    result = session.run_prepared(runner, inputs)
    assert result["status"] == "PASS"
    assert "prepared_component_admission" not in result
    assert not any(event[0] in ("plan", "component") for event in events)
    assert set(returned) == {0, 1}
    assert len(result["completed_member_results"]) == 2
    assert (tmp_path / "ensemble-restart.json").is_file()


@pytest.mark.parametrize("reason", ["complete native allocation exceeds the selected budget",
                                  "selected land has no original packed leaf binding"])
def test_cold_memory_or_scheme_decline_keeps_concurrent_originals_and_prints_reason(tmp_path, monkeypatch, capsys, reason):
    session, runner, inputs, collector, events, _ = fixture(tmp_path, monkeypatch, decline=reason)
    result = session.run_prepared(runner, inputs)
    assert ("component", (0, 1)) not in events
    assert result["members_completed"] == [0, 1] and collector.finished
    assert result["member_results"][0]["result"]["backend"] == "ordinary_concurrent_members"
    assert reason in capsys.readouterr().err
    assert not result["prepared_component_admission"][0]["eligible"]
    assert result["prepared_component_waves"] == []


def test_sparse_global_member_order_and_reports_survive_default_adapter(tmp_path, monkeypatch):
    session, runner, inputs, _, events, returned = fixture(tmp_path, monkeypatch, ids=(5, 2))
    result = session.run_prepared(runner, inputs)
    assert result["member_order"] == [5, 2] and set(returned) == {5, 2}
    assert ("component", (5, 2)) in events
    assert [row["member_id"] for row in result["member_results"][0]["result"]["members"]] == [5, 2]


def test_default_adapter_error_preserves_original_exception_and_never_retries(tmp_path, monkeypatch):
    failure = RuntimeError("original callback failed")
    session, runner, inputs, collector, events, _ = fixture(tmp_path, monkeypatch, failure=failure)
    with pytest.raises(RuntimeError) as caught:
        session.run_prepared(runner, inputs)
    assert caught.value is failure and not collector.finished
    assert sum(event == ("runner", 0) for event in events) == 1
    assert sum(event == ("component", (0, 1)) for event in events) == 1
    assert {event[1] for event in events if event[0] == "open"} == {event[1] for event in events if event[0] == "close"}
    manifest = json.loads((tmp_path / "ensemble-run.json").read_text())
    assert manifest["status"] == "failed" and manifest["members_completed"] == []


def test_custom_native_executor_keeps_its_existing_door(tmp_path, monkeypatch):
    calls = []
    def native(**kwargs):
        calls.append(kwargs["batch"].member_indices)
        return {"status": "PASS"}
    session, runner, inputs, _, events, _ = fixture(tmp_path, monkeypatch, native_executor=native)
    result = session.run_prepared(runner, inputs)
    assert calls == [(0, 1)]
    assert not any(event[0] in ("plan", "component", "runner") for event in events)
    assert "prepared_component_admission" not in result


def test_singleton_keeps_its_original_door_without_component_receipts(tmp_path, monkeypatch):
    session, runner, inputs, _, events, _ = fixture(tmp_path, monkeypatch, ids=(0,))
    result = session.run_prepared(runner, inputs)
    assert result["members_completed"] == [0]
    assert not any(event[0] in ("plan", "component") for event in events)
    assert "prepared_component_admission" not in result


@pytest.mark.parametrize("source_kind,component", [("ordinary", True), ("posted", False)])
def test_concrete_source_owner_keeps_its_original_lifetime_and_durable_manifest(tmp_path, monkeypatch, source_kind, component):
    session, runner, inputs, _, events, _ = fixture(tmp_path, monkeypatch, source_kind=source_kind)
    result = session.run_prepared(runner, inputs)
    assert result["posted_source_execution"]["status"] == "complete"
    assert {event[1] for event in events if event[0] == "source"} == {0, 1}
    assert (("component", (0, 1)) in events) is component
    for member in (0, 1):
        assert events.index(("source", member)) < events.index(("runner", member))
    if not component:
        assert "prepared_component_admission" not in result


def test_runner_without_schedule_handoff_keeps_original_calls_and_names_breakage(tmp_path, monkeypatch, capsys):
    session, original, inputs, _, events, _ = fixture(tmp_path, monkeypatch)
    def runner(inputs, *, output_directory, observer=None, first_products=None):
        return original(inputs, output_directory=output_directory, observer=observer, first_products=first_products)
    result = session.run_prepared(runner, inputs)
    assert result["members_completed"] == [0, 1]
    assert ("component", (0, 1)) not in events
    assert "no schedule_dispatch handoff" in capsys.readouterr().err


def test_component_selection_excludes_the_old_first_member_native_probe(tmp_path, monkeypatch):
    from gpuwm.core import preflight
    from gpuwm.ensemble import prepared_execution, production_memory
    session, runner, inputs, _, events, _ = fixture(tmp_path, monkeypatch)
    supplied = session.memory_model
    session.cards = session.memory_model = None
    inputs.execution_plan = object()
    session.array_module.cuda.runtime = SimpleNamespace(getDeviceCount=lambda: 1, getDevice=lambda: 0,
        memGetInfo=lambda: (1000, 1000), getDeviceProperties=lambda unused: {"name": b"CPU admission fixture"})
    session.array_module.get_default_memory_pool = lambda: SimpleNamespace(free_all_blocks=lambda: None, free_bytes=lambda: 0)
    monkeypatch.setattr(preflight, "local_memory_profile_from_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(preflight, "estimate_experiment", lambda *args, **kwargs: object())
    monkeypatch.setattr(production_memory, "ordinary_ensemble_memory_model", lambda *args, **kwargs: (supplied, (), None))
    def old_probe(*args, **kwargs):
        raise AssertionError("the old first-member probe must not precede an admitted component route")
    monkeypatch.setattr(prepared_execution, "make_automatic_prepared_executor", old_probe)
    result = session.run_prepared(runner, inputs)
    assert result["status"] == "PASS" and ("component", (0, 1)) in events
    assert "automatic_admission" not in result


def test_four_default_component_waves_collect_closed_model_cycles_before_next_wave_initialization(tmp_path, monkeypatch):
    import gc
    import weakref
    from dataclasses import replace
    from gpuwm.ensemble import production
    weak, released = {}, []
    automatic_collection = gc.isenabled()
    original_release = production.release_finished_member
    def create(member, model):
        # The budget represents two live initialization owners. A previous
        # wave's retained cycle must not consume a later wave's reservation.
        previous = [ref for old, ref in weak.items() if old // 2 < member // 2]
        if any(ref() is not None for ref in previous):
            raise MemoryError("finished component wave retained its original model allocation")
        model._owned_allocation_cycle = model
        model._owned_payload = bytearray(1024)
        weak[member] = weakref.ref(model)
    session, runner, inputs, _, events, _ = fixture(tmp_path, monkeypatch,
        ids=tuple(range(8)), model_created=create)
    session.request = replace(session.request, max_ordinary_members_per_device=2)
    def release(xp=None):
        if xp is None:
            return original_release()
        completed = json.loads((tmp_path / "ensemble-run.json").read_text())["members_completed"]
        assert completed and len(completed) % 2 == 0
        for member in completed:
            assert ("writer-close", member) in events and ("close", member) in events
        original_release(xp)
        assert all(weak[member]() is None for member in completed)
        released.append(tuple(completed))
    monkeypatch.setattr(production, "release_finished_member", release)
    try:
        gc.disable()
        result = session.run_prepared(runner, inputs)
        assert result["members_completed"] == list(range(8))
        assert released == [tuple(range(count)) for count in (2, 4, 6, 8)]
        assert all(ref() is None for ref in weak.values())
        assert sum(event[0] == "component" for event in events) == 4
    finally:
        if automatic_collection:
            gc.enable()
        gc.collect()
