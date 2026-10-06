"""CPU original-source/capture/progress and joined route-owner controls."""
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
import threading

import pytest

from gpuwm.core.cfl_member import member_cfl_scope
from gpuwm.core.model import execute_experiment
from gpuwm.ensemble.batch_state import BatchStateUnsupported
from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
from gpuwm.ensemble.execution import MemberRunControl, MemberStopRequested, current_run_control
from gpuwm.ensemble.ordinary_execution import OrdinaryRecipeExecution
from gpuwm.ensemble.prepared_route_wave import RouteWaveReservation
from gpuwm.ensemble.production_component_route import run_production_component_wave, source_component_route_reason
from gpuwm.ensemble.progress import EnsembleProgressAdapter
from gpuwm.ensemble.runtime_context import current_capture
from test_model import _model


@dataclass
class OriginalReport:
    status: str
    member: int
    prepared_tag: str


def setup(tmp_path, *, source=False, callback_error=None, stop=False, ids=(0, 1)):
    trace, frames, initialized, returned, configured = [], [], [], {}, []
    control = MemberRunControl()
    shared = SimpleNamespace(tag="one shared prepared root", experiment=_model()[0])
    original_source = SimpleNamespace(tag="actual original source input", experiment=shared.experiment)
    collector = SimpleNamespace(submit=lambda **frame: frames.append(frame))
    source_owner = None
    if source:
        # The source protocol is isolated here; seal verification remains
        # covered by the original owner's own prepared-input tests.
        source_owner = object.__new__(OrdinaryRecipeExecution)
        def run_member(member_id, *, forecast, observer):
            trace.append(("source", member_id, current_capture().member_id, observer))
            return forecast(original_source)
        source_owner.run_member = run_member
        source_owner.receipt = lambda: {"original_source": "forecasting"}
    session = SimpleNamespace(source_execution=source_owner, array_module=SimpleNamespace(), collector=collector,
        request=SimpleNamespace(keep_member_files=False), _stochastic_authorities={})
    def initialization(member):
        return lambda **kwargs: initialized.append((member, kwargs["model"]))
    session._initialization_callback = initialization
    def configure(actual):
        assert actual is original_source and current_run_control() is control
        result = SimpleNamespace(tag="configured actual source input", experiment=shared.experiment)
        session._stochastic_authorities[id(result)] = {"source": "original stochastic authority"}
        configured.append(result)
        return result
    session._configured_member_inputs = configure
    def source_context(**kwargs):
        assert any(kwargs["actual_inputs"] is value for value in configured)
        assert kwargs["authority"] == {"source": "original stochastic authority"}
        assert kwargs["source_receipt"] == {"original_source": "forecasting"}
        trace.append(("manifest", kwargs["member_id"], kwargs["actual_inputs"]))
    @contextmanager
    def scope(*, member_id, device_id):
        assert device_id == 0 and current_run_control() is control
        trace.append(("open", member_id))
        with member_cfl_scope():
            try:
                yield SimpleNamespace(stream=None)
            finally:
                trace.append(("close", member_id))
    def runner(inputs, *, output_directory, schedule_dispatch, progress_callback, first_products):
        member = current_capture().member_id
        assert first_products is None and output_directory.is_dir()
        assert current_run_control() is control
        assert inputs is shared if not source else inputs in configured
        trace.append(("runner", member, inputs))
        exp, model = _model()
        current_capture().initialize_callback(model=model)
        def step(state, cfg, **unused):
            if callback_error is not None and member == 0:
                raise callback_error
            if stop and member == 0:
                control.request_stop("fixture stop at original member boundary")
            if cfg.grid_id == 1:
                progress_callback(model_elapsed_seconds=state.elapsed_seconds,
                                  outer_step=model.root.clock.step_count)
        def history(tree, node, ticks):
            current_capture().callback(member_id=member, grid_id=node.cfg.grid_id, ticks=ticks)
        try:
            execute_experiment(model, validate_state=False, experiment=exp,
                steppers={1: step, 2: step}, history_handler=history,
                pool_trim_per_period=False, schedule_dispatch=schedule_dispatch)
            report = returned[member] = OriginalReport("PASS", member, inputs.tag)
            return report
        finally:
            trace.append(("writer-close", member))
    progress = EnsembleProgressAdapter(None, member_ids=ids, run_seconds=shared.experiment.run_seconds, control=control)
    plan = BatchMemoryPlan((BatchArraySpec("unavailable fixture component", (100000,), "member"),), 0)
    reservation = RouteWaveReservation({member: 4096 for member in ids}, {"fixture": plan},
        {"collector": 128, "stochastic": 0, "cuda_owners": 128, "allocator_margin": 128},
        {"ordinary": "complete CPU fixture envelopes", "native": "large declared component fallback",
         "collector": "bounded CPU collector metadata", "stochastic": "inactive fixture",
         "cuda_owners": "CPU fixture owners", "allocator_margin": "explicit fixture margin"}, 20000)
    kwargs = dict(member_inputs={member: shared for member in ids}, output_directory=tmp_path,
        reservation=reservation, progress_adapter=progress, control=control, member_scope=scope,
        pack_factory=object(), operation_authority=lambda **kwargs: {"member": kwargs["member_id"]},
        source_context_callback=source_context if source else None)
    return session, runner, kwargs, trace, frames, initialized, returned, progress


@pytest.mark.parametrize("source", [False, True])
def test_original_reports_captures_progress_and_source_callback_survive_joined_route(tmp_path, source):
    session, runner, kwargs, trace, frames, initialized, returned, progress = setup(tmp_path, source=source)
    result = run_production_component_wave(session, runner, **kwargs)
    assert result.member_results == returned
    assert all(result.member_results[member] is returned[member] for member in returned)
    assert set(result.captures) == set(result.output_directories) == {0, 1}
    assert {member for member, _ in initialized} == {0, 1} and len(initialized) == 2
    assert {frame["member_id"] for frame in frames} == {0, 1}
    assert result.execution["execution"]["packed_components"] is False
    assert "native component wave needs" in result.execution["execution"]["fallback_reason"]
    assert result.execution["route_finalization"] == "original_runner"
    for member in (0, 1):
        assert trace.index(("writer-close", member)) < trace.index(("close", member))
        assert result.captures[member].member_id == member
    assert progress.receipt()["members"]
    if source:
        source_rows = [row for row in trace if row[0] == "source"]
        assert len(source_rows) == 2 and all(row[1] == row[2] and callable(row[3]) for row in source_rows)
        for member in (0, 1):
            assert next(i for i, row in enumerate(trace) if row[:2] == ("manifest", member)) < next(
                i for i, row in enumerate(trace) if row[:2] == ("runner", member))


def test_global_member_ids_and_original_observer_binding_are_preserved(tmp_path):
    session, runner, kwargs, trace, frames, initialized, returned, progress = setup(
        tmp_path, source=True, ids=(5, 2))
    def observer_runner(inputs, *, observer, **options):
        return runner(inputs, progress_callback=observer, **options)
    kwargs["progress_key"] = "observer"
    result = run_production_component_wave(session, observer_runner, **kwargs)
    assert result.execution["member_order"] == [5, 2]
    assert set(result.member_results) == set(result.captures) == {5, 2}
    assert {frame["member_id"] for frame in frames} == {5, 2}
    assert {member for member, _ in initialized} == {5, 2}
    assert result.output_directories[5].name == "member-0005"
    assert progress.receipt()["member_order"] == [5, 2]


def test_source_manifest_failure_wakes_partial_capture_and_keeps_original_exception(tmp_path):
    session, runner, kwargs, trace, *_ = setup(tmp_path, source=True)
    original_error = ValueError("original source manifest authority failure")
    previous = kwargs["source_context_callback"]
    def context(**values):
        if values["member_id"] == 0:
            raise original_error
        previous(**values)
    kwargs["source_context_callback"] = context
    with pytest.raises(ValueError) as caught:
        run_production_component_wave(session, runner, **kwargs)
    assert caught.value is original_error and caught.value.ensemble_member_ids == (0,)
    assert not any(row[:2] == ("runner", 0) for row in trace)
    assert {row[1] for row in trace if row[0] == "open"} == {row[1] for row in trace if row[0] == "close"}


def test_existing_stop_precedes_directories_or_initialization(tmp_path):
    session, runner, kwargs, trace, _, initialized, *_ = setup(tmp_path)
    kwargs["control"].request_stop("already stopped")
    with pytest.raises(MemberStopRequested, match="already stopped"):
        run_production_component_wave(session, runner, **kwargs)
    assert not trace and not initialized and not (tmp_path / "members").exists()


def test_real_session_array_module_attribute_and_missing_qualification_keep_original_wave(tmp_path):
    session, runner, kwargs, *_ = setup(tmp_path)
    assert not hasattr(session, "xp")
    kwargs["pack_factory"] = None
    result = run_production_component_wave(session, runner, **kwargs)
    assert result.execution["execution"]["packed_components"] is False
    assert "no qualified component factory was supplied" in result.execution["execution"]["fallback_reason"]


def test_absent_session_array_module_imports_the_original_cupy_boundary(tmp_path, monkeypatch):
    import sys
    session, runner, kwargs, *_ = setup(tmp_path)
    session.array_module = None
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    kwargs["pack_factory"] = None
    result = run_production_component_wave(session, runner, **kwargs)
    assert result.execution["status"] == "PASS"
    assert result.execution["execution"]["packed_components"] is False


@pytest.mark.parametrize("failure", ["callback", "stop"])
def test_error_and_stop_join_every_started_original_owner_without_retry(tmp_path, failure):
    original_error = RuntimeError("original STEP failure")
    session, runner, kwargs, trace, *_ = setup(tmp_path,
        callback_error=original_error if failure == "callback" else None, stop=failure == "stop")
    expected = RuntimeError if failure == "callback" else MemberStopRequested
    with pytest.raises(expected) as caught:
        run_production_component_wave(session, runner, **kwargs)
    if failure == "callback":
        assert caught.value is original_error
        assert caught.value.ensemble_member_ids == (0,)
    assert len([row for row in trace if row[0] == "runner"]) <= 2
    assert {row[1] for row in trace if row[0] == "open"} == {row[1] for row in trace if row[0] == "close"}


def test_existing_member_directory_refuses_before_callbacks_and_preserves_contents(tmp_path):
    session, runner, kwargs, trace, *_ = setup(tmp_path)
    directory = tmp_path / "members/member-0001"
    directory.mkdir(parents=True)
    sentinel = directory / "original.txt"
    sentinel.write_text("existing member output")
    with pytest.raises(FileExistsError, match="fresh"):
        run_production_component_wave(session, runner, **kwargs)
    assert not trace and sentinel.read_text() == "existing member output"
    assert not (tmp_path / "members/member-0000").exists()


def test_ordinary_memory_refusal_precedes_directory_or_member_scope(tmp_path):
    from dataclasses import replace
    session, runner, kwargs, trace, *_ = setup(tmp_path)
    kwargs["reservation"] = replace(kwargs["reservation"], available_bytes=1)
    with pytest.raises(MemoryError, match="ordinary route wave"):
        run_production_component_wave(session, runner, **kwargs)
    assert not trace and not (tmp_path / "members").exists()


def test_other_physical_source_owner_keeps_ordinary_route_before_side_effects(tmp_path):
    session, runner, kwargs, trace, *_ = setup(tmp_path)
    session.source_execution = SimpleNamespace(run_member=lambda **kwargs: None)
    assert "prepared fields" in source_component_route_reason(session.source_execution)
    with pytest.raises(BatchStateUnsupported, match="priced route handoff"):
        run_production_component_wave(session, runner, **kwargs)
    assert not trace and not (tmp_path / "members").exists()


def test_ordinary_source_requires_its_durable_manifest_callback(tmp_path):
    session, runner, kwargs, trace, *_ = setup(tmp_path, source=True)
    kwargs["source_context_callback"] = None
    with pytest.raises(BatchStateUnsupported, match="durable source/stochastic manifest"):
        run_production_component_wave(session, runner, **kwargs)
    assert not trace and not (tmp_path / "members").exists()
