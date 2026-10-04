"""Run control of an ensemble: stopping it, naming a failed member, and progress.

Every test here drives the real wave executor (``execute_member_packing``)
or the real session around it. A member is a function that reports each
step through the session's own progress adapter, which is the seam a real
forecast runner reports through.
"""
from contextlib import nullcontext
from datetime import datetime
import io
import json
import os
import signal
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
from gpuwm.ensemble.execution import (
    MemberRunControl, MemberStopRequested, execute_member_packing, failed_member_rows,
    name_failing_members,
)
from gpuwm.ensemble.packing import CardBudget, pack_members
from gpuwm.ensemble.production import PreparedEnsembleSession
from gpuwm.ensemble.progress import (
    EnsembleProgressAdapter, MemberTerminalProgress, progress_host,
)
from gpuwm.ensemble.runtime_context import current_capture

#: One member's whole forecast, in wall seconds, when nothing stops it.
MEMBER_SECONDS = 6.0
#: When the interrupt is sent.
INTERRUPT_AFTER = 0.5
#: "Well before the member's full duration": the stop must be back by here.
STOPPED_BY = MEMBER_SECONDS / 2


def _model():
    return EnsembleMemoryModel((MemoryComponent("ordinary", "ordinary", fixed_bytes=100),))


def _plan(members, cards=1):
    return pack_members(members, tuple(CardBudget(device, 1000) for device in range(cards)),
                        _model(), batched=False, reason="one ordinary member per card")


@pytest.fixture
def live_sigint():
    """Ctrl-C as a terminal delivers it, whatever shell started this run.

    A test runner started in the background inherits SIGINT ignored, and
    Python then installs no handler: the interrupt this file sends would do
    nothing and the tests would wait out the full member duration.
    """
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("an interrupt is delivered to the main thread, and this runner executes tests on another")
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous if previous is not None else signal.SIG_DFL)


def _interrupt_soon(after=INTERRUPT_AFTER):
    """Send this process a SIGINT from another thread after ``after`` seconds."""
    def send():
        if sys.platform == "win32":
            import _thread
            _thread.interrupt_main()
        else:
            os.kill(os.getpid(), signal.SIGINT)
    timer = threading.Timer(after, send)
    timer.daemon = True
    timer.start()
    return timer


def _stepping_member(adapter, member, record, *, seconds=MEMBER_SECONDS, step=0.02, run_seconds=60.0):
    """A member that reports every step, as the forecast runners do."""
    callback = adapter.callback_for_member(member)
    steps = int(seconds / step)
    record[member] = {"started": time.monotonic(), "steps": 0, "ended": None}
    try:
        for index in range(1, steps + 1):
            time.sleep(step)
            callback(model_elapsed_seconds=run_seconds * index / steps, outer_step=index)
            record[member]["steps"] = index
    finally:
        record[member]["ended"] = time.monotonic()
    return {"status": "PASS", "steps": steps}


# -- finding 1: Ctrl-C stops a running ensemble --------------------------------


def test_interrupt_stops_the_running_member_and_cancels_the_queued_one(live_sigint):
    plan = _plan(2)                                   # one card: two waves of one member
    assert plan.waves == 2
    control = MemberRunControl()
    adapter = EnsembleProgressAdapter(None, member_ids=(0, 1), run_seconds=60.0, control=control)
    record = {}
    timer = _interrupt_soon()
    began = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            execute_member_packing(plan, lambda batch: _stepping_member(
                adapter, batch.member_indices[0], record), control=control)
    finally:
        timer.cancel()
    elapsed = time.monotonic() - began
    assert not isinstance(caught.value, MemberStopRequested), "the user's own interrupt is what surfaces"
    assert elapsed < STOPPED_BY, f"the interrupt was held for {elapsed:.2f} s of a {MEMBER_SECONDS} s member"
    assert control.stop_requested
    # The running member ended at a step boundary and was joined before the raise.
    assert record[0]["ended"] is not None
    assert 0 < record[0]["steps"] < int(MEMBER_SECONDS / 0.02) // 2
    # The second wave never started.
    assert 1 not in record


def test_interrupt_stops_every_card_and_joins_before_it_returns(live_sigint):
    plan = _plan(2, cards=2)                          # two cards: one wave, both running
    assert plan.waves == 1
    control = MemberRunControl()
    adapter = EnsembleProgressAdapter(None, member_ids=(0, 1), run_seconds=60.0, control=control)
    record = {}
    timer = _interrupt_soon()
    began = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            execute_member_packing(plan, lambda batch: _stepping_member(
                adapter, batch.member_indices[0], record), control=control)
    finally:
        timer.cancel()
    assert time.monotonic() - began < STOPPED_BY
    assert sorted(record) == [0, 1]
    assert all(row["ended"] is not None for row in record.values()), "both members were joined"
    assert all(0 < row["steps"] < int(MEMBER_SECONDS / 0.02) // 2 for row in record.values())


@pytest.mark.parametrize("stop", [SystemExit(143), KeyboardInterrupt()])
def test_a_stop_raised_by_one_member_stops_the_others_and_surfaces_unchanged(stop):
    plan = _plan(2, cards=2)
    control = MemberRunControl()
    adapter = EnsembleProgressAdapter(None, member_ids=(0, 1), run_seconds=60.0, control=control)
    record = {}

    def member(batch):
        if batch.member_indices[0] == 0:
            time.sleep(0.2)
            raise stop
        return _stepping_member(adapter, 1, record)

    began = time.monotonic()
    with pytest.raises(type(stop)) as caught:
        execute_member_packing(plan, member, control=control)
    assert caught.value is stop
    assert time.monotonic() - began < STOPPED_BY
    assert record[1]["ended"] is not None and record[1]["steps"] < int(MEMBER_SECONDS / 0.02) // 2


def test_a_stopped_run_starts_no_further_wave():
    control = MemberRunControl()
    control.request_stop("stopped before the wave")
    started = []
    with pytest.raises(MemberStopRequested):
        execute_member_packing(_plan(2), lambda batch: started.append(batch), control=control)
    assert not started


def test_members_stop_at_any_reported_step_once_the_run_is_stopped():
    control = MemberRunControl()
    adapter = EnsembleProgressAdapter(None, member_ids=(4, 9), run_seconds=24.0, control=control)
    callback = adapter.callback_for_member(4)
    callback(model_elapsed_seconds=12.0, outer_step=1)
    control.request_stop("KeyboardInterrupt during the ensemble forecast")
    with pytest.raises(MemberStopRequested, match="KeyboardInterrupt"):
        callback(model_elapsed_seconds=24.0, outer_step=2)
    # A native pack reports through one member's callback for all of its members.
    with pytest.raises(MemberStopRequested):
        adapter.callback_for_member(9)(member_ids=(4, 9), model_elapsed_seconds=24.0, outer_step=2)
    assert issubclass(MemberStopRequested, KeyboardInterrupt)
    assert adapter.receipt()["members"][0]["outer_step"] == 1, "a stopped step is not recorded"


# -- the session around the waves ---------------------------------------------


class Collector:
    def __init__(self):
        self.rows = []
        self.finished = False

    def submit(self, **row):
        self.rows.append(row)

    def finish_run(self):
        self.finished = True
        return {"frames": len(self.rows)}

    def require_complete(self):
        return {}


def _inputs(run_seconds=60):
    cfg = SimpleNamespace(dt=3, use_adaptive_time_step=True, mp_physics=16)
    exp = SimpleNamespace(run_seconds=run_seconds, start_time=datetime(2024, 1, 1),
                          root=SimpleNamespace(run=cfg))
    return SimpleNamespace(experiment=exp, boundary_interval_seconds=3600)


def _session(tmp_path, *, members=3, cards=1, **kwargs):
    # Every member is handed its inputs by a member source, as the doors that
    # run more than one member do: a roster when the test names one, and a
    # provider otherwise.
    if "member_roster" not in kwargs:
        kwargs.setdefault("input_provider",
                          lambda *, shared_inputs, member_id, request: shared_inputs)
    return PreparedEnsembleSession({"members": members}, output_directory=tmp_path,
        cards=tuple(CardBudget(device, 1000) for device in range(cards)),
        memory_model=_model(), device_scope=lambda _: nullcontext(), **kwargs)


def test_session_interrupt_ends_as_interrupted_with_no_aggregate_products(tmp_path, live_sigint, capsys):
    collector, record = Collector(), {}

    def runner(member_inputs, *, output_directory, first_products, progress_callback=None, **kw):
        member = current_capture().member_id
        steps = int(MEMBER_SECONDS / 0.02)
        record[member] = 0
        for index in range(1, steps + 1):
            time.sleep(0.02)
            progress_callback(model_elapsed_seconds=60.0 * index / steps, outer_step=index)
            record[member] = index
        return {"status": "PASS"}

    timer = _interrupt_soon()
    began = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            _session(tmp_path, collector=collector).run_prepared(runner, _inputs())
    finally:
        timer.cancel()
    assert time.monotonic() - began < STOPPED_BY
    manifest = json.loads((tmp_path / "ensemble-run.json").read_text())
    assert manifest["status"] == "interrupted" and manifest["error_type"] == "KeyboardInterrupt"
    assert manifest["members_completed"] == [] and manifest["members_not_completed"] == [0, 1, 2]
    assert manifest["failed_members"] == []
    assert sorted(record) == [0], "members not yet started were cancelled"
    assert not collector.finished, "an interrupted run never closes its aggregate products"
    assert not (tmp_path / "report.json").exists()
    assert "stopping: 1 running member batch ends at the next model step" in capsys.readouterr().err
    assert current_capture() is None


# -- finding 4: a failed member is named ---------------------------------------


def test_wave_reports_every_error_and_names_each_member():
    plan = _plan(2, cards=2)

    def member(batch):
        raise RuntimeError(f"fixture failure on card {batch.device_id}")

    with pytest.raises(RuntimeError) as caught:
        execute_member_packing(plan, member)
    error = caught.value
    assert str(error) == "member 0: fixture failure on card 0"
    assert error.ensemble_member_ids == (0,) and error.ensemble_device_id == 0
    assert [str(other) for other in error.ensemble_wave_errors] == [
        "member 0: fixture failure on card 0", "member 1: fixture failure on card 1"]
    assert any("also failed in this wave" in note and "member 1" in note for note in error.__notes__)
    rows = failed_member_rows(error)
    assert [(row["member_ids"], row["device_id"], row["error_type"]) for row in rows] == [
        ([0], 0, "RuntimeError"), ([1], 1, "RuntimeError")]


def test_naming_keeps_the_error_type_and_the_first_attribution():
    class Refused(MemoryError):
        pass
    error = name_failing_members(Refused("does not fit"), (7,), device_id=1, wave=0)
    assert type(error) is Refused and str(error) == "member 7: does not fit"
    # A pack-level attribution does not overwrite the member the gate named.
    again = name_failing_members(error, (5, 6, 7), device_id=1, wave=0)
    assert again is error and error.ensemble_member_ids == (7,)
    assert str(error) == "member 7: does not fit"
    # An error whose text is not its plain first argument keeps its own text.
    missing = name_failing_members(KeyError("smois"), (3,))
    assert missing.ensemble_member_ids == (3,) and str(missing) == "'smois'"
    assert any("ensemble member 3" in note for note in missing.__notes__)


def test_failed_member_is_named_in_the_error_and_the_manifest(tmp_path):
    shared, collector = _inputs(), Collector()
    roster_ids = (19, 3, 44)
    member_inputs = {member: SimpleNamespace(experiment=shared.experiment) for member in roster_ids}
    roster = SimpleNamespace(members=tuple(SimpleNamespace(member_id=member) for member in roster_ids),
        receipts=(), select=lambda ids: tuple(SimpleNamespace(inputs=member_inputs[member]) for member in ids),
        receipt=lambda: {"member_order": list(roster_ids)})

    def runner(prepared, **kw):
        member = current_capture().member_id
        if member == 3:
            raise FloatingPointError("full-state health gate failed during post-d01-sync.d01: qv(3, 10, 12): non-finite")
        return {"status": "PASS"}

    with pytest.raises(FloatingPointError) as caught:
        _session(tmp_path, collector=collector, member_roster=roster).run_prepared(runner, shared)
    assert str(caught.value).startswith("member 3: full-state health gate failed")
    manifest = json.loads((tmp_path / "ensemble-run.json").read_text())
    assert manifest["status"] == "failed" and manifest["error"].startswith("member 3: ")
    assert manifest["members_completed"] == [19]
    assert manifest["members_not_completed"] == [3, 44]
    (failed,) = manifest["failed_members"]
    assert failed["member_ids"] == [3] and failed["device_id"] == 0 and failed["wave"] == 1
    assert failed["error_type"] == "FloatingPointError" and failed["execution_mode"] == "ordinary_member"
    # The run fails whole: aggregate products are never closed as complete.
    assert not collector.finished
    assert not (tmp_path / "report.json").exists() and not (tmp_path / "progress.json").exists()


def test_two_card_wave_records_both_failed_members_and_publishes_nothing(tmp_path):
    collector = Collector()

    def runner(prepared, **kw):
        raise RuntimeError(f"fixture failure of member {current_capture().member_id}")

    with pytest.raises(RuntimeError, match="member 0: fixture failure of member 0"):
        _session(tmp_path, members=2, cards=2, collector=collector).run_prepared(runner, _inputs())
    manifest = json.loads((tmp_path / "ensemble-run.json").read_text())
    assert [row["member_ids"] for row in manifest["failed_members"]] == [[0], [1]]
    assert {row["device_id"] for row in manifest["failed_members"]} == {0, 1}
    assert manifest["members_completed"] == [] and not collector.finished


def test_batch_health_gate_names_the_member_of_the_pack():
    """A pack shares one step count, so only the gate can say which member."""
    from gpuwm.core.health import HealthCheckError, ValidationReport
    from gpuwm.ensemble.batch_health import PreparedBatchStateHealth
    gate = PreparedBatchStateHealth.__new__(PreparedBatchStateHealth)
    gate.member_ids = (10, 4, 2, 9)
    bad = ValidationReport(False, 1, "qv", (3, 10, 12), 0, float("nan"), "non-finite", "post-d01-sync.d01")
    reports = (ValidationReport(True, 0), ValidationReport(True, 0), bad, bad)
    gate.validate = lambda *, phase=None: reports
    with pytest.raises(HealthCheckError) as caught:
        gate.require_healthy(phase="post-d01-sync.d01")
    text = str(caught.value)
    assert text.startswith("member 2: full-state health gate failed during post-d01-sync.d01: qv(3, 10, 12)")
    assert text.endswith("also failing: member 9")
    assert caught.value.ensemble_member_ids == (2,)
    assert caught.value.ensemble_failing_member_ids == (2, 9)
    assert caught.value.report is bad
    # Healthy packs return their reports unchanged.
    gate.validate = lambda *, phase=None: reports[:2]
    gate.member_ids = (10, 4)
    assert gate.require_healthy(phase="final.d01") == reports[:2]


# -- finding 3: the observer handoff, and progress on the terminal -------------


def test_a_stage_observer_is_never_the_runner_progress_callback():
    from gpuwm.chain_events import GoChainEvents
    chain = GoChainEvents()
    assert progress_host(chain) is None
    host = lambda **event: None
    assert progress_host(host) is host and progress_host(None) is None

    class CallableTelemetry:
        hosts_forecast = False
        def __call__(self, **event):
            raise AssertionError("a non-host was used as the progress callback")
    assert progress_host(CallableTelemetry()) is None
    adapter = EnsembleProgressAdapter(chain, member_ids=(0, 1), run_seconds=24.0)
    assert adapter.callback is None
    # The call that ended every `gpuwm go --members` run at its first step.
    adapter.callback_for_member(0)(model_elapsed_seconds=0.0, outer_step=0, phase="stepping:outer-1")
    adapter.callback_for_member(0)(model_elapsed_seconds=12.0, outer_step=1)
    assert adapter.receipt()["members"][0]["outer_step"] == 1
    # Optional hooks a runner looks up on its observer stay absent, not broken.
    assert getattr(adapter.callback_for_member(0), "preparing", None) is None


def test_session_survives_the_go_stage_observer_and_says_member_progress(tmp_path, capsys):
    from gpuwm.chain_events import GoChainEvents

    def runner(member_inputs, *, output_directory, first_products, observer=None, **kw):
        observer(model_elapsed_seconds=0.0, outer_step=0, phase="stepping:outer-1")
        observer(model_elapsed_seconds=30.0, outer_step=1)
        observer(model_elapsed_seconds=60.0, outer_step=2)
        return {"status": "PASS"}

    result = _session(tmp_path, members=2, collector=Collector()).run_prepared(
        runner, _inputs(), observer=GoChainEvents())
    assert result["status"] == "PASS" and result["members_completed"] == [0, 1]
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("ensemble: ")]
    assert lines[0] == "ensemble: member 0 started (1 of 2 members started)"
    assert "ensemble: member 0 finished 60 model seconds in" in lines[1] and lines[1].endswith("(1 of 2 done)")
    assert lines[2] == "ensemble: member 1 started (2 of 2 members started)"
    assert lines[3].endswith("(2 of 2 done)")


def test_a_hosting_observer_still_receives_every_event_and_no_terminal_lines(tmp_path, capsys):
    events = []

    def runner(member_inputs, *, output_directory, first_products, observer=None, **kw):
        observer(model_elapsed_seconds=60.0, outer_step=5)
        return {"status": "PASS"}

    host = lambda **event: events.append(event)
    _session(tmp_path, members=2, collector=Collector()).run_prepared(runner, _inputs(), observer=host)
    assert [event["ensemble_member_ids"] for event in events] == [(0,), (1,)]
    assert [event["model_elapsed_seconds"] for event in events] == [30.0, 60.0]
    assert "ensemble: member" not in capsys.readouterr().err


def test_terminal_progress_is_one_line_per_interval_and_names_native_packs():
    now = [100.0]
    stream = io.StringIO()
    terminal = MemberTerminalProgress((0, 1, 2, 3), 3600.0, stream=stream,
                                      interval_seconds=20.0, clock=lambda: now[0])
    adapter = EnsembleProgressAdapter(None, member_ids=(0, 1, 2, 3), run_seconds=3600.0, terminal=terminal)
    pack = adapter.callback_for_member(0)
    for step in range(1, 61):
        now[0] += 1.0
        pack(member_ids=(0, 1), model_elapsed_seconds=30.0 * step, outer_step=step)
    now[0] += 1.0
    pack(member_ids=(0, 1), model_elapsed_seconds=3600.0, outer_step=120)
    lines = stream.getvalue().splitlines()
    assert lines[0] == "ensemble: members 0, 1 started (2 of 4 members started)"
    progress = [line for line in lines if " at " in line]
    assert len(progress) == 2, "sixty steps in sixty seconds are two lines, not sixty"
    assert progress[0] == "ensemble: members 0, 1 at 630 of 3600 model seconds, step 21 (0 of 4 done)"
    assert lines[-1] == "ensemble: members 0, 1 finished 3600 model seconds in 60 s (2 of 4 done)"
    assert len(lines) == 4


def _forecast_plan(tmp_path):
    return ({"runner": "gpuwm.prepared_single_domain_forecast", "source": "gfs",
             "prepared": tmp_path / "prep", "authority": tmp_path / "auth",
             "run": tmp_path / "run", "profile": None},
            {"proof": "a", "source_manifest": "b", "prepared_content": "c"})


def test_go_gives_the_hosted_runner_no_stage_observer_under_an_ensemble_session(monkeypatch, tmp_path):
    """The crash class: go's stage observer handed to the member runners."""
    import gpuwm.go_cli as go_cli
    from gpuwm.chain_events import GoChainEvents
    from gpuwm.ensemble.runtime_context import ensemble_scope

    plan, digests = _forecast_plan(tmp_path)
    seen, heard = {}, []

    class Runner:
        @staticmethod
        def main(argv, *, observer):
            seen["observer"] = observer
            return 0

    class Stages(GoChainEvents):
        def stage_begin(self, *, label, command):
            heard.append(("begin", label))
        def stage_end(self, *, label, exit_code, ok, **kw):
            heard.append(("end", label, ok))

    monkeypatch.setattr("importlib.import_module", lambda name: Runner)
    monkeypatch.setattr(go_cli, "_run_stage", lambda *args, **kw: pytest.fail(
        "an ensemble session hosts its members in this process"))
    with ensemble_scope(SimpleNamespace(request=None)):
        go_cli._run_forecast(plan, digests, explain=False, observer=Stages())
        assert seen == {"observer": None}
        assert heard == [("begin", "forecast"), ("end", "forecast", True)]
        # No observer at all (the recipe door from the command line) is the same route.
        go_cli._run_forecast(plan, digests, explain=False, observer=None)
        assert seen == {"observer": None}
        # A hosting observer (gpuwm run-plan) still receives the runner's progress.
        host = lambda **event: None
        go_cli._run_forecast(plan, digests, explain=False, observer=host)
        assert seen["observer"] is host


def test_hosted_forecast_interrupt_is_go_s_own_interrupt_result(monkeypatch, tmp_path):
    import gpuwm.go_cli as go_cli
    from gpuwm.chain_events import GoChainEvents
    from gpuwm.ensemble.runtime_context import ensemble_scope

    plan, digests = _forecast_plan(tmp_path)

    class Runner:
        @staticmethod
        def main(argv, *, observer):
            raise KeyboardInterrupt

    monkeypatch.setattr("importlib.import_module", lambda name: Runner)
    with ensemble_scope(SimpleNamespace(request=None)):
        with pytest.raises(go_cli.GoInterrupted) as caught:
            go_cli._run_forecast(plan, digests, explain=False, observer=GoChainEvents())
        assert caught.value.label == "forecast" and caught.value.pid is None
        assert caught.value.exit_code == 130 and caught.value.hosted
        # A host owns its own stop: it sees the interrupt itself.
        with pytest.raises(KeyboardInterrupt):
            go_cli._run_forecast(plan, digests, explain=False, observer=lambda **event: None)
    # One sentence either way. The reason behind --explain is the hosted
    # stage's own: there was no stage subprocess for the terminal to signal.
    from gpuwm.explain import render
    report = go_cli._interrupt_report(caught.value, {"root": tmp_path / "run-1", "config": tmp_path / "case.toml"})
    said = render(report, explain=False, command="gpuwm go")
    assert said.startswith(f"go: interrupted during forecast; no later stage ran and {tmp_path / 'run-1'} ")
    assert "process was pid" not in said and "stage subprocess" not in said
    explained = render(report, explain=True, command="gpuwm go")
    assert "ended at their next model step" in explained and "members not yet started were cancelled" in explained
    assert "the stage subprocess received it directly" not in explained
    # A stage subprocess keeps its own reason and names its pid.
    child = render(go_cli._interrupt_report(go_cli.GoInterrupted("fetch", 4242),
                                            {"root": tmp_path, "config": tmp_path / "case.toml"}),
                   explain=True, command="gpuwm go")
    assert "the fetch process was pid 4242" in child and "the stage subprocess received it directly" in child
    assert "ended at their next model step" not in child


# -- finding 6: the closing line names where the pictures are ------------------


def test_go_names_the_ensemble_maps_folder_not_a_png_folder_that_was_never_made(tmp_path):
    import gpuwm.go_cli as go_cli
    from gpuwm.ensemble.runtime_context import ensemble_scope

    plan = {"run": tmp_path / "run", "render": tmp_path / "png"}
    assert go_cli._rendered_root(plan) == tmp_path / "png"
    session = SimpleNamespace(request=None, last_output_directory=tmp_path / "run")
    with ensemble_scope(session):
        assert go_cli._rendered_root(plan) is None, "no pictures, so no path is printed"
        (tmp_path / "run" / "maps" / "d01").mkdir(parents=True)
        assert go_cli._rendered_root(plan) == tmp_path / "run" / "maps"
        assert not (tmp_path / "png").exists()


# -- finding 5: the ensemble alias in the command-keyed tables -----------------


def test_the_ensemble_alias_has_go_s_preflight_and_interrupt_notice(monkeypatch, capsys, tmp_path):
    from gpuwm import capabilities, cli

    args = cli.build_parser().parse_args(["ensemble", "case.toml", "--members", "4"])
    assert args.command == "ensemble"
    assert capabilities.COMMAND_REQUIREMENTS["ensemble"] == capabilities.COMMAND_REQUIREMENTS["go"]
    assert "ensemble" in cli._LONG_RUNNING_COMMANDS

    monkeypatch.setattr(signal, "getsignal", lambda number: signal.SIG_IGN)
    cli._warn_if_interrupt_is_ignored("ensemble")
    notice = capsys.readouterr().err
    assert notice.count("warning:") == 1 and "gpuwm ensemble" in notice and "SIGTERM" in notice
    monkeypatch.undo()

    # The front door refuses an install with no GPU runtime before any member
    # is fetched or prepared, in the reader's own command.
    monkeypatch.setattr(capabilities, "is_installed", lambda module: module != "cupy")
    with pytest.raises(capabilities.CapabilityMissing) as caught:
        capabilities.require_for_command("ensemble")
    assert str(caught.value).startswith("gpuwm ensemble:")
    assert "before any member's forcing data is downloaded or prepared" in str(caught.value)
    code = cli.main(["ensemble", str(tmp_path / "not-here.toml"), "--recipe", "time-lagged",
                     "--members", "2", "--outdir", str(tmp_path / "out")])
    assert code == 2
    text = capsys.readouterr().err
    assert "gpuwm ensemble: this command needs cupy" in text and "Traceback" not in text
    assert not (tmp_path / "out").exists()


# -- finding 7: native runs say what they advanced, and an audit refusal declines ----


def test_native_handoff_closes_the_step_log_with_the_steps_the_packs_advanced(tmp_path):
    """A native run used to close its log with "SUCCESS COMPLETE SIMULATION, 0 steps"."""
    from gpuwm.ensemble.runtime_context import initialized_bootstrap_handoff
    from gpuwm.progress_log import open_step_log

    text = io.StringIO()
    step_log = open_step_log(outdir=tmp_path, start_time=datetime(2024, 5, 25, 18),
                             run_seconds=24.0, text_stream=text)

    def native(*, inputs, model, node, output_directory, observer, step_observer):
        for step in (1, 2):
            step_observer(grid_id=1, step_count=step, model_seconds=12.0 * step,
                          step_wall_seconds=0.25, dt=12.0)
        pack = {"status": "PASS", "executor": {"steps": 2, "member_steps": 8}}
        return {"status": "PASS", "wall_seconds": 3.5, "members_completed": [0, 1, 2, 3],
                "member_results": [{"batch": {}, "global_member_ids": [0, 1, 2, 3], "result": pack}]}

    report = initialized_bootstrap_handoff(native, inputs=None, model=None, node=None,
        output_directory=tmp_path, observer=None, step_log=step_log)
    assert report["status"] == "PASS"
    lines = text.getvalue().splitlines()
    assert any(line.startswith("Timing for main: time 2024-05-25_18:00:24 on domain   1:")
               and line.endswith("step 2") for line in lines)
    assert any(line.startswith("gpuwm: phase native ensemble forecast, 4 members, 2 steps each:")
               for line in lines)
    assert lines[-1].startswith("gpuwm: SUCCESS COMPLETE SIMULATION, 2 steps, ")
    records = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    (pack,) = [row for row in records if row["event"] == "phase"]
    assert (pack["members"], pack["steps_per_member"], pack["member_steps"], pack["packs"]) == (4, 2, 8, 1)
    assert records[-1]["event"] == "run_end" and records[-1]["steps"] == 2

    # A bootstrap that takes no step (one extra card's own initialization) still says zero.
    quiet = io.StringIO()
    other = open_step_log(outdir=tmp_path / "card", start_time=datetime(2024, 5, 25, 18),
                          run_seconds=24.0, text_stream=quiet)
    initialized_bootstrap_handoff(lambda **kw: {"status": "PASS", "forecast_steps": 0},
        inputs=None, model=None, node=None, output_directory=tmp_path, observer=None, step_log=other)
    assert "native ensemble forecast" not in quiet.getvalue()
