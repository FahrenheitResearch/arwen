"""The doors of an as-posted run (A136 L3 (i)): go, source_cli and the forecast.

``gpuwm go`` on a source whose preparation waits on posted leads runs its
fetch beside the preparation (DESIGN A136 2.5): the preparation starts
with ``--as-posted`` once the fetch has scheduled the window, there is no
manifest stage (the seal writes the manifest), and a fetch that ends
without a marker for every lead tells the preparation so.  ``source_cli``
forwards ``--as-posted``.  The single-domain forecast binds an as-posted
head by its input plan and accepts its seal on the terms of the L3 design
ruling.  The schedule the fetch keeps replacing is read through a replace.

CPU only; no device, no source data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading

import pytest

from gpuwm import go_cli, source_cli
from gpuwm.ingest import boundary_stream
from gpuwm.ingest.boundary_stream import (
    BoundaryProducerFailed, PostedLeads, read_replaced_json,
)

from test_posted_preparation import _as_posted_tree


# ---------------------------------------------------------------------------
# gpuwm go: the fetch beside the preparation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gfs_config(tmp_path_factory):
    from test_go_chain import _emit

    return _emit(tmp_path_factory.mktemp("posted-go"), "posted")


def _args(config, outdir):
    from test_go_chain import _args as chain_args

    return chain_args(config, outdir)


def _quiet_gates(monkeypatch):
    monkeypatch.setattr(go_cli, "memory_gate", lambda plan, **kw: {
        "verdict": "fits", "refuse": False, "warn": False,
        "free_bytes": 30 * 1024 ** 3})
    monkeypatch.setattr(go_cli, "_require_forecast_device", lambda: None)
    monkeypatch.setattr(go_cli.capabilities, "require_for_command",
                        lambda command: None)
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))


def _post_window(command, *, leads=(0, 1), markers=True):
    """What the fetch stage leaves as posted: its schedule and lead markers."""

    out = Path(command[command.index("--out") + 1])
    cycle = command[command.index("--cycle") + 1]
    posting = out / "posting"
    posting.mkdir(parents=True, exist_ok=True)
    (posting / "schedule.json").write_text(json.dumps({
        "schema": "gpuwm.posting-schedule.v1", "source": "gfs",
        "cycle": cycle, "table_sha256": "7" * 64,
        "leads": [{"lead": lead} for lead in leads]}), encoding="utf-8")
    if markers:
        for lead in leads:
            (posting / f"f{lead:03d}.json").write_text(json.dumps({
                "schema": "gpuwm.posted-lead.v1", "source": "gfs",
                "cycle": cycle, "lead": lead, "objects": []}),
                encoding="utf-8")
    return posting


def test_the_gfs_chain_prepares_beside_an_as_posted_fetch(
        tmp_path, monkeypatch, gfs_config):
    """No manifest stage; the preparation waits on the fetch's markers."""

    ran: list[tuple[str, list[str]]] = []

    def stage(label, command, **kw):
        ran.append((label, list(command)))
        if label == "fetch":
            _post_window(command)
        if label == "prepare":
            raise go_cli.GoStageFailed(9)

    _quiet_gates(monkeypatch)
    monkeypatch.setattr(go_cli, "_run_stage", stage)
    assert go_cli.go_main(_args(gfs_config, tmp_path / "go")) == 9
    assert [label for label, _ in ran] == ["authority", "fetch", "prepare"]
    fetch, prepare = ran[1][1], ran[2][1]
    posting = Path(fetch[fetch.index("--out") + 1]) / "posting"
    assert prepare[prepare.index("--as-posted") + 1] == str(posting)
    assert "--source-manifest" not in prepare
    assert "--source-manifest-sha256" not in prepare


def test_a_window_already_here_whole_keeps_the_manifest_stage(
        tmp_path, monkeypatch, gfs_config):
    """A fetch that scheduled nothing (a cached window) binds it whole."""

    ran: list[str] = []

    def stage(label, command, **kw):
        ran.append(label)
        if label == "manifest":
            raise go_cli.GoStageFailed(9)

    _quiet_gates(monkeypatch)
    monkeypatch.setattr(go_cli, "_run_stage", stage)
    assert go_cli.go_main(_args(gfs_config, tmp_path / "go")) == 9
    assert ran == ["authority", "fetch", "manifest"]


def test_a_schedule_left_by_an_earlier_fetch_is_not_this_ones(
        tmp_path, monkeypatch, gfs_config):
    """An old schedule in a reused download does not start an as-posted prep."""

    import os

    ran: list[str] = []

    def stage(label, command, **kw):
        ran.append(label)
        if label == "fetch":
            # Dated in the past before it appears, as a file an earlier
            # fetch left behind is: the chain may look at any moment.
            out = Path(command[command.index("--out") + 1])
            staged = _post_window(["--out", str(out / "earlier"),
                                   "--cycle", command[command.index(
                                       "--cycle") + 1]])
            os.utime(staged / "schedule.json", (1.0e9, 1.0e9))
            (out / "posting").mkdir(parents=True, exist_ok=True)
            os.replace(staged / "schedule.json",
                       out / "posting" / "schedule.json")
        if label == "manifest":
            raise go_cli.GoStageFailed(9)

    _quiet_gates(monkeypatch)
    monkeypatch.setattr(go_cli, "_run_stage", stage)
    assert go_cli.go_main(_args(gfs_config, tmp_path / "go")) == 9
    assert ran == ["authority", "fetch", "manifest"]


def test_a_fetch_that_fails_ends_the_waiting_preparation_and_is_the_chains(
        tmp_path, monkeypatch, gfs_config):
    """The preparation hears the fetch's failure; go exits with the fetch's code."""

    heard: dict[str, str] = {}
    scheduled = threading.Event()

    def stage(label, command, **kw):
        if label == "fetch":
            posting = _post_window(command, markers=False)
            scheduled.set()
            heard["posting"] = str(posting)
            raise go_cli.GoStageFailed(5, "the host refused the request")
        if label == "prepare":
            posting = Path(command[command.index("--as-posted") + 1])
            cycle = json.loads((posting / "schedule.json").read_text())["cycle"]
            waiter = PostedLeads(posting, source="gfs", cycle=cycle,
                                 poll_seconds=0.01)
            try:
                waiter.wait(0)
            except BoundaryProducerFailed as error:
                heard["prepare"] = str(error)
            raise go_cli.GoStageFailed(1)

    _quiet_gates(monkeypatch)
    monkeypatch.setattr(go_cli, "_run_stage", stage)
    assert go_cli.go_main(_args(gfs_config, tmp_path / "go")) == 5
    record = json.loads(
        (Path(heard["posting"]) / "failed.json").read_text())
    assert record["code"] == go_cli.FETCH_FAILED_CODE
    assert "the host refused the request" in record["message"]
    assert "the host refused the request" in heard["prepare"]


def test_a_fetch_that_ends_without_a_lead_says_so_and_keeps_source_behind(
        tmp_path):
    plan = {"data": tmp_path, "source": "gfs", "cycle": "2026-09-30T18"}
    command = ["fetch", "--out", str(tmp_path), "--cycle", "2026-09-30T18"]
    posting = _post_window(command, leads=(0, 1, 2), markers=False)
    (posting / "f000.json").write_text("{}")
    go_cli._record_fetch_end(posting, plan, since=0.0, error=None)
    record = json.loads((posting / "failed.json").read_text())
    assert record["code"] == go_cli.FETCH_INCOMPLETE_CODE
    assert record["leads_missing"] == [1, 2]
    # The fetch's own record is never replaced.
    (posting / "failed.json").write_text(json.dumps(
        {"code": "source_behind", "lead": 1}))
    go_cli._record_fetch_end(posting, plan, since=0.0,
                             error=go_cli.GoStageFailed(75))
    assert json.loads((posting / "failed.json").read_text())["code"] \
        == "source_behind"


def test_a_whole_window_fetch_with_every_marker_writes_no_failure(tmp_path):
    plan = {"data": tmp_path, "source": "gfs", "cycle": "2026-09-30T18"}
    command = ["fetch", "--out", str(tmp_path), "--cycle", "2026-09-30T18"]
    posting = _post_window(command, leads=(0, 1))
    go_cli._record_fetch_end(posting, plan, since=0.0, error=None)
    assert not (posting / "failed.json").exists()
    # Stopped after its last marker: nothing waits, nothing to say.
    go_cli._record_fetch_end(posting, plan, since=0.0,
                             error=go_cli.GoStageStopped(130), stopped=True)
    assert not (posting / "failed.json").exists()


# A fetch stage that is a real process: it schedules the window, then waits
# for leads until it is stopped (exit 130 on SIGINT, as `gpuwm fetch` does),
# or, with a lead past its budget, writes its own source_behind record and
# exits 75 once the preparation has ended on it.
_FETCH_STAGE = r"""
import json, os, signal, sys, time

out, cycle, mode = sys.argv[1], sys.argv[2], sys.argv[3]
# How long the fetch takes to end once the preparation has ended on its
# source_behind record (a loaded host is slow to schedule it).
ends_after = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
posting = os.path.join(out, "posting")
os.makedirs(posting, exist_ok=True)


def publish(name, document):
    staged = os.path.join(posting, name + ".tmp")
    with open(staged, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    os.replace(staged, os.path.join(posting, name))


publish("schedule.json", {
    "schema": "gpuwm.posting-schedule.v1", "source": "gfs", "cycle": cycle,
    "table_sha256": "7" * 64,
    "leads": [{"lead": 0, "state": "waiting"},
              {"lead": 1, "state": "scheduled"}]})
print("fetch gfs: f000 not posted yet", flush=True)
if mode == "late-behind":
    deadline = time.monotonic() + 30.0
    while not os.path.exists(os.path.join(out, "forecast-failed")):
        assert time.monotonic() < deadline
        time.sleep(0.05)
if mode in ("behind", "late-behind"):
    publish("failed.json", {"code": "source_behind", "source": "gfs",
                            "cycle": cycle, "lead": 1,
                            "message": "gfs f001 has not posted by its late time"})
    done = os.path.join(out, "prepare-ended")
    deadline = time.monotonic() + 30.0
    while not os.path.exists(done) and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(ends_after)
    print("fetch: gfs f001 has not posted by its late time", file=sys.stderr)
    sys.exit(75)
time.sleep(120)
sys.exit(0)
"""

# A preparation stage that is a real process: it fails (``fail``), publishes
# a head and seals (``head``), or ends on the fetch's failure record
# (``behind``).
_PREPARE_STAGE = r"""
import os, sys, time

prepared, posting, mode = sys.argv[1], sys.argv[2], sys.argv[3]
fetch_end_marker = sys.argv[4] if len(sys.argv) > 4 else None
if mode == "fail":
    print("prepare: the decode refused the first lead", file=sys.stderr)
    sys.exit(9)
if mode == "behind":
    while not os.path.exists(os.path.join(posting, "failed.json")):
        time.sleep(0.05)
    print("prepare: gfs f001 fell behind", file=sys.stderr)
    open(os.path.join(os.path.dirname(posting), "prepare-ended"), "w").close()
    sys.exit(75)
stream = os.path.join(prepared, "boundary-stream")
os.makedirs(stream, exist_ok=True)
with open(os.path.join(stream, "head.json"), "w", encoding="utf-8") as handle:
    handle.write("{}")
if mode == "head-fail":
    deadline = time.monotonic() + 30.0
    while not os.path.exists(os.path.join(os.path.dirname(posting), "forecast-running")):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    print("prepare: the later input failed validation", file=sys.stderr)
    sys.exit(9)
if mode == "head-behind":
    deadline = time.monotonic() + 30.0
    while not os.path.exists(os.path.join(posting, "failed.json")):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    open(os.path.join(os.path.dirname(posting), "prepare-ended"), "w").close()
    if fetch_end_marker is not None:
        while not os.path.exists(fetch_end_marker):
            assert time.monotonic() < deadline
            time.sleep(0.05)
    sys.exit(75)
time.sleep(1.0)
sys.exit(0)
"""


class _Host:
    """The run observer a hosting run-plan wraps in ``_GoObserver``."""

    def __init__(self):
        self.warnings: list[tuple[str, dict]] = []

    def enter_stage(self, stage, *, phase=None):
        pass

    def warn(self, code, message, **fields):
        self.warnings.append((code, fields))


def _real_stages(monkeypatch, tmp_path, *, fetch_mode, prepare_mode,
                 fetch_ends_after=0.5, fetch_end_marker=None):
    """go's chain with a real fetch and preparation process (authority stubbed)."""

    import sys

    fetch_script = tmp_path / "fetch_stage.py"
    fetch_script.write_text(_FETCH_STAGE, encoding="utf-8")
    prepare_script = tmp_path / "prepare_stage.py"
    prepare_script.write_text(_PREPARE_STAGE, encoding="utf-8")
    real_run_stage = go_cli._run_stage
    real_fetch_command = go_cli.fetch_command

    def fetch_command(plan):
        command = real_fetch_command(plan)
        return [sys.executable, str(fetch_script),
                command[command.index("--out") + 1],
                command[command.index("--cycle") + 1], fetch_mode,
                str(fetch_ends_after)]

    def prepare_command(plan, bridge, **kw):
        command = [sys.executable, str(prepare_script), str(plan["prepared"]),
                   str(kw["as_posted"]), prepare_mode]
        if fetch_end_marker is not None:
            command.append(str(fetch_end_marker))
        return command

    def stage(label, command, **kw):
        if label == "authority":
            return None
        return real_run_stage(label, command, **kw)

    _quiet_gates(monkeypatch)
    monkeypatch.setattr(go_cli, "fetch_command", fetch_command)
    monkeypatch.setattr(go_cli, "prepare_command", prepare_command)
    monkeypatch.setattr(go_cli, "_run_stage", stage)
    return real_run_stage


def _go(args, hosted):
    from gpuwm import runplan

    observer = runplan._GoObserver(_Host()) if hosted else None
    return go_cli.go_main(args, observer=observer), observer


@pytest.mark.parametrize("hosted", [False, True], ids=["typed", "run-plan"])
def test_a_preparation_that_fails_beside_the_fetch_is_the_chains_failure(
        tmp_path, monkeypatch, capsys, gfs_config, hosted):
    """The preparation's exit and words, not the stopped fetch's 130."""

    _real_stages(monkeypatch, tmp_path, fetch_mode="wait",
                 prepare_mode="fail")
    code, observer = _go(_args(gfs_config, tmp_path / "go"), hosted)
    printed = capsys.readouterr().out
    assert code == 9
    assert "go: stopped at prepare" in printed
    assert "the decode refused the first lead" in printed
    assert "stopped at fetch" not in printed
    assert "FAILED  fetch" not in printed
    assert "  stopped fetch (exit" in printed
    # A fetch the chain stopped says so in posting/, not as its own failure,
    # and a preparer that outlived its stage stops waiting on it by name.
    schedules = list((tmp_path / "go").rglob("posting/schedule.json"))
    assert len(schedules) == 1
    posting = schedules[0].parent
    record = json.loads((posting / "failed.json").read_text())
    assert record["code"] == go_cli.FETCH_STOPPED_CODE
    assert "exit" not in record["message"]
    cycle = json.loads((posting / "schedule.json").read_text())["cycle"]
    with pytest.raises(BoundaryProducerFailed, match="was stopped because"):
        PostedLeads(posting, source="gfs", cycle=cycle,
                    poll_seconds=0.01).wait(0)
    if hosted:
        assert observer.failure["stage"] == "prepare"
        assert observer.failure["exit_code"] == 9
        assert [fields["stage"] for said, fields in observer._observer.warnings
                if said == "chain_stage_failed"] == ["prepare"]


@pytest.mark.parametrize("hosted", [False, True], ids=["typed", "run-plan"])
def test_a_forecast_that_fails_beside_the_fetch_is_the_chains_failure(
        tmp_path, monkeypatch, capsys, gfs_config, hosted):
    import sys

    real_run_stage = _real_stages(monkeypatch, tmp_path, fetch_mode="wait",
                                  prepare_mode="head")
    monkeypatch.setattr(
        boundary_stream, "_fresh_head",
        lambda root, since: {"head_sha256": "a" * 64,
                             "decision": {"chained": True}})
    monkeypatch.setattr(go_cli, "head_digests", lambda root, head: {})

    def forecast(plan, digests, *, explain, observer):
        real_run_stage("forecast", [
            sys.executable, "-c",
            "import sys; print('forecast: the runner refused', "
            "file=sys.stderr); sys.exit(7)"],
            explain=explain, observer=observer)

    monkeypatch.setattr(go_cli, "_run_forecast", forecast)
    code, observer = _go(_args(gfs_config, tmp_path / "go"), hosted)
    printed = capsys.readouterr().out
    assert code == 7
    assert "go: stopped at forecast" in printed
    assert "the runner refused" in printed
    assert "stopped at fetch" not in printed
    assert "FAILED  fetch" not in printed
    if hosted:
        assert observer.failure["stage"] == "forecast"
        assert observer.failure["exit_code"] == 7


@pytest.mark.parametrize("hosted", [False, True], ids=["typed", "run-plan"])
@pytest.mark.parametrize("failure_kind", ["refusal", "stage"])
def test_forecast_validation_failure_precedes_a_later_source_timeout(
        tmp_path, monkeypatch, gfs_config, hosted, failure_kind, capsys):
    """Head validation is part of the forecast callback, before its runner.

    Preparation waits for the later lead after validation fails. Its fetch
    times out during that hold, so sampling failure order after the hold
    would report the source timeout instead of the validation failure.
    """

    fetch_end_marker = tmp_path / "fetch-stage-ended"
    _real_stages(monkeypatch, tmp_path, fetch_mode="late-behind",
                 prepare_mode="head-behind", fetch_ends_after=0.0,
                 fetch_end_marker=fetch_end_marker)
    notify = go_cli._notify

    def observed_notify(observer, event, **fields):
        notify(observer, event, **fields)
        if (event in {"stage_end", "stage_secondary_end"}
                and fields.get("label") == "fetch"
                and fields.get("exit_code") == 75):
            fetch_end_marker.touch()

    monkeypatch.setattr(go_cli, "_notify", observed_notify)
    monkeypatch.setattr(
        boundary_stream, "_fresh_head",
        lambda root, since: {"head_sha256": "a" * 64,
                             "decision": {"chained": True}})
    message = "forecast head validation failed before the later lead"
    original = (go_cli.GoRefusal(message) if failure_kind == "refusal"
                else go_cli.GoStageFailed(7, message))

    def head_digests(root, head):
        raise original

    data = tmp_path / "data"
    held = threading.Event()
    hold_for_seal = boundary_stream._hold_for_seal

    def hold(root, worker, observer):
        held.set()
        (data / "forecast-failed").touch()
        return hold_for_seal(root, worker, observer, report_seconds=0.01)

    monkeypatch.setattr(go_cli, "head_digests", head_digests)
    monkeypatch.setattr(boundary_stream, "_hold_for_seal", hold)
    args = _args(gfs_config, tmp_path / "go")
    args.data_dir = data
    from gpuwm import runplan

    observer = runplan._GoObserver(_Host()) if hosted else None
    if failure_kind == "refusal":
        with pytest.raises(go_cli.GoRefusal, match=message) as caught:
            go_cli.go_main(args, observer=observer)
        assert caught.value is original
    else:
        assert go_cli.go_main(args, observer=observer) == 7
    assert held.is_set()
    assert fetch_end_marker.is_file()
    printed = capsys.readouterr().out
    assert "FAILED  fetch" not in printed
    assert "stopped at fetch" not in printed
    assert "FAILED  prepare" not in printed
    assert "stopped at prepare" not in printed
    record = json.loads((data / "posting" / "failed.json").read_text())
    assert record["code"] == "source_behind"
    if hosted:
        assert observer.failure == {
            "stage": "forecast", "exit_code": 2 if failure_kind == "refusal" else 7,
            "diagnostic": message}
        secondary = [fields for said, fields in observer._observer.warnings
                     if said == "secondary_source_behind"]
        assert secondary[-1]["source_behind"]["message"] == record["message"]
        assert not any(fields.get("stage") == "fetch"
                       for said, fields in observer._observer.warnings
                       if said == "chain_stage_failed")


@pytest.mark.parametrize("hosted", [False, True], ids=["typed", "run-plan"])
def test_a_preparation_failure_before_its_forecast_consequence_stays_primary(
        tmp_path, monkeypatch, gfs_config, hosted):
    """Both callbacks retain the first failure's stage, diagnostic and code."""

    _real_stages(monkeypatch, tmp_path, fetch_mode="wait",
                 prepare_mode="head-fail")
    monkeypatch.setattr(
        boundary_stream, "_fresh_head",
        lambda root, since: {"head_sha256": "a" * 64,
                             "decision": {"chained": True}})
    monkeypatch.setattr(go_cli, "head_digests", lambda root, head: {})
    failed = threading.Event()
    mark_failed = boundary_stream._mark_failed_if_unsealed

    def mark(root, error, **kwargs):
        try:
            return mark_failed(root, error, **kwargs)
        finally:
            # run_chained invokes this after the complete preparation
            # callback captured and re-raised its first exception.
            failed.set()

    data = tmp_path / "data"

    def forecast(plan, digests, **kwargs):
        (data / "forecast-running").touch()
        assert failed.wait(20), "the preparation did not fail first"
        raise go_cli.GoStageFailed(7, "forecast stopped after its input failed")

    monkeypatch.setattr(boundary_stream, "_mark_failed_if_unsealed", mark)
    monkeypatch.setattr(go_cli, "_run_forecast", forecast)
    args = _args(gfs_config, tmp_path / "go")
    args.data_dir = data
    code, observer = _go(args, hosted)
    assert code == 9
    if hosted:
        assert observer.failure["stage"] == "prepare"
        assert observer.failure["exit_code"] == 9
        assert "the later input failed validation" in observer.failure["diagnostic"]
        assert "forecast stopped" not in observer.failure["diagnostic"]


@pytest.mark.parametrize("slow", [False, True],
                         ids=["prompt", "past-the-stop-grace"])
def test_a_fetch_behind_its_budget_is_the_chains_failure_when_it_ends_last(
        tmp_path, monkeypatch, gfs_config, slow):
    """A fetch writes source_behind before it exits, so the preparation can
    end on it first; the fetch's failure is still the run's.  Slow: it
    ends later than the 2 s any other stop allows, as on a loaded host,
    where the chain stopped it and named the preparation."""

    _real_stages(monkeypatch, tmp_path, fetch_mode="behind",
                 prepare_mode="behind",
                 fetch_ends_after=(go_cli.END_STAGE_GRACE_SECONDS + 1.5
                                   if slow else 0.5))
    code, observer = _go(_args(gfs_config, tmp_path / "go"), True)
    assert code == 75
    assert observer.failure["stage"] == "fetch"
    assert observer.failure["exit_code"] == 75
    assert "has not posted by its late time" in observer.failure["diagnostic"]


def test_a_run_plan_chain_failure_is_the_first_stage_to_fail():
    """A stage that fails because an earlier one did does not replace it."""

    from gpuwm import runplan

    chain = runplan._GoObserver(_Host())
    chain.stage_failed(label="fetch", exit_code=5, diagnostic="host refused")
    chain.stage_failed(label="prepare", exit_code=1, diagnostic="no f001")
    chain.stage_end(label="prepare", exit_code=1, ok=False,
                    elapsed_seconds=1.0, progress=None)
    assert chain.failure == {"stage": "fetch", "exit_code": 5,
                             "diagnostic": "host refused"}
    # go names the chain's failure when it decides it after the fact.
    later = runplan._GoObserver(_Host())
    later.stage_failed(label="prepare", exit_code=75, diagnostic="behind")
    later.chain_failed(label="fetch", exit_code=75, diagnostic="f001 late")
    assert later.failure["stage"] == "fetch"


@pytest.mark.parametrize("change, beside", [
    ({}, True),
    ({"as_posted": False}, False),
    ({"source": "gdas"}, False),
])
def test_which_chains_prepare_beside_the_fetch(change, beside):
    plan = {"source": "gfs", "as_posted": None, "domains": 1, **change}
    assert go_cli.posts_beside_preparation(plan) is beside


def test_the_dry_run_shows_the_as_posted_order(tmp_path, capsys, monkeypatch,
                                                gfs_config):
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))
    args = _args(gfs_config, tmp_path / "go")
    args.dry_run = True
    assert go_cli.go_main(args) == 0
    printed = capsys.readouterr().out
    assert "2. fetch (as posted, beside step 4" in printed
    assert "3. manifest (only for a window the fetch finds already here" \
        in printed
    prepare = printed.split("4. prepare")[1].split("\n")[1]
    assert "--as-posted" in prepare and "--source-manifest" not in prepare


# ---------------------------------------------------------------------------
# The forecast binds an as-posted head by its plan
# ---------------------------------------------------------------------------

def test_go_binds_an_as_posted_head_without_a_manifest_digest(tmp_path):
    _writer, output, _digest = _as_posted_tree(tmp_path)
    head = boundary_stream.read_head(output)
    digests = go_cli.head_digests(output, head["head_sha256"])
    assert digests == {"prepared_head": head["head_sha256"],
                       "source_manifest": None}
    plan = {"runner": go_cli.RUNNER_MODULE, "source": "gfs",
            "prepared": output, "authority": tmp_path / "authority",
            "run": tmp_path / "run", "profile": None}
    command = go_cli.forecast_command(plan, digests)
    assert command[command.index("--prepared-head-sha256") + 1] \
        == head["head_sha256"]
    assert "--source-manifest-sha256" not in command


def test_an_as_posted_head_binds_its_plan_where_a_manifest_digest_goes():
    from gpuwm import prepared_single_domain_forecast as runner

    plan = boundary_stream.input_plan(
        {"files": {"grib-f000": {"name": "a", "sha256": "1" * 64}}},
        lead_role_prefix="grib-f", route_table_sha256="7" * 64)
    digest = boundary_stream.input_plan_sha256(plan)
    posted = {"input_plan": plan, "input_plan_sha256": digest}
    assert runner._as_posted_head_binding("gfs", posted, None) \
        == boundary_stream.as_posted_placeholder(digest)
    with pytest.raises(ValueError, match="omit the flag"):
        runner._as_posted_head_binding("gfs", posted, "0" * 64)
    with pytest.raises(ValueError, match="not the plan its digest"):
        runner._as_posted_head_binding(
            "gfs", {**posted, "input_plan_sha256": "e" * 64}, None)
    # A mapped head binds its plan too since A136 L10: what its seal writes
    # is checked by the sealed preflight (it was refused by name before).
    mapped = sorted(runner._MAPPED_SOURCES)[0]
    assert runner._as_posted_head_binding(mapped, posted, None)         == boundary_stream.as_posted_placeholder(digest)


def test_only_the_plans_pending_roles_may_carry_no_digest():
    from gpuwm import prepared_single_domain_forecast as runner

    manifest = {
        "schema": "gpuwm-gfs-direct-input-manifest-v1",
        "source": {"model": "GFS", "product": "pgrb2.0p25",
                   "cycle": "2026-09-30T18:00:00Z"},
        "files": {
            "bridge": {"name": "bridge", "sha256": "b" * 64},
            "experiment_config": {"name": "e.toml", "sha256": "e" * 64},
            "wps_namelist": {"name": "n.wps", "sha256": "d" * 64},
            "series": {"name": "gfs-series.tsv", "sha256": None},
            "grib-f000": {"name": "gfs.f000", "sha256": None},
            "grib-f001": {"name": "gfs.f001", "sha256": None},
        }}
    posted = {"lead_role_prefix": "grib-f", "derived_roles": ["series"]}
    pending = runner._as_posted_pending_roles(posted, manifest)
    assert pending == {"series", "grib-f000", "grib-f001"}
    assert runner._as_posted_pending_roles(None, manifest) == frozenset()
    files = manifest["files"]
    # A role outside the plan's pending set still needs its digest.
    with pytest.raises(ValueError, match="bridge sha256 must be"):
        runner._manifest_file_specs(
            "gfs", {**manifest, "files": {
                **files, "bridge": {"name": "bridge", "sha256": None}}},
            None, {}, pending=pending)


def test_the_seal_preflight_binds_the_manifest_its_seal_wrote():
    from types import SimpleNamespace

    from gpuwm import prepared_single_domain_forecast as runner

    inputs = SimpleNamespace(preflight_arguments={
        "source": "gfs", "source_manifest_sha256": None})
    sealed = {"proof_sha256": "1" * 64, "content_sha256": "2" * 64,
              "as_posted": {"input_manifest_sha256": "3" * 64}}
    assert runner._sealed_arguments(inputs, sealed) == {
        "source": "gfs", "source_manifest_sha256": "3" * 64,
        "proof_sha256": "1" * 64, "prepared_content_sha256": "2" * 64}
    one_shot = {"proof_sha256": "1" * 64, "content_sha256": "2" * 64}
    inputs = SimpleNamespace(preflight_arguments={
        "source": "gfs", "source_manifest_sha256": "4" * 64})
    assert runner._sealed_arguments(inputs, one_shot)[
        "source_manifest_sha256"] == "4" * 64


def test_the_sealed_identity_may_change_only_where_the_manifest_goes():
    from types import SimpleNamespace

    from gpuwm import prepared_single_domain_forecast as runner

    placeholder = boundary_stream.as_posted_placeholder("p" * 64)
    head = {"basis": {"as_posted": {
        "input_plan_sha256": "p" * 64,
        "manifest_bound_identity_keys": [
            "bridge_manifest_sha256", "input_manifest_sha256",
            "source_manifest_sha256"]}}}
    at_head = SimpleNamespace(cache_identity={
        "bridge_manifest_sha256": placeholder,
        "source_manifest_sha256": placeholder,
        "source_identity": {"input_manifest_sha256": placeholder, "a": 1}})
    manifest = "a" * 64

    def sealed(**identity):
        return SimpleNamespace(cache_identity=identity,
                               file_sha256={"source_manifest": manifest})

    runner._require_sealed_identity(at_head, sealed(
        bridge_manifest_sha256=manifest, source_manifest_sha256=manifest,
        source_identity={"input_manifest_sha256": manifest, "a": 1}), head)
    with pytest.raises(RuntimeError, match="as-posted head"):
        runner._require_sealed_identity(at_head, sealed(
            bridge_manifest_sha256=manifest, source_manifest_sha256=manifest,
            source_identity={"input_manifest_sha256": manifest, "a": 2}),
            head)
    with pytest.raises(RuntimeError, match="as-posted head"):
        runner._require_sealed_identity(at_head, sealed(
            bridge_manifest_sha256="c" * 64, source_manifest_sha256=manifest,
            source_identity={"input_manifest_sha256": manifest, "a": 1}),
            head)
    # A head that is not as posted accepts no change at all.
    same = SimpleNamespace(cache_identity={"a": 1})
    runner._require_sealed_identity(same, same, {"basis": {}})
    with pytest.raises(RuntimeError, match="differs from the head"):
        runner._require_sealed_identity(
            same, SimpleNamespace(cache_identity={"a": 2}), {"basis": {}})


def test_a_checkpoint_before_an_as_posted_seal_resumes_after_it():
    """The head binds the plan, so the manifest joins the seal's authorities."""

    from types import SimpleNamespace

    from gpuwm import prepared_single_domain_forecast as runner

    authorities = {"static": "s" * 64, "experiment_config": "e" * 64}
    at_head = SimpleNamespace(
        source="gfs", proof={},
        stream_head={"head_sha256": "h" * 64,
                     "basis": {"as_posted": {"input_plan_sha256": "p"}}},
        file_sha256={**authorities, "prepared_head": "x" * 64})
    after_seal = SimpleNamespace(
        source="gfs",
        proof={"boundary_stream": {"head_sha256": "h" * 64},
               "posting": {"as_posted": True}},
        file_sha256={**authorities, "proof": "y" * 64,
                     "cache_header": "z" * 64, "source_manifest": "m" * 64})
    assert (runner._single_checkpoint_identity(at_head, {"r": 1})
            == runner._single_checkpoint_identity(after_seal, {"r": 1}))
    # A chained one-shot preparation keeps binding its manifest.
    one_shot = SimpleNamespace(
        source="gfs", proof={"boundary_stream": {"head_sha256": "h" * 64}},
        file_sha256={**authorities, "source_manifest": "m" * 64})
    assert "source_manifest" in runner._single_checkpoint_identity(
        one_shot, {"r": 1})["authority_sha256"]


def test_a_sealed_binding_still_needs_its_manifest_digest(tmp_path):
    """The flag is optional only for an as-posted head, which binds its plan."""

    from gpuwm import prepared_single_domain_forecast as runner

    with pytest.raises(ValueError, match="source-manifest-sha256 must be"):
        runner.preflight_prepared_forecast(
            source="gfs", prepared_root=tmp_path, proof_sha256="1" * 64,
            source_manifest_sha256=None, prepared_content_sha256="2" * 64,
            experiment_config=tmp_path / "e.toml",
            wps_namelist=tmp_path / "n.wps", run_seconds=3600.0,
            history_interval_seconds=3600.0)
    head = argparse.Namespace(prepared_head_sha256="4" * 64,
                              proof_sha256=None, prepared_content_sha256=None,
                              source_manifest_sha256=None)
    assert runner._preparation_binding_refusal(head) is None


def test_the_stage_door_binds_an_as_posted_head_without_a_manifest_digest(
        tmp_path):
    from gpuwm import stage_cli

    bundle = {"document": tmp_path / "boundary-stream" / "head.json",
              "root": tmp_path, "source": "gfs", "layout": "single",
              "domains": 1, "payload": {}, "head_sha256": "h" * 64,
              "source_manifest_sha256": None}
    command = stage_cli.sim_command(
        bundle, experiment_config=tmp_path / "e.toml",
        wps_namelist=tmp_path / "n.wps", outdir=tmp_path / "run")
    assert command[command.index("--prepared-head-sha256") + 1] == "h" * 64
    assert "--source-manifest-sha256" not in command
    command = stage_cli.sim_command(
        {**bundle, "source_manifest_sha256": "m" * 64},
        experiment_config=tmp_path / "e.toml",
        wps_namelist=tmp_path / "n.wps", outdir=tmp_path / "run")
    assert command[command.index("--source-manifest-sha256") + 1] == "m" * 64


# ---------------------------------------------------------------------------
# source_cli forwards --as-posted
# ---------------------------------------------------------------------------

def _gfs_args(tmp_path, **extra):
    parser = source_cli._parser()
    argv = ["--source", "gfs", "--gfs-series", str(tmp_path / "gfs-series.tsv"),
            "--cycle", "2026-09-30_18:00:00", "--bridge", str(tmp_path / "b"),
            "--wps-namelist", str(tmp_path / "n.wps"),
            "--experiment-config", str(tmp_path / "e.toml"),
            "--geog-root", str(tmp_path / "geog"),
            "--output-root", str(tmp_path / "prepared")]
    for flag, value in extra.items():
        argv += [flag, str(value)]
    return parser.parse_args(argv)


def test_source_cli_forwards_as_posted_and_the_seals_manifest_path(tmp_path):
    from gpuwm.fetch import preparation_manifest_path

    args = _gfs_args(tmp_path, **{"--as-posted": tmp_path / "posting"})
    assert source_cli._required_gfs_args(args) == []
    command = source_cli._gfs_command(args)
    assert command[command.index("--as-posted") + 1] \
        == str(tmp_path / "posting")
    assert command[command.index("--input-manifest") + 1] == str(
        preparation_manifest_path(tmp_path / "prepared"))
    assert "--input-manifest-sha256" not in command


def test_source_cli_refuses_a_manifest_pair_beside_as_posted(tmp_path):
    args = _gfs_args(tmp_path, **{
        "--as-posted": tmp_path / "posting",
        "--source-manifest": tmp_path / "m.json",
        "--source-manifest-sha256": "0" * 64})
    assert any("takes no --source-manifest pair" in error
               for error in source_cli._required_gfs_args(args))


def test_only_a_runner_that_waits_on_posted_leads_prepares_as_posted():
    assert source_cli.prepares_as_posted("gfs")
    # The mapped engine waits on posted leads since A136 L3 (ii)
    # (mapped_direct --as-posted), and its door forwards the flag (L10).
    assert source_cli.prepares_as_posted("gefs")
    assert source_cli.prepares_as_posted("hrrr")
    assert not source_cli.prepares_as_posted("no-such-source")


# ---------------------------------------------------------------------------
# The schedule the fetch keeps replacing is read through a replace
# ---------------------------------------------------------------------------

def _flaky_reads(monkeypatch, path: Path, failures: int):
    """``path`` fails to read ``failures`` times, as a read meeting a replace does."""

    real = Path.read_text
    left = {"count": failures}

    def read_text(self, *args, **kwargs):
        if Path(self) == path and left["count"] > 0:
            left["count"] -= 1
            raise PermissionError(13, "The process cannot access the file")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    return left


def test_a_read_that_meets_a_replace_is_tried_again(tmp_path, monkeypatch):
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps({"table_sha256": "7" * 64}))
    left = _flaky_reads(monkeypatch, path, 3)
    slept: list[float] = []
    assert read_replaced_json(path, sleep=slept.append) == {
        "table_sha256": "7" * 64}
    assert left["count"] == 0 and len(slept) == 3
    # A file that stays unreadable is still said, by its own error.
    _flaky_reads(monkeypatch, path, 100)
    with pytest.raises(PermissionError):
        read_replaced_json(path, sleep=lambda seconds: None)


def test_the_route_table_is_read_through_the_fetchs_replace(
        tmp_path, monkeypatch):
    """The L3 check's note: one failed read no longer fails the seal."""

    folder = tmp_path / "posting"
    folder.mkdir()
    schedule = folder / "schedule.json"
    schedule.write_text(json.dumps({"table_sha256": "7" * 64}))
    monkeypatch.setattr(boundary_stream.time, "sleep", lambda seconds: None)
    _flaky_reads(monkeypatch, schedule, 2)
    posted = PostedLeads(folder, source="gfs", cycle="2026-09-30T18")
    assert posted.route_table_sha256() == "7" * 64
    _flaky_reads(monkeypatch, schedule, 100)
    with pytest.raises(BoundaryProducerFailed, match="is not readable"):
        posted.route_table_sha256()


def test_the_fetch_beat_line_says_the_lead_the_run_waits_for(tmp_path):
    """Before the first leads post nothing else of the run moves, so the
    fetch's beat line reads its own schedule rather than a bare clock."""

    schedule = tmp_path / "schedule.json"
    rows = [{"lead": 0, "state": "ready"},
            {"lead": 1, "state": "waiting",
             "expected_at": "2026-09-30T21:31:07Z"},
            {"lead": 2, "state": "scheduled"}]
    schedule.write_text(json.dumps({
        "schema": "gpuwm.posting-schedule.v1", "source": "gfs",
        "leads": rows}))
    assert go_cli._progress_note(schedule) == (
        ", 1 of 3 leads in, waiting for gfs f001, not posted yet "
        "(scheduled from about 21:31Z)")
    rows[1]["state"] = "ready"
    schedule.write_text(json.dumps({
        "schema": "gpuwm.posting-schedule.v1", "source": "gfs",
        "leads": rows}))
    assert go_cli._progress_note(schedule) == ", 2 of 3 leads in"


def test_a_hosting_run_plan_stream_carries_the_fetchs_posting(tmp_path):
    """run-plan hosts the GFS chain with its own stream: go's posting relay
    carries the schedule and each lead onto it."""

    from gpuwm import chain_events

    command = ["fetch", "--out", str(tmp_path), "--cycle", "2026-09-30T18"]
    posting = _post_window(command, leads=(0, 1))
    schedule = json.loads((posting / "schedule.json").read_text())
    schedule["leads"] = [
        {"lead": lead, "valid_time": f"2026-09-30T{18 + lead:02d}:00:00Z",
         "expected_at": "2026-09-30T21:31:00Z",
         "first_seen_at": "2026-09-30T21:35:56Z", "state": "ready"}
        for lead in (0, 1)]
    (posting / "schedule.json").write_text(json.dumps(schedule))

    class Stream:
        def __init__(self):
            self.said = []

        def emit(self, event, **fields):
            self.said.append((event, fields))

    stream = Stream()
    relay = chain_events.HostedPostingRelay(stream, data_dir=tmp_path)
    relay.start(since_unix_ms=0)
    relay.stop()
    names = [event for event, _ in stream.said]
    assert names.count("posting_schedule") == 1
    assert names.count("lead_ready") == 2
    assert names.count("lead_posted") == 2
    ready = [fields for event, fields in stream.said if event == "lead_ready"]
    assert [fields["lead"] for fields in ready] == [0, 1]


@pytest.mark.parametrize("asked", [True, False])
def test_a_fetch_stage_of_its_own_carries_its_posting_onto_the_stream(
        tmp_path, monkeypatch, asked):
    """run-plan's fetch stage (every chain whose preparation reads a whole
    window, and gpuwm go on those sources, which run-plan hosts) carries
    the fetch's schedule and each lead as it is dated, while the fetch
    runs; a caller that relays the folder itself leaves it off."""

    import argparse as _argparse

    from gpuwm import cli as cli_module
    from gpuwm import runplan

    out = tmp_path / "data"
    command = ["fetch", "--out", str(out), "--cycle", "2026-10-01T00"]

    def fetch(_args):
        # What an as-posted fetch leaves while it runs: its schedule, each
        # lead's row dated when a host first held it, and its markers.
        posting = _post_window(command, leads=(0, 3))
        schedule = json.loads((posting / "schedule.json").read_text())
        schedule["leads"] = [
            {"lead": lead, "valid_time": f"2026-10-01T{lead:02d}:00:00Z",
             "expected_at": "2026-10-01T07:30:00Z",
             "first_seen_at": "2026-10-01T07:34:40Z", "state": "ready"}
            for lead in (0, 3)]
        (posting / "schedule.json").write_text(json.dumps(schedule))
        return 0

    monkeypatch.setattr(cli_module, "parse_fetch_arguments",
                        lambda argv: _argparse.Namespace(out=out, func=fetch))

    class Stream:
        def __init__(self):
            self.said = []

        def emit(self, event, **fields):
            self.said.append((event, fields))

    stream = Stream()
    runplan._run_fetch(command, tmp_path, events=stream, posting_relay=asked)
    names = [event for event, _ in stream.said]
    if not asked:
        assert not {"posting_schedule", "lead_posted", "lead_ready"} & set(names)
        return
    assert names.count("posting_schedule") == 1
    posted = [fields for event, fields in stream.said if event == "lead_posted"]
    assert [fields["lead"] for fields in posted] == [0, 3]
    assert posted[0]["first_seen_at"] == "2026-10-01T07:34:40Z"
    assert posted[0]["minutes_after_expected"] == pytest.approx(4.667, abs=0.01)
    assert names.count("lead_ready") == 2


#: A GFS single domain's start needs as the fetch writes them into its
#: schedule (``start_needs``, from source_readiness.start_needs).
_START_NEEDS = [
    {"role": "analysis", "source": "gfs", "lead": 0,
     "expected_at": "2026-09-30T21:31:00Z", "late_at": "2026-09-30T23:31:00Z"},
    {"role": "first_boundary", "source": "gfs", "lead": 1,
     "expected_at": "2026-09-30T21:32:00Z", "late_at": "2026-09-30T23:32:00Z"}]


class _WaitHost:
    def __init__(self):
        self.said = []

    def enter_stage(self, stage, *, phase=None):
        self.said.append(("stage", phase))

    def waiting(self, on, **record):
        self.said.append(("waiting", on, record["lead"]))

    def waited(self):
        self.said.append(("waited",))


def test_a_run_plan_heartbeat_does_not_wait_on_a_lead_the_start_does_not_need():
    """The fetch polls later leads while the preparation builds the head
    from start needs that are in: the run is not waiting on the source."""

    from gpuwm import runplan

    host = _WaitHost()
    chain = runplan._GoObserver(host)
    chain.stage_begin(label="fetch", command=[])
    leads = [{"lead": lead, "state": "ready"} for lead in range(5)]
    leads.append({"lead": 5, "state": "waiting",
                  "expected_at": "2026-09-30T21:36:00Z"})
    schedule = {"schema": "gpuwm.posting-schedule.v1", "source": "gfs",
                "start_needs": _START_NEEDS, "leads": leads}
    chain.stage_heartbeat(label="fetch", elapsed_seconds=20.0,
                          progress=schedule)
    assert not [entry for entry in host.said if entry[0] == "waiting"]
    # A start need still waiting is the run's wait, whatever else waits.
    leads[1]["state"] = "waiting"
    chain.stage_heartbeat(label="fetch", elapsed_seconds=40.0,
                          progress=schedule)
    assert host.said[-1] == ("waiting", "source", 1)
    # A donor's lead is not this window's row of the same number.
    schedule["start_needs"] = [dict(_START_NEEDS[0]),
                               {**_START_NEEDS[1], "source": "gdas"}]
    chain.stage_heartbeat(label="fetch", elapsed_seconds=60.0,
                          progress=schedule)
    assert host.said[-1] == ("waited",)


def test_a_run_plan_heartbeat_says_the_start_wait_on_the_source():
    """Before the forecast begins, a lead the as-posted fetch waits for is
    the run's wait: waiting:source on the heartbeat, ended when it posts."""

    from gpuwm import runplan

    class Host:
        def __init__(self):
            self.said = []

        def enter_stage(self, stage, *, phase=None):
            self.said.append(("stage", phase))

        def waiting(self, on, **record):
            self.said.append(("waiting", on, record["lead"],
                              record["expected_at"], record["since_utc"]))

        def waited(self):
            self.said.append(("waited",))

    host = Host()
    chain = runplan._GoObserver(host)
    chain.stage_begin(label="fetch", command=[])
    schedule = {"schema": "gpuwm.posting-schedule.v1", "source": "gfs",
                "start_needs": _START_NEEDS, "leads": [
                    {"lead": 0, "state": "waiting",
                     "expected_at": "2026-09-30T21:31:00Z",
                     "late_at": "2026-09-30T23:31:00Z"},
                    {"lead": 1, "state": "scheduled"}]}
    chain.stage_heartbeat(label="fetch", elapsed_seconds=20.0,
                          progress=schedule)
    chain.stage_heartbeat(label="fetch", elapsed_seconds=40.0,
                          progress=schedule)
    waits = [entry for entry in host.said if entry[0] == "waiting"]
    assert [entry[1:4] for entry in waits] == [
        ("source", 0, "2026-09-30T21:31:00Z")] * 2
    assert waits[0][4] == waits[1][4]      # one wait, refreshed
    schedule["leads"][0]["state"] = "ready"
    chain.stage_heartbeat(label="fetch", elapsed_seconds=60.0,
                          progress=schedule)
    assert host.said[-1] == ("waited",)
    # Once the forecast begins, its own seam waits speak; the fetch's
    # schedule no longer does.
    schedule["leads"][1]["state"] = "waiting"
    chain.stage_begin(label="forecast", command=[])
    chain.stage_heartbeat(label="fetch", elapsed_seconds=80.0,
                          progress=schedule)
    assert host.said[-1] == ("stage", "forecast")


# ---------------------------------------------------------------------------
# The run's phase: start source wait, on its stream (DESIGN A136 3.5, 3.7)
# ---------------------------------------------------------------------------

class _Stream:
    """A run's event stream: what was emitted, in order."""

    def __init__(self):
        self.said = []

    def emit(self, event, **fields):
        self.said.append((event, fields))


def _lead_row(lead, state, **extra):
    return {"lead": lead, "valid_time": f"2026-09-30T{18 + lead:02d}:00:00Z",
            "expected_at": f"2026-09-30T21:{31 + lead:02d}:00Z",
            "late_at": f"2026-09-30T23:{31 + lead:02d}:00Z", "state": state,
            **extra}


def _start_schedule(posting, *rows):
    posting.mkdir(parents=True, exist_ok=True)
    (posting / "schedule.json").write_text(json.dumps({
        "schema": "gpuwm.posting-schedule.v1", "source": "gfs",
        "cycle": "2026-09-30T18", "start_needs": _START_NEEDS,
        "leads": list(rows)}))


def _waits(stream, since=0):
    return [(event, fields) for event, fields in stream.said[since:]
            if event.startswith("source_wait")]


def test_a_run_waiting_on_its_start_needs_says_phase_start_source_waits(
        tmp_path):
    """Launched before its first leads post (the site rule), the run's
    stream says each start need it waits on, as a seam wait is said."""

    from gpuwm import chain_events, runplan

    stream = _Stream()
    relay = chain_events.HostedPostingRelay(stream, data_dir=tmp_path)._relay
    posting = tmp_path / "posting"
    _start_schedule(posting, _lead_row(0, "waiting"),
                    _lead_row(1, "scheduled"), _lead_row(2, "scheduled"))
    relay.relay_posting()
    relay.relay_posting()
    [(event, started)] = _waits(stream)
    assert event == "source_wait_started"
    assert set(runplan.POSTING_EVENT_FIELDS[event]) <= set(started)
    assert started["phase"] == "start" and started["lead"] == 0
    assert (started["source"], started["cycle"]) == ("gfs", "2026-09-30T18")
    assert started["waited_seconds"] == 0.0
    assert started["model_elapsed_seconds"] is None
    assert started["model_valid_time"] is None
    assert started["interval"] is None
    assert started["expected_at"] == "2026-09-30T21:31:00Z"
    assert started["reason"] == ("gfs f000 is not posted yet (scheduled from "
                                 "about 21:31Z; late at 23:31Z)")
    # Every 60 s while it lasts, so a reader tailing the stream can tell
    # the wait from a hang.
    relay._start_wait["since"] -= 61.0
    relay._start_wait["said"] -= 61.0
    mark = len(stream.said)
    relay.relay_posting()
    [(event, progress)] = _waits(stream, mark)
    assert event == "source_wait_progress" and progress["lead"] == 0
    assert set(runplan.POSTING_EVENT_FIELDS[event]) <= set(progress)
    assert progress["waited_seconds"] >= 61.0
    # f000 posts: its wait ends with when it posted, and f001's begins.
    mark = len(stream.said)
    _start_schedule(posting,
                    _lead_row(0, "posted", first_seen_at="2026-09-30T21:31:40Z"),
                    _lead_row(1, "waiting"), _lead_row(2, "scheduled"))
    relay.relay_posting()
    said = stream.said[mark:]
    assert [event for event, _ in said] == [
        "lead_posted", "source_wait_finished", "source_wait_started"]
    finished = said[1][1]
    assert set(runplan.POSTING_EVENT_FIELDS["source_wait_finished"]) <= set(
        finished)
    assert finished["phase"] == "start" and finished["lead"] == 0
    assert finished["first_seen_at"] == "2026-09-30T21:31:40Z"
    assert finished["waited_seconds"] >= 61.0
    assert said[2][1]["lead"] == 1
    # The start needs are in: the fetch polling a later lead while the
    # preparation builds the head is not the run's wait.
    mark = len(stream.said)
    _start_schedule(posting,
                    _lead_row(0, "ready", first_seen_at="2026-09-30T21:31:40Z"),
                    _lead_row(1, "ready", first_seen_at="2026-09-30T21:32:10Z"),
                    _lead_row(2, "waiting"))
    relay.relay_posting()
    relay.relay_posting()
    waits = _waits(stream, mark)
    assert [(event, fields["lead"]) for event, fields in waits] == [
        ("source_wait_finished", 1)]



def test_after_the_head_a_start_need_asked_again_is_not_a_start_wait(
        tmp_path):
    """Live on hrrr-prs 2026-10-01T06 (A136 L10): the fetch asked f001 again
    after the head was published (a transfer that did not verify is asked
    once more), and the stream said ``source_wait_started`` with ``phase:
    start`` while the forecast stepped.  The head read every start need, so
    once it is out the relay says no start wait; the forecast says its own
    waits at its seams."""

    from gpuwm import chain_events

    stream = _Stream()
    hosted = chain_events.HostedPostingRelay(stream, data_dir=tmp_path)
    relay = hosted._relay
    posting = tmp_path / "posting"
    _start_schedule(posting,
                    _lead_row(0, "ready", first_seen_at="2026-09-30T21:31:40Z"),
                    _lead_row(1, "ready", first_seen_at="2026-09-30T21:32:10Z"),
                    _lead_row(2, "scheduled"))
    relay.relay_posting()
    hosted.head_ready()
    mark = len(stream.said)
    _start_schedule(posting,
                    _lead_row(0, "ready", first_seen_at="2026-09-30T21:31:40Z"),
                    _lead_row(1, "waiting", first_seen_at="2026-09-30T21:32:10Z"),
                    _lead_row(2, "scheduled"))
    relay.relay_posting()
    relay.relay_posting()
    assert _waits(stream, mark) == []
    # The same row before the head is the run's start wait, as before.
    early = _Stream()
    before = chain_events.HostedPostingRelay(early, data_dir=tmp_path)._relay
    before.relay_posting()
    assert [(event, fields["lead"]) for event, fields in _waits(early)] == [
        ("source_wait_started", 1)]


def test_a_start_need_that_falls_behind_is_not_said_as_arrived(tmp_path):
    """A start wait that ends with the lead late says no finish: the
    run's source_behind and failed events say how it ended."""

    from gpuwm import chain_events

    stream = _Stream()
    relay = chain_events.HostedPostingRelay(stream, data_dir=tmp_path)._relay
    posting = tmp_path / "posting"
    _start_schedule(posting, _lead_row(0, "waiting"), _lead_row(1, "scheduled"))
    relay.relay_posting()
    _start_schedule(posting, _lead_row(0, "late"), _lead_row(1, "scheduled"))
    relay.relay_posting()
    assert [event for event, _ in _waits(stream)] == ["source_wait_started"]
    assert relay._start_wait is None


def test_typed_go_carries_the_start_wait_on_its_own_stream(tmp_path):
    """`gpuwm go`'s own stream (no hosting run) says the same wait, through
    run-plan's event stream writer, whose tags it must pass."""

    from gpuwm import chain_events

    data = tmp_path / "data"
    _start_schedule(data / "posting", _lead_row(0, "waiting"),
                    _lead_row(1, "scheduled"))
    chain = chain_events.GoChainEvents()
    chain.open(tmp_path / "run" / chain_events.CHAIN_EVENTS_FILENAME,
               plan={"data": data})
    chain.relay_posting()
    chain.close()
    records = chain_events.read_chain_events(
        tmp_path / "run" / chain_events.CHAIN_EVENTS_FILENAME)
    starts = [record for record in records
              if record.get("event") == "source_wait_started"]
    assert [(record["phase"], record["lead"]) for record in starts] == [
        ("start", 0)]
