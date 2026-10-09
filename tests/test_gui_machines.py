"""Machines in ``gpuwm gui``: the table, the card claim, cloud dry runs, the API, the mirror.

No SSH and no GPU: machines are stood in for by a registry whose calls
answer with documents shaped like :mod:`gpuwm.machine_agent`'s.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
from pathlib import Path

import pytest

from gpuwm import machine_agent as agent
from gpuwm.gui import cloud, runs
from gpuwm.gui.api import Api
from gpuwm.gui.jobs import Runner
from gpuwm.gui.machines import MachineError, Registry, check_row, load_rows, save_rows
from gpuwm.gui.remote_runs import Follower, plan_for


@pytest.fixture(autouse=True)
def _no_publication_probe(monkeypatch):
    # New forecast puts a recent start to the fetch's object probe before it accepts it; no server is asked in a
    # test, or a start near the publication frontier waits on the network and the page's request times out.
    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    # A page asks through its own short probe; no server is asked in a test either.
    monkeypatch.setattr("gpuwm.source_availability.quick_head", lambda url: True)
    yield
    # A page's background checks stop before the stand-ins above are undone: left running, they went on to ask
    # the real hosts, or counted HEADs in whichever test ran next.
    while OPENED:
        OPENED.pop().close()


#: Every Api a test here made, closed when the test ends.
OPENED: list = []


def aws_row(**extra):
    row = {"name": "cloud-a", "kind": "aws", "image": "ami-1", "instance_type": "g6e.xlarge",
           "region": "us-east-1", "key_name": "k", "security_group": "sg-1",
           "spend_cap_usd": 10, "price_per_hour_usd": 2.0}
    row.update(extra)
    return row


# ------------------------------------------------------------------ the table

def test_rows_round_trip_and_refusals(tmp_path):
    path = tmp_path / "machines.toml"
    row = check_row({"name": "box", "host": "me@box.local", "workspace": "/data/w",
                     "owner_file": "/data/OWNER", "env": {"GPUWM_X": "1"}})
    save_rows([row, check_row(aws_row())], path)
    rows = load_rows(path)
    assert rows[0]["host"] == "me@box.local" and rows[0]["env"] == {"GPUWM_X": "1"}
    assert rows[1]["kind"] == "aws" and rows[1]["spend_cap_usd"] == 10
    with pytest.raises(MachineError, match="Passwords are never stored"):
        check_row({"name": "box", "host": "h", "password": "x"})
    with pytest.raises(MachineError, match="no field owner-file"):
        check_row({"name": "box", "host": "h", "owner-file": "/x"})
    with pytest.raises(MachineError, match="not allowed"):
        check_row({"name": "this-computer", "host": "h"})
    with pytest.raises(MachineError, match="full path"):
        check_row({"name": "box", "host": "h", "workspace": "relative/dir"})
    with pytest.raises(MachineError, match="no cloud provider"):
        check_row({"name": "c", "kind": "nimbus"})


# ------------------------------------------------------------------ the card claim

def test_owner_claim_is_one_test_and_append(tmp_path, monkeypatch):
    owner = tmp_path / "OWNER"
    monkeypatch.setattr(agent, "cards", lambda: {"devices": [], "processes": [], "cuda_major": 13})
    owner.write_text(f"other 2026-01-01T00:00Z pid {os.getpid()} bounded 60 min\n")
    held = agent.claim_card(str(owner), "gui", 424242, 30)
    assert not held["claimed"] and "other" in held["why"]
    owner.write_text("stale 2026-01-01T00:00Z pid 999999999 bounded 60 min\n")
    claimed = agent.claim_card(str(owner), "gui", os.getpid(), 30)
    assert claimed["claimed"]
    lines = owner.read_text().splitlines()
    assert lines[-1].startswith("gui ") and lines[-1].endswith(f"pid {os.getpid()} bounded 30 min")
    agent.release_card(str(owner), os.getpid())
    assert owner.read_text().splitlines() == ["stale 2026-01-01T00:00Z pid 999999999 bounded 60 min"]
    monkeypatch.setattr(agent, "cards", lambda: {"devices": [], "processes": [{"pid": 7, "used_mib": 1}]})
    assert not agent.claim_card(str(owner), "gui", os.getpid(), 30)["claimed"]


# ------------------------------------------------------------------ cloud

def test_cloud_start_dry_run_lists_every_call_and_the_cap():
    document = cloud.plan(check_row(aws_row(spent_usd=4.0)), "start")
    steps = [step["step"] for step in document["steps"]]
    assert steps == ["identity", "launch", "wait", "address", "host_key", "settle", "set_cap", "check"]
    launch = document["steps"][1]["argv"]
    assert launch[:3] == ["aws", "ec2", "run-instances"] and "ami-1" in launch and "g6e.xlarge" in launch
    assert "--instance-initiated-shutdown-behavior" in launch
    assert document["cap_minutes"] == 180 and document["remaining_usd"] == 6.0
    assert "180" in document["user_data"] and "shutdown -h now" in document["user_data"]
    again = cloud.plan(check_row(aws_row(instance_id="i-1")), "start")
    assert [step["step"] for step in again["steps"]][:2] == ["identity", "start"]
    terminate = cloud.plan(check_row(aws_row(instance_id="i-1")), "terminate")["steps"]
    assert [step["step"] for step in terminate] == ["settle", "terminate"] and terminate[1]["argv"][-3] == "i-1"


def test_cloud_refuses_what_it_cannot_bound():
    with pytest.raises(MachineError, match="cannot be enforced"):
        cloud.plan(check_row(aws_row(price_per_hour_usd=0)), "start")
    with pytest.raises(MachineError, match="used its spend cap"):
        cloud.plan(check_row(aws_row(spent_usd=10)), "start")
    with pytest.raises(MachineError, match="no instance"):
        cloud.plan(check_row(aws_row()), "stop")


# ------------------------------------------------------------------ the API

class FakeRunner(Runner):
    kind = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.helpers: list[list[str]] = []

    def query(self, argv, *, cwd=None, timeout=0, log=None):
        if "--sources" in argv:
            # New forecast starts only from a source the engine offers.
            return {"sources": [{"source_id": "hrrr", "display_name": "HRRR", "max_forecast_hour": 48,
                                 "run_plan": {"intent_supported": True, "intent_routes": ["prepared"],
                                              "requires_source_root": False}}]}
        if "--physics-profiles" in argv:
            return {"sources": [], "profiles": []}
        return {"devices": []}

    def launch_helper(self, rundir, argv, kind):
        self.helpers.append(list(argv))
        return {"job_id": "job-x", "jobs_dir": str(rundir), "wrapper_pid": None, "argv": argv}

    def card_holder(self):
        return None

    def runtime_gap(self):
        # A start launches nothing here, so it needs no CuPy in this Python.
        return None


class FakeMachine:
    def __init__(self, row):
        self.row = row
        self.calls = []
        self.snapshots = []

    name = property(lambda self: self.row["name"])
    workspace = property(lambda self: self.row.get("workspace") or "~/gpuwm-machine")
    python = property(lambda self: self.workspace + "/venv/bin/python")
    is_local = False

    def call(self, verb, *args, payload=None, timeout=0):
        self.calls.append((verb, args, payload))
        if verb == "snapshot":
            return self.snapshots.pop(0)
        if verb == "launch":
            return {"ok": True, "rundir": self.workspace + "/runs/" + args[args.index("--run") + 1], "job": {}}
        return {"ok": True}


class FakeRegistry(Registry):
    def __init__(self, path, machine):
        super().__init__(path)
        self.machine = machine

    def get(self, name):
        if name == self.machine.name:
            return self.machine
        return super().get(name)

    versions = {"version_here": "9.9", "version_there": "9.9", "version_matches": True}

    def probe(self, name, *, fresh=False):
        return {"name": name, "state": "idle", "card": "16gb", **self.versions}


def make_api(tmp_path, machine=None):
    registry = FakeRegistry(tmp_path / "machines.toml", machine or FakeMachine({"name": "box", "host": "me@box"}))
    api = Api(tmp_path / "runs", FakeRunner(), token="t", port=1, version="9.9", bind="127.0.0.1",
              machines=registry)
    OPENED.append(api)
    return api, registry


def post(api, path, payload):
    reply = api.handle("POST", path, {}, json.dumps(payload).encode(), "application/json")
    return reply.status, reply.body


def test_api_add_refuses_passwords_and_dry_runs(tmp_path):
    api, registry = make_api(tmp_path)
    status, body = post(api, "/api/machines/add", {"name": "b", "host": "me@b", "password": "x"})
    assert status == 400 and "never stored" in body["message"]
    status, body = post(api, "/api/machines/add", {"name": "b", "host": "me@b", "dry_run": True})
    assert status == 200 and body["dry_run"] and body["row"]["host"] == "me@b"
    assert registry.rows() == []
    status, body = post(api, "/api/machines/add", aws_row())
    assert status == 200 and registry.rows()[0]["kind"] == "aws"
    status, body = post(api, "/api/machines/cloud-a/start", {"dry_run": True})
    assert status == 200 and body["steps"][1]["step"] == "launch" and body["cap_minutes"] == 300
    listed = api.handle("GET", "/api/machines", {}, b"").body
    assert listed["machines"][0]["name"] == "this-computer"


def test_api_starts_a_forecast_on_a_machine(tmp_path):
    machine = FakeMachine({"name": "box", "host": "me@box", "workspace": "/w", "geog_root": "/g"})
    api, _ = make_api(tmp_path, machine)
    draft = {"name": "r1", "source": "hrrr", "lat": 39, "lon": -97, "hours": 2, "machine": "box",
             "render_on": "box", "wait_min": 30}
    status, body = post(api, "/api/create/start", {**draft, "dry_run": True})
    assert status == 200 and body["plan"]["output_root"] == "/w/runs/r1"
    assert body["plan"]["config"]["intent"]["polygon"] == "/w/runs/r1/region.geojson"
    assert body["plan"]["run_options"] == {"geog_root": "/g", "render_products": "none"}
    assert not (tmp_path / "runs" / "r1").exists()
    status, body = post(api, "/api/create/start", draft)
    assert status == 200, body
    verb, args, payload = machine.calls[0]
    assert verb == "launch" and payload["wait_min"] == 30 and "plan.json" in payload["files"]
    rundir = tmp_path / "runs" / "r1"
    assert json.loads((rundir / runs.REMOTE).read_text())["machine"] == "box"
    assert json.loads((rundir / runs.RENDER).read_text())["machine"] == "box"
    assert api.runner.helpers and api.runner.helpers[0][-1] == "r1"
    row = api.runs()["runs"][0]
    assert row["machine"] == "box" and row["status"]["state"] == "running"


def test_follower_mirrors_events_and_the_end(tmp_path):
    machine = FakeMachine({"name": "box", "host": "me@box", "workspace": "/w"})
    root = tmp_path / "runs"
    rundir = root / "r1"
    rundir.mkdir(parents=True)
    (rundir / runs.REMOTE).write_text(json.dumps({"machine": "box", "alive": True, "events_offset": 0}))
    first = b'{"event":"stage_started","sequence":1,"stage":"forecast"}\n'
    last = b'{"event":"completed","sequence":2}\n'
    machine.snapshots = [
        {"ok": True, "alive": True, "job": {"state": "running"}, "events_b64": base64.b64encode(first).decode(),
         "events_size": len(first), "heartbeat": {"status": "running"}, "manifest": {"pid": 1}},
        {"ok": True, "alive": False, "job": {"state": "finished"}, "events_b64": base64.b64encode(last).decode(),
         "events_size": len(first) + len(last), "heartbeat": None, "manifest": None},
    ]
    registry = FakeRegistry(tmp_path / "m.toml", machine)
    follower = Follower(root, "r1", registry)
    assert follower.tick() is True
    assert runs.status(rundir)["state"] == "running"
    assert follower.tick() is False
    assert (rundir / runs.EVENTS).read_bytes() == first + last
    assert machine.calls[1][1][-1] == str(len(first))
    assert runs.status(rundir)["state"] == "finished"


def test_mirror_status_ignores_a_local_pid(tmp_path):
    rundir = tmp_path / "r"
    rundir.mkdir()
    (rundir / runs.REMOTE).write_text(json.dumps({"machine": "box", "alive": False, "ended": True,
                                                  "job": {"state": "refused", "message": "busy"}}))
    (rundir / runs.REMOTE_MANIFEST).write_text(json.dumps({"pid": os.getpid()}))
    info = runs.status(rundir)
    assert info["state"] == "failed" and info["end"]["message"] == "busy" and info["machine"] == "box"


def test_plan_for_keeps_home_relative_workspaces():
    class M:
        row = {"name": "b", "data_dir": "~/d"}
        workspace = "~/gpuwm-machine"

    plan = {"config": {"intent": {"polygon": "C:/x/region.geojson"}}, "output_root": "C:/x"}
    out = plan_for(M(), plan, "r1")
    assert out["output_root"] == "~/gpuwm-machine/runs/r1"
    assert out["run_options"]["data_dir"] == "~/d"


# ------------------------------------------------------------------ review round

def test_a_machine_run_stopped_while_waiting_reads_stopped_and_never_starts_here(tmp_path):
    api, _ = make_api(tmp_path)
    rundir = tmp_path / "runs" / "rv3"
    rundir.mkdir(parents=True)
    (rundir / runs.PLAN).write_text(json.dumps({"output_root": "~/w/runs/rv3", "config": {"intent": {}}}))
    (rundir / runs.REMOTE).write_text(json.dumps({"machine": "n4", "alive": False, "ended": True,
                                                  "job": {"state": "stopped"}}))
    info = runs.status(rundir)
    assert info["state"] == "stopped" and info["end"]["interrupted"]
    (rundir / runs.REMOTE).write_text(json.dumps({"machine": "n4", "alive": False, "job": {}}))
    assert runs.status(rundir)["state"] == "stale"
    for dry in (True, False):
        status, body = post(api, "/api/runs/rv3/start", {"dry_run": dry})
        assert status == 409 and "belongs to n4" in body["message"], body
    assert not api.runner.helpers


def test_a_machine_with_another_gpuwm_version_is_refused(tmp_path):
    machine = FakeMachine({"name": "box", "host": "me@box", "workspace": "/w"})
    api, registry = make_api(tmp_path, machine)
    registry.versions = {"version_here": "9.9", "version_there": "9.8", "version_matches": False}
    draft = {"name": "r1", "source": "hrrr", "lat": 39, "lon": -97, "hours": 2, "machine": "box"}
    status, body = post(api, "/api/create/start", {**draft, "dry_run": True})
    assert status == 409 and "9.8" in body["message"] and "Install" in body["fix"]
    assert not machine.calls and not (tmp_path / "runs" / "r1").exists()
    registry.versions = {"version_here": "9.9", "version_there": "9.9", "version_matches": True}
    status, body = post(api, "/api/create/start", {**draft, "dry_run": True})
    assert status == 200 and body["version_there"] == "9.9"


def _guard():
    import re as _re

    user_data = cloud.providers()["aws"]["user_data"]
    pattern = _re.search(r"pgrep -a?f '([^']+)'", user_data).group(1)
    body = user_data.split("<<'GUARD'\n", 1)[1].split("\nGUARD\n", 1)[0] + "\n"
    return pattern, user_data, body


def test_the_guard_does_not_count_itself_as_activity():
    import re as _re

    pattern, user_data, _ = _guard()
    assert "/usr/local/bin/arwen-cap-guard" in user_data and "gpuwm-guard" not in user_data
    assert not _re.search(pattern, "/bin/bash /usr/local/bin/arwen-cap-guard")
    assert _re.search(pattern, "/w/venv/bin/python -m gpuwm run-plan /w/runs/r/plan.json")
    assert _re.search(pattern, "python3 /w/.agent/machine_agent-0123abcd.py render-loop --job j")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_guard_counts_idle_minutes_and_stops_at_the_cap(tmp_path):
    import subprocess

    _, _, body = _guard()
    guard = tmp_path / "guard.sh"
    guard.write_bytes(body.encode())
    stubs = tmp_path / "bin"
    stubs.mkdir()
    log = tmp_path / "shutdown.log"
    for name, text in (("pgrep", "exit 1"), ("nvidia-smi", "exit 0"),
                       ("shutdown", 'echo "$*" >> "' + log.as_posix() + '"')):
        (stubs / name).write_bytes(("#!/bin/sh\n" + text + "\n").encode())
        (stubs / name).chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    (state / "cap_minutes").write_text("100\n")
    (state / "idle_minutes").write_text("30\n")
    env = {**os.environ, "ARWEN_CAP_GUARD_STATE": state.as_posix(), "ARWEN_CAP_GUARD_TICK_S": "0",
           "ARWEN_CAP_GUARD_ROUNDS": "3", "PATH": stubs.as_posix() + os.pathsep + os.environ.get("PATH", "")}
    bash = shutil.which("bash")
    subprocess.run([bash, guard.as_posix()], env=env, check=True, timeout=60)
    assert (state / "idle_count").read_text().strip() == "3"
    assert (state / "used_minutes").read_text().strip() == "4"  # one for the boot, three ticks
    assert not log.exists()
    # A second boot keeps counting from the disk and stops at the cap.
    (state / "cap_minutes").write_text("6\n")
    subprocess.run([bash, guard.as_posix()], env=env, check=True, timeout=60)
    assert (state / "used_minutes").read_text().strip() == "8"
    assert "spend cap reached" in log.read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_guard_counts_a_render_worker_waiting_for_frames_as_idle(tmp_path):
    import subprocess

    _, _, body = _guard()
    guard = tmp_path / "guard.sh"
    guard.write_bytes(body.encode())
    stubs = tmp_path / "bin"
    stubs.mkdir()
    listing = tmp_path / "processes"
    # pgrep -af prints "pid command line"; the stub prints the scenario's list.
    for name, text in (("pgrep", f'cat "{listing.as_posix()}"'), ("nvidia-smi", "exit 0"),
                       ("shutdown", "exit 0")):
        (stubs / name).write_bytes(("#!/bin/sh\n" + text + "\n").encode())
        (stubs / name).chmod(0o755)
    state = tmp_path / "state"
    env = {**os.environ, "ARWEN_CAP_GUARD_STATE": state.as_posix(), "ARWEN_CAP_GUARD_TICK_S": "0",
           "ARWEN_CAP_GUARD_ROUNDS": "2", "PATH": stubs.as_posix() + os.pathsep + os.environ.get("PATH", "")}
    waiting = "4242 python3 /w/.agent/machine_agent-0123abcd.py render-loop --workspace /w --job j\n"
    drawing = "4243 /w/venv/bin/python -m gpuwm render /w/renders/j/inbox/wrfout_d01 --series\n"
    supervising = "4244 python3 /w/.agent/machine_agent-0123abcd.py supervise --workspace /w --run r\n"
    for processes, idle in ((waiting, "2"), (waiting + drawing, "0"), (supervising, "0")):
        state.mkdir(exist_ok=True)
        (state / "cap_minutes").write_text("100\n")
        (state / "idle_minutes").write_text("30\n")
        listing.write_text(processes)
        subprocess.run([shutil.which("bash"), guard.as_posix()], env=env, check=True, timeout=60)
        assert (state / "idle_count").read_text().strip() == idle, processes


def test_a_start_counts_a_stop_it_did_not_see():
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
    began = (now - timedelta(hours=32)).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = check_row(aws_row(instance_id="i-1", spend_cap_usd=20, price_per_hour_usd=1.86,
                            started_utc=began, cap_minutes_granted=645))
    with pytest.raises(MachineError, match="may have used its spend cap"):
        cloud.plan(row, "start", now=now)
    row["cap_minutes_granted"] = 60  # a one-hour grant: at most $1.86 more
    document = cloud.plan(row, "start", now=now)
    assert document["unsettled_usd"] == 1.86 and document["remaining_usd"] == 18.14
    assert [step["step"] for step in document["steps"]] == [
        "identity", "start", "wait", "address", "host_key", "settle", "set_cap", "check"]


class CloudMachine:
    def __init__(self, registry, name):
        self.registry, self.name = registry, name

    def run(self, words, *, data=b"", timeout=0):
        import subprocess

        self.registry.ssh.append((dict(self.registry.row_of(self.name)), list(words)))
        code, out = self.registry.answers.get(words[0], (0, b""))
        return subprocess.CompletedProcess(words, code, out, b"sudo: a password is required" if code else b"")


class CloudRegistry(Registry):
    def __init__(self, path):
        super().__init__(path)
        self.ssh, self.answers = [], {}

    def row_of(self, name):
        return next(r for r in self.rows() if r["name"] == name)

    def get(self, name):
        return CloudMachine(self, name)

    def probe(self, name, *, fresh=False):
        return {"state": "idle"}


CONSOLE = """[   12.1] cloud-init[900]: ok
ec2: -----BEGIN SSH HOST KEY KEYS-----
ec2: ecdsa-sha2-nistp256 AAAAE2VjZHNh root@ip-10-0-0-1
ec2: ssh-ed25519 AAAAC3NzaC1lZDI1 root@ip-10-0-0-1
ec2: -----END SSH HOST KEY KEYS-----
"""


def cloud_runner(calls):
    import subprocess

    def run(argv):
        calls.append(argv)
        out = {"describe-instances": "ec2-1-2-3-4.compute.amazonaws.com", "get-console-output": CONSOLE}
        return subprocess.CompletedProcess(argv, 0, out.get(argv[2], "{}"), "")

    return run


def test_a_real_start_pins_the_host_key_settles_and_sets_the_cap(tmp_path):
    from gpuwm.gui.machines import Machine

    registry = CloudRegistry(tmp_path / "machines.toml")
    registry.put(check_row(aws_row(instance_id="i-1", spend_cap_usd=20, price_per_hour_usd=2.0,
                                   spent_usd=1.0)))
    registry.answers["cat"] = (0, b"30\n")
    calls = []
    result = cloud.execute(registry, "cloud-a", "start", runner=cloud_runner(calls), sleep=lambda s: None)
    row = registry.row_of("cloud-a")
    known = (tmp_path / "known_hosts").read_text().splitlines()
    assert known == ["i-1 ecdsa-sha2-nistp256 AAAAE2VjZHNh", "i-1 ssh-ed25519 AAAAC3NzaC1lZDI1"]
    assert row["host_key_alias"] == "i-1" and row["known_hosts"] == str(tmp_path / "known_hosts")
    assert row["spent_usd"] == 2.0  # $1 before plus 30 guard minutes at $2/h
    assert row["cap_minutes_granted"] == 540 and row["started_utc"] and row["guard_minutes_charged"] == 30
    set_cap = next(words for _, words in registry.ssh if words[0] == "sudo")
    # The guard's count keeps growing: its cap is the 30 charged plus the 540 granted.
    assert "echo 570 >" in set_cap[-1] and "used_minutes" not in set_cap[-1]
    assert [item["step"] for item in result["ran"]][-3:] == ["settle", "set_cap", "check"]
    # Every SSH call after the key was pinned checks it under the instance's name.
    assert all(r.get("host_key_alias") == "i-1" for r, _ in registry.ssh)
    argv = Machine(dict(row)).ssh_argv("true")
    assert f"UserKnownHostsFile={tmp_path / 'known_hosts'}" in argv and "HostKeyAlias=i-1" in argv


def test_a_start_whose_cap_cannot_be_set_is_stopped_again(tmp_path):
    registry = CloudRegistry(tmp_path / "machines.toml")
    registry.put(check_row(aws_row(instance_id="i-1", spend_cap_usd=20, price_per_hour_usd=2.0)))
    registry.answers["cat"] = (0, b"0\n")
    registry.answers["sudo"] = (1, b"")
    calls = []
    with pytest.raises(MachineError, match="cap could not be set.*stopped again"):
        cloud.execute(registry, "cloud-a", "start", runner=cloud_runner(calls), sleep=lambda s: None)
    assert calls[-1][:3] == ["aws", "ec2", "stop-instances"]
    row = registry.row_of("cloud-a")
    # Nothing is charged twice: the minutes since the settle stay unsettled,
    # bounded at the time up to the stop, for the next settle to read.
    assert "host" not in row and row["spent_usd"] == 0.0 and row["cap_minutes_granted"] <= 2
    assert cloud.unsettled_minutes(row) <= 2.0


def _minutes_ago(minutes):
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_start_stop_start_charges_one_session_once(tmp_path):
    registry = CloudRegistry(tmp_path / "machines.toml")
    registry.put(check_row(aws_row(instance_id="i-1", spend_cap_usd=20, price_per_hour_usd=2.0)))
    registry.answers["cat"] = (0, b"0\n")
    cloud.execute(registry, "cloud-a", "start", runner=cloud_runner([]), sleep=lambda s: None)
    registry.update("cloud-a", started_utc=_minutes_ago(60))  # an hour of use; the guard counted 61
    registry.answers["cat"] = (0, b"61\n")
    cloud.execute(registry, "cloud-a", "stop", runner=cloud_runner([]), sleep=lambda s: None)
    row = registry.row_of("cloud-a")
    assert row["spent_usd"] == 2.03 and "host" not in row and row["cap_minutes_granted"] == 1
    # The guard's count is unchanged while the machine is stopped.
    cloud.execute(registry, "cloud-a", "start", runner=cloud_runner([]), sleep=lambda s: None)
    row = registry.row_of("cloud-a")
    assert row["spent_usd"] == 2.03 and row["guard_minutes_charged"] == 61
    set_cap = [words for _, words in registry.ssh if words[0] == "sudo"][-1]
    assert f"echo {61 + row['cap_minutes_granted']} >" in set_cap[-1]


def test_a_stop_that_cannot_read_the_guard_leaves_a_bound_the_next_start_replaces(tmp_path):
    registry = CloudRegistry(tmp_path / "machines.toml")
    registry.put(check_row(aws_row(instance_id="i-1", spend_cap_usd=20, price_per_hour_usd=2.0,
                                   host="ubuntu@h", guard_minutes_charged=10, started_utc=_minutes_ago(30),
                                   cap_minutes_granted=500)))
    registry.answers["cat"] = (1, b"")
    cloud.execute(registry, "cloud-a", "stop", runner=cloud_runner([]), sleep=lambda s: None)
    row = registry.row_of("cloud-a")
    assert row["spent_usd"] == 0.0 and 31 <= row["cap_minutes_granted"] <= 32  # frozen near 30 min, one for the stop
    registry.answers["cat"] = (0, b"41\n")
    cloud.execute(registry, "cloud-a", "start", runner=cloud_runner([]), sleep=lambda s: None)
    row = registry.row_of("cloud-a")
    assert row["spent_usd"] == 1.03 and row["guard_minutes_charged"] == 41  # 31 guard minutes, once


def test_a_new_machine_is_charged_for_its_boot_before_the_cap_is_set(tmp_path):
    registry = CloudRegistry(tmp_path / "machines.toml")
    registry.put(check_row(aws_row(spend_cap_usd=20, price_per_hour_usd=2.0)))
    registry.answers["cat"] = (0, b"12\n")  # boot to SSH took 12 guard minutes
    result = cloud.execute(registry, "cloud-a", "start", runner=cloud_runner([]), sleep=lambda s: None)
    row = registry.row_of("cloud-a")
    assert [item["step"] for item in result["ran"]][-3:] == ["settle", "set_cap", "check"]
    assert row["spent_usd"] == 0.4 and row["guard_minutes_charged"] == 12
    set_cap = [words for _, words in registry.ssh if words[0] == "sudo"][-1]
    assert f"echo {12 + row['cap_minutes_granted']} >" in set_cap[-1] and "used_minutes" not in set_cap[-1]


def test_terminate_after_stop_charges_only_the_stops_last_minute(tmp_path):
    registry = CloudRegistry(tmp_path / "machines.toml")
    registry.put(check_row(aws_row(instance_id="i-1", spend_cap_usd=20, price_per_hour_usd=60.0,
                                   host="ubuntu@h", started_utc=_minutes_ago(5), cap_minutes_granted=100)))
    registry.answers["cat"] = (0, b"5\n")
    cloud.execute(registry, "cloud-a", "stop", runner=cloud_runner([]), sleep=lambda s: None)
    assert registry.row_of("cloud-a")["spent_usd"] == 5.0
    cloud.execute(registry, "cloud-a", "terminate", runner=cloud_runner([]), sleep=lambda s: None)
    row = registry.row_of("cloud-a")
    assert 5.0 <= row["spent_usd"] <= 6.0 and "instance_id" not in row and "guard_minutes_charged" not in row


def test_render_inbox_keeps_only_each_domains_newest_relayed_frame(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    own = tmp_path / "run" / "wrfout_d01_2026-09-25_00.00.00"
    own.parent.mkdir()
    own.write_bytes(b"x")
    names = ["wrfout_d01_2026-09-25_01.00.00", "wrfout_d01_2026-09-25_02.00.00"]
    for name in names:
        (inbox / name).write_bytes(b"y" * 10)
        (inbox / f"{name}.frame.json").write_text(json.dumps({"path": str(inbox / name)}))
    (inbox / "own.frame.json").write_text(json.dumps({"path": str(own)}))
    drawn = {str(inbox / name) for name in names} | {str(own)}
    freed = agent.prune_inbox(inbox, keep={str(inbox / names[1])}, drawn=drawn)
    assert freed == 10 and not (inbox / names[0]).exists() and (inbox / names[1]).exists()
    assert own.exists() and not (inbox / f"{names[0]}.frame.json").exists()
    agent.prune_inbox(inbox, keep=set(), drawn=drawn)
    assert not (inbox / names[1]).exists() and own.exists()


def test_render_inbox_never_deletes_a_frame_that_was_not_drawn(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    first, second = (inbox / f"wrfout_d01_2026-09-25_0{hour}.00.00" for hour in (1, 2))
    for frame in (first, second):
        frame.write_bytes(b"y")
        (inbox / f"{frame.name}.frame.json").write_text(json.dumps({"path": str(frame)}))
    agent.prune_inbox(inbox, keep={str(first)}, drawn={str(first)})
    assert first.exists() and second.exists() and (inbox / f"{second.name}.frame.json").exists()


def test_render_loop_draws_a_frame_fed_while_a_batch_draws(tmp_path, monkeypatch):
    import io
    import subprocess

    workspace = tmp_path / "w"
    folder = agent.render_dir(workspace, "job1")
    inbox = folder / "inbox"
    inbox.mkdir(parents=True)
    frames = [tmp_path / f"wrfout_d01_2026-09-25_0{hour}.00.00" for hour in (1, 2)]
    for frame in frames:
        frame.write_bytes(b"y")

    def feed(frame):
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps([str(frame)])))
        agent.cmd_render_feed(type("A", (), {"workspace": str(workspace), "job": "job1"})())
        # A relayed copy lands in the inbox itself; stand the copy in for the relay.
        record = inbox / f"{frame.name}.frame.json"
        copy = inbox / frame.name
        shutil.copy(frame, copy)
        record.write_text(json.dumps({"path": str(copy)}))

    drawn = []

    def fake_run(argv, **kwargs):
        batch = argv[argv.index("render") + 1:argv.index("--series")]
        drawn.append([Path(item).name for item in batch])
        if len(drawn) == 1:
            feed(frames[1])        # fed while the first batch draws
        else:
            agent.cmd_render_end(type("A", (), {"workspace": str(workspace), "job": "job1"})())
        return subprocess.CompletedProcess(argv, 0)

    agent.write_json(folder / "render-job.json", {"python": "python", "out": str(tmp_path / "out"), "done": []})
    feed(frames[0])
    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    monkeypatch.setattr(agent, "RENDER_IDLE_S", 2.0)   # a lost frame ends the loop, not the test run
    assert agent.cmd_render_loop(type("A", (), {"workspace": str(workspace), "job": "job1"})()) == 0
    assert drawn == [[frames[0].name], [frames[1].name]]
    job = agent.read_json(folder / "render-job.json")
    assert job["state"] == "finished" and job["rendered"] == 2
    assert not list(inbox.glob("wrfout_*"))


def test_install_script_quotes_paths_with_spaces(tmp_path, monkeypatch):
    import io
    import shlex as _shlex

    workspace = tmp_path / "my work"
    wheel = tmp_path / "wheels dir" / "gpuwm-9.9-cp312-linux_x86_64.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"w")
    monkeypatch.setattr(agent, "spawn", lambda *a, **k: 0)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"wheels": [str(wheel)], "extra": "gpu-cu13"})))
    agent.cmd_install(type("A", (), {"workspace": str(workspace)})())
    # The pip steps run from their own script (install-steps.sh), so a
    # venv made on 3.14t can be made again on python3 and the steps rerun.
    script = (workspace / "install" / "install-steps.sh").read_text().splitlines()
    last = _shlex.split(script[-1], posix=True)
    assert last[0] == str(workspace / "venv" / "bin" / "python") and last[-1] == f"{wheel}[gpu-cu13]"
    driver = (workspace / "install" / "install.sh").read_text()
    venv_lines = [_shlex.split(line, posix=True) for line in driver.splitlines() if " -m venv " in line]
    assert venv_lines and all(words[-1] == str(workspace / "venv") for words in venv_lines)
    assert _shlex.quote(str(workspace / "install" / "install-steps.sh")) in driver


def test_this_computer_draws_with_the_python_this_page_runs_on():
    import sys

    from gpuwm.gui.machines import Machine, local_row

    assert Machine(local_row()).python == sys.executable
    assert Machine({"name": "box", "host": "me@box", "workspace": "/w"}).python == "/w/venv/bin/python"
    assert Machine({"name": "box", "host": "me@box", "python": "/opt/py"}).python == "/opt/py"


def test_a_draw_that_failed_is_asked_for_again_and_one_going_is_kept(tmp_path):
    from gpuwm.gui.remote_runs import request_render

    rundir = tmp_path / "r1"
    rundir.mkdir()
    first = request_render(rundir, "box")
    assert first["state"] == "requested"
    (rundir / runs.RENDER).write_text(json.dumps({**first, "state": "rendering", "fed": ["a"]}))
    assert request_render(rundir, "box")["state"] == "rendering"
    for ended in ("failed", "stopped", "finished"):
        (rundir / runs.RENDER).write_text(json.dumps({**first, "state": ended, "fed": ["a"], "message": "x"}))
        again = request_render(rundir, "box")
        assert again["state"] == "requested" and again["fed"] == [], ended


def test_pictures_drawn_on_this_computer_come_back_in_their_folders(tmp_path):
    from gpuwm.gui.machines import Machine, local_row, relay

    out = tmp_path / "worker-out"
    picture = out / "d01-3km" / "2m_temperature" / "2026-09-25" / "f001.png"
    picture.parent.mkdir(parents=True)
    picture.write_bytes(b"png-bytes")
    here = Machine(local_row())
    target = tmp_path / "run" / "render-this-computer"
    moved = relay(here, here, ["d01-3km/2m_temperature/2026-09-25/f001.png"], str(target), base=str(out))
    assert moved == len(b"png-bytes")
    assert (target / "d01-3km" / "2m_temperature" / "2026-09-25" / "f001.png").read_bytes() == b"png-bytes"
    frame = tmp_path / "wrfout_d01"
    frame.write_bytes(b"frame")
    inbox = tmp_path / "inbox"
    relay(here, here, [str(frame)], str(inbox))
    assert (inbox / "wrfout_d01").read_bytes() == b"frame"


def test_a_worker_reads_alive_while_it_runs_and_dead_once_it_exits():
    # On Windows os.kill(pid, 0) sends a console event and succeeds for an exited process, so a render worker
    # that died read as alive forever and its draw never ended.
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert agent.pid_alive(child.pid)
        assert agent.pid_alive(child.pid)
        assert child.poll() is None
    finally:
        child.kill()
        child.wait()
    assert not agent.pid_alive(child.pid)


def test_a_worker_that_died_says_why_in_one_line_without_its_traceback():
    from gpuwm.gui.remote_runs import _last_lines

    log = ("render: starting\nTraceback (most recent call last):\n  File \"agent.py\", line 693, in loop\n"
           "    code = run(argv)\n           ^^^^^^^^^\nFileNotFoundError: [WinError 2] The system cannot find the file")
    assert _last_lines(log) == "[WinError 2] The system cannot find the file"
    assert _last_lines(log.split("\n", 2)[2]) == "[WinError 2] The system cannot find the file"
    assert _last_lines("one\ntwo") == "one two"


def test_a_render_worker_lists_the_map_record_beside_its_pictures(tmp_path):
    from types import SimpleNamespace

    folder = tmp_path / "renders" / "r1--box"
    out = folder / "run-1"
    picture = out / "d01-3km" / "radar" / "2026-09-25" / "f000.png"
    picture.parent.mkdir(parents=True)
    picture.write_bytes(b"png")
    (out / "render-georef.json").write_text('{"panels": {}}')
    (folder / "render-job.json").write_text(json.dumps({"out": str(out), "pid": None, "state": "finished"}))
    listing = agent.cmd_render_list(SimpleNamespace(workspace=str(tmp_path), job="r1--box"))
    assert listing["files"] == [["d01-3km/radar/2026-09-25/f000.png", 3]]
    assert listing["manifests"] == [["render-georef.json", len('{"panels": {}}')]]


def _render_args(workspace, **extra):
    from types import SimpleNamespace

    return SimpleNamespace(workspace=str(workspace), job="job1", python="python", **extra)


def _render_start(workspace, monkeypatch, fresh):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"run": "r1", "fresh": fresh})))
    return agent.cmd_render_start(_render_args(workspace))


def test_drawing_again_waits_for_its_own_end_not_the_last_attempts(tmp_path, monkeypatch):
    # A finished draw leaves its END in the inbox. Drawing again with fresh: true started a worker that read that
    # END and finished at once, before any frame was fed: the page said Finished with no new pictures.
    import subprocess

    workspace = tmp_path / "w"
    folder = agent.render_dir(workspace, "job1")
    frame = tmp_path / "wrfout_d01_2026-09-25_01.00.00"
    frame.write_bytes(b"y")
    monkeypatch.setattr(agent, "spawn", lambda *a, **k: 0)
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    monkeypatch.setattr(agent, "RENDER_IDLE_S", 2.0)
    drawn = []

    def fake_run(argv, **kwargs):
        drawn.append([Path(item).name for item in argv[argv.index("render") + 1:argv.index("--series")]])
        agent.cmd_render_end(_render_args(workspace))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    # the first attempt: ended with nothing to draw
    _render_start(workspace, monkeypatch, fresh=False)
    agent.cmd_render_end(_render_args(workspace))
    assert agent.cmd_render_loop(_render_args(workspace)) == 0
    assert agent.read_json(folder / "render-job.json")["state"] == "finished"
    # the second: its first frame is fed after it starts, and its own end comes after that frame is drawn
    job = _render_start(workspace, monkeypatch, fresh=True)["job"]
    assert not (folder / "inbox" / "END").exists()
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps([str(frame)])))
    agent.cmd_render_feed(_render_args(workspace))
    assert agent.cmd_render_loop(_render_args(workspace)) == 0
    assert drawn == [[frame.name]]
    assert agent.read_json(folder / "render-job.json")["rendered"] == 1 and job["attempt"] == 2


def test_an_end_for_one_attempt_never_ends_the_next(tmp_path, monkeypatch):
    workspace = tmp_path / "w"
    folder = agent.render_dir(workspace, "job1")
    monkeypatch.setattr(agent, "spawn", lambda *a, **k: 0)
    _render_start(workspace, monkeypatch, fresh=False)
    agent.cmd_render_end(_render_args(workspace))
    assert agent._ended_attempt(folder) == 1
    # a worker still winding down on attempt 1 when Draw is pressed again takes attempt 2 over, all frames again
    # (a live worker's record names its process by PID and start, as every agent record does)
    agent.update_json(folder / "render-job.json", pid=os.getpid(), pid_start=agent.process_start(os.getpid()),
                      done=["a"], rendered=1)
    job = _render_start(workspace, monkeypatch, fresh=True)["job"]
    assert job["attempt"] == 2 and job["done"] == [] and job["state"] == "waiting"
    assert agent._ended_attempt(folder) is None
    # an END written from before attempts were numbered still ends a worker (a new attempt clears it first)
    (folder / "inbox" / "END").write_text("2026-09-26T00:00:00Z\n")
    assert agent._ended_attempt(folder) == -1


def test_a_fresh_draw_pressed_while_a_batch_draws_gets_every_frame_and_its_own_products(tmp_path, monkeypatch):
    # Draw again is pressed while the worker is still drawing the last attempt's final batch. The frames are
    # relayed and fed again during that batch. The worker must not record them against the old attempt: that
    # pruned the relayed copy of every frame but the newest, so the new attempt drew one frame of two, with the
    # old attempt's products.
    import io
    import subprocess

    workspace = tmp_path / "w"
    folder = agent.render_dir(workspace, "job1")
    names = ["wrfout_d01_2026-09-25_01.00.00", "wrfout_d01_2026-09-25_02.00.00"]
    monkeypatch.setattr(agent, "spawn", lambda *a, **k: os.getpid())
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    monkeypatch.setattr(agent, "RENDER_IDLE_S", 2.0)

    def relay_and_feed():
        for item in names:
            (folder / "inbox" / item).write_bytes(b"y")
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(names)))
        agent.cmd_render_feed(_render_args(workspace))

    batches = []

    def fake_run(argv, **kwargs):
        frames = [Path(item).name for item in argv[argv.index("render") + 1:argv.index("--series")]]
        products = argv[argv.index("--products") + 1] if "--products" in argv else None
        batches.append((frames, products))
        if len(batches) == 1:
            monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"run": "r1", "fresh": True,
                                                                      "products": "radar"})))
            assert agent.cmd_render_start(_render_args(workspace))["job"]["attempt"] == 2
            relay_and_feed()
            agent.cmd_render_end(_render_args(workspace))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    _render_start(workspace, monkeypatch, fresh=False)
    relay_and_feed()
    agent.cmd_render_end(_render_args(workspace))
    assert agent.cmd_render_loop(_render_args(workspace)) == 0
    assert batches == [(names, None), (names, "radar")]
    job = agent.read_json(folder / "render-job.json")
    assert job["state"] == "finished" and job["attempt"] == 2 and job["rendered"] == 2
    assert [batch["frames"] for batch in job["batches"]] == [names]


def test_a_draw_again_that_lands_as_the_worker_finishes_still_gets_a_worker(tmp_path, monkeypatch):
    # Draw again lands in the instant the worker records its finish, its process still alive. If the start reads
    # "worker alive" and hands it the new attempt while the worker has already decided to finish, the worker exits
    # and the new attempt waits for nobody: nothing is drawn and the page waits for ever. The start and the
    # worker's finish take one lock, so the start either hands the attempt over before the worker decides or, as
    # here, finds the worker done and starts its own.
    import io
    import threading

    workspace = tmp_path / "w"
    folder = agent.render_dir(workspace, "job1")
    spawned = []
    monkeypatch.setattr(agent, "spawn", lambda *a, **k: spawned.append(a) or os.getpid())
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    _render_start(workspace, monkeypatch, fresh=False)
    agent.cmd_render_end(_render_args(workspace))
    real_update = agent.update_json
    started, joined = [], []

    def draw_again():
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"run": "r1", "fresh": True})))
        started.append(agent.cmd_render_start(_render_args(workspace)))

    def update(path, **fields):
        if fields.get("state") == "finished" and not joined:
            thread = threading.Thread(target=draw_again)
            thread.start()
            thread.join(0.5)
            joined.append(thread)
        return real_update(path, **fields)

    monkeypatch.setattr(agent, "update_json", update)
    assert agent.cmd_render_loop(_render_args(workspace)) == 0
    joined[0].join(10.0)
    assert started and "already" not in started[0], started
    assert len(spawned) == 2
    job = agent.read_json(folder / "render-job.json")
    assert job["attempt"] == 2 and job["state"] == "starting" and job["done"] == []
    # a finished worker's job is not a live worker's, whatever its process is doing
    assert not agent._worker_live({**job, "state": "finished"})
    assert agent._worker_live(job)


def test_a_forecast_queued_for_a_machine_starts_there_when_its_card_is_idle(tmp_path):
    machine = FakeMachine({"name": "box", "host": "me@box", "workspace": "/w", "geog_root": "/g"})
    api, registry = make_api(tmp_path, machine)
    state = {"now": "running"}
    registry.probe = lambda name, *, fresh=False: {"name": name, "state": state["now"], "card": "16gb",
                                                   **registry.versions}
    draft = {"name": "q1", "source": "hrrr", "lat": 39, "lon": -97, "hours": 2, "machine": "box",
             "render_on": "box", "queue": True}
    status, body = post(api, "/api/create/start", draft)
    assert status == 200 and body["queued"] and body["place"] == 1, body
    rundir = tmp_path / "runs" / "q1"
    assert (rundir / runs.QUEUED).is_file() and machine.calls == []
    assert api.runs()["runs"][0]["status"]["state"] == "queued"

    # The machine's card is busy: the forecast waits and says why.
    api.queue._probe_remote("box")
    assert api.queue.tick() == [] and machine.calls == []
    assert "box" in json.loads((rundir / runs.QUEUED).read_text())["waiting"]

    # Idle: it starts on that machine, in the folder the queue wrote, and leaves the queue.
    state["now"] = "idle"
    api.queue._probe_remote("box")
    assert api.queue.tick() == ["q1"]
    verb, _, payload = machine.calls[0]
    assert verb == "launch" and "plan.json" in payload["files"]
    assert not (rundir / runs.QUEUED).exists()
    assert json.loads((rundir / runs.REMOTE).read_text())["machine"] == "box"
    assert api.queue.order() == []


def test_a_queued_machine_start_that_fails_keeps_its_place_and_is_tried_again(tmp_path):
    from gpuwm.gui.machines import MachineError

    machine = FakeMachine({"name": "box", "host": "me@box", "workspace": "/w", "geog_root": "/g"})
    api, registry = make_api(tmp_path, machine)
    for name in ("q1", "q2"):
        status, body = post(api, "/api/create/start", {"name": name, "source": "hrrr", "lat": 39, "lon": -97,
                                                       "hours": 2, "machine": "box", "render_on": "box",
                                                       "queue": True})
        assert status == 200, body
    real = machine.call

    def dropped(verb, *args, **kwargs):
        if verb == "launch":
            raise MachineError("box dropped the connection.", "Check that box is on.")
        return real(verb, *args, **kwargs)

    machine.call = dropped
    api.queue._probe_remote("box")
    assert api.queue.tick() == []
    rundir = tmp_path / "runs" / "q1"
    assert api.queue.order() == ["q1", "q2"]
    assert "dropped the connection" in json.loads((rundir / runs.QUEUED).read_text())["waiting"]
    assert sorted(path.name for path in rundir.iterdir()) == sorted(["commands.log", runs.QUEUED])
    machine.call = real
    assert api.queue.tick() == ["q1"]
    assert api.queue.order() == ["q2"]


# ------------------------------------------------------------------ an install that has not put its requirements in

WHEEL = "/w/install/wheels/gpuwm-9.9-cp312-cp312-linux_x86_64.whl"


class ProbedMachine(FakeMachine):
    """A machine whose probe answers as machine_agent's does, with its interpreter and last install as set."""

    kind = "ssh"

    def __init__(self, row):
        super().__init__(row)
        self.interpreter = {"python": self.python, "version": "9.9", "missing": []}
        self.install = None

    def call(self, verb, *args, payload=None, timeout=0):
        if verb == "probe":
            return {"ok": True, "state": "idle", "detail": "idle",
                    "cards": {"devices": [{"name": "card", "memory_total_mib": 16384}]},
                    "gpuwm": dict(self.interpreter), "install": self.install}
        return super().call(verb, *args, payload=payload, timeout=timeout)


def probed_api(tmp_path, monkeypatch):
    from gpuwm.gui import machines as machines_module

    monkeypatch.setattr(machines_module, "local_version", lambda: "9.9")
    machine = ProbedMachine({"name": "box", "host": "me@box", "workspace": "/w", "geog_root": "/g"})
    api, registry = make_api(tmp_path, machine)
    # The real probe and check, so the doors read what the Machines row reads.
    registry.probe = lambda name, *, fresh=False: Registry.probe(registry, name, fresh=fresh)
    return api, machine


def test_the_interpreter_probe_names_the_requirements_gpuwm_declares_and_lacks(tmp_path):
    # The installer puts gpuwm in before its requirements; the probe run in the machine's Python must say which
    # of them are not there, not only that gpuwm imports and which version it is.
    import subprocess
    import sys

    site = tmp_path / "site"
    (site / "gpuwm").mkdir(parents=True)
    (site / "gpuwm" / "__init__.py").write_text('__version__ = "9.9"\n', encoding="utf-8")
    for name, lines in (("gpuwm-9.9", ["Name: gpuwm", "Version: 9.9", "Requires-Dist: numpy>=2.0",
                                       "Requires-Dist: arwen-absent-requirement>=1.0",
                                       "Requires-Dist: gpuwm-data==9.9",
                                       'Requires-Dist: arwen-absent-gpu-extra>=13; extra == "gpu-cu13"']),
                        ("gpuwm_data-9.9", ["Name: gpuwm-data", "Version: 9.9"])):
        (site / f"{name}.dist-info").mkdir()
        (site / f"{name}.dist-info" / "METADATA").write_text(
            "\n".join(["Metadata-Version: 2.1", *lines]) + "\n", encoding="utf-8")
    done = subprocess.run([sys.executable, "-c", agent.VERSION_PROBE], capture_output=True, text=True, cwd=tmp_path,
                          env={**os.environ, "PYTHONPATH": str(site)}, timeout=60)
    document = json.loads(done.stdout.strip().splitlines()[-1])
    assert document["version"] == "9.9"
    assert document["missing"] == ["arwen-absent-requirement"]
    # gpuwm_version carries the list through from the interpreter it asks.
    assert isinstance(agent.gpuwm_version(sys.executable)["missing"], list)


@pytest.mark.parametrize("case", ["installing", "missing", "other-python", "installed"])
def test_a_matching_version_is_ready_only_once_its_requirements_are_in(monkeypatch, case):
    from gpuwm.gui import machines as machines_module

    monkeypatch.setattr(machines_module, "local_version", lambda: "9.9")
    machine = ProbedMachine({"name": "box", "host": "me@box", "workspace": "/w"})
    install = {"state": "installing", "python": "/w/venv/bin/python", "wheels": [WHEEL]}
    if case == "installing":
        machine.install = install
    elif case == "missing":
        machine.install = {**install, "state": "failed"}
        machine.interpreter["missing"] = ["numpy", "scipy"]
    elif case == "other-python":
        # A machine set to a Python of its own keeps working while its workspace venv is installed.
        machine.install = install
        machine.interpreter["python"] = "/opt/python"
    else:
        machine.install = {**install, "state": "installed"}
    row = machines_module.check(machine)
    assert row["version_matches"]
    if case == "installing":
        assert row["detail"] == "idle; installing gpuwm 9.9"
        assert "offer" not in row
        assert row.get("not_ready", {}).get("state") == "installing"
    elif case == "missing":
        assert row["detail"] == "idle; the install of gpuwm 9.9 did not finish"
        assert "missing numpy, scipy" in (row.get("offer") or {}).get("words", "")
        assert row["offer"]["action"] == "install"
        assert row["not_ready"] == {"state": "incomplete", "missing": ["numpy", "scipy"],
                                    "label": "missing numpy, scipy"}
    elif case == "other-python":
        assert row["detail"] == "idle; installing gpuwm 9.9"
        assert row.get("not_ready") is None
    else:
        assert row.get("not_ready") is None and "offer" not in row and row["detail"] == "idle"


@pytest.mark.parametrize("case", ["installing", "missing"])
def test_no_draw_or_forecast_starts_on_a_machine_whose_requirements_are_not_in(tmp_path, monkeypatch, case):
    # The defect: while the install was still putting numpy and the rest in, the row said "same as here", Draw
    # answered 200 and the draw then failed there with "No module named 'numpy'".
    api, machine = probed_api(tmp_path, monkeypatch)
    machine.install = {"state": "installing" if case == "installing" else "failed",
                       "python": "/w/venv/bin/python", "wheels": [WHEEL]}
    if case == "missing":
        machine.interpreter["missing"] = ["numpy"]
    folder = tmp_path / "runs" / "r1"
    folder.mkdir(parents=True)
    (folder / runs.PLAN).write_text(json.dumps({"name": "r1", "config": {"intent": {}}}), encoding="utf-8")
    words = "still installing" if case == "installing" else "missing numpy"

    status, body = post(api, "/api/runs/r1/render", {"machine": "box"})
    assert status == 409 and words in body["message"], body
    assert not (folder / runs.RENDER).exists() and not api.runner.helpers
    draft = {"name": "r2", "source": "hrrr", "lat": 39, "lon": -97, "hours": 2, "machine": "box",
             "render_on": "box"}
    status, body = post(api, "/api/create/start", draft)
    assert status == 409 and words in body["message"], body
    assert not (tmp_path / "runs" / "r2").exists()
    assert [verb for verb, _, _ in machine.calls if verb != "probe"] == []

    # Once the install has put everything in, the same Draw is taken.
    machine.install = {**machine.install, "state": "installed"}
    machine.interpreter["missing"] = []
    status, body = post(api, "/api/runs/r1/render", {"machine": "box"})
    assert status == 200, body
    assert json.loads((folder / runs.RENDER).read_text())["machine"] == "box"


def test_a_forecast_queued_while_its_machine_installs_waits_and_starts_once_it_is_ready(tmp_path, monkeypatch):
    api, machine = probed_api(tmp_path, monkeypatch)
    machine.install = {"state": "installing", "python": "/w/venv/bin/python", "wheels": [WHEEL]}
    status, body = post(api, "/api/create/start", {"name": "q1", "source": "hrrr", "lat": 39, "lon": -97,
                                                   "hours": 2, "machine": "box", "render_on": "box",
                                                   "queue": True})
    assert status == 200 and body["queued"], body
    rundir = tmp_path / "runs" / "q1"
    api.queue._probe_remote("box")
    assert api.queue.tick() == []
    assert "still installing" in json.loads((rundir / runs.QUEUED).read_text())["waiting"]
    assert [verb for verb, _, _ in machine.calls if verb == "launch"] == []
    machine.install = {**machine.install, "state": "installed"}
    assert api.queue.tick() == ["q1"]
    assert [verb for verb, _, _ in machine.calls if verb == "launch"] == ["launch"]


def test_a_queued_machine_start_whose_log_line_fails_leaves_the_line_and_is_not_started_again(tmp_path, monkeypatch):
    # The machine accepted the forecast; the commands.log line written after it could not be.  That is not a
    # failed start: put back in line, the same forecast was launched on the machine a second time.
    from gpuwm.gui import api as gui_api

    machine = FakeMachine({"name": "box", "host": "me@box", "workspace": "/w", "geog_root": "/g"})
    api, registry = make_api(tmp_path, machine)
    status, body = post(api, "/api/create/start", {"name": "q1", "source": "hrrr", "lat": 39, "lon": -97,
                                                   "hours": 2, "machine": "box", "render_on": "box",
                                                   "queue": True})
    assert status == 200, body

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied", "commands.log")

    monkeypatch.setattr(gui_api, "log_command", denied)
    api.queue._probe_remote("box")
    first = api.queue.tick()
    left = api.queue.order()
    api.queue.tick()
    assert [verb for verb, _, _ in machine.calls if verb == "launch"] == ["launch"],         "the accepted start was launched on the machine a second time"
    assert first == ["q1"] and left == [] and not (tmp_path / "runs" / "q1" / runs.QUEUED).exists()


def test_a_start_the_machine_took_whose_record_here_cannot_be_written_is_said_started(tmp_path, monkeypatch):
    # The machine has the forecast; its record in this computer's folder could not be written.  Read as a failed
    # start, the page removed the folder of a forecast running there and said it had not started.
    from gpuwm.gui import remote_runs

    machine = FakeMachine({"name": "box", "host": "me@box", "workspace": "/w", "geog_root": "/g"})
    api, _ = make_api(tmp_path, machine)
    real = remote_runs.write_json

    def full(path, document, *args, **kwargs):
        if Path(path).name == remote_runs.MACHINE:
            raise OSError(28, "No space left on device", str(path))
        return real(path, document, *args, **kwargs)

    monkeypatch.setattr(remote_runs, "write_json", full)
    status, body = post(api, "/api/create/start", {"name": "r1", "source": "hrrr", "lat": 39, "lon": -97,
                                                   "hours": 2, "machine": "box", "render_on": "box"})
    assert status == 200, body
    assert "could not keep its record" in body["message"] and "/w/runs/r1" in body["message"]
    assert (tmp_path / "runs" / "r1" / runs.PLAN).is_file()
    assert [verb for verb, _, _ in machine.calls if verb == "launch"] == ["launch"]
    assert not api.runner.helpers
