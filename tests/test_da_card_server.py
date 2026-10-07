"""Card servers: one long-lived member worker per card, same protocol.

CPU only.  THE BREAKAGE THIS PREVENTS: on box E's 9 km CONUS run every
packed member was a fresh process that imported the engine, opened a CUDA
context and re-read and re-hashed the 4 GB prepared cache before its first
step: about 40 s of host work around 25 s of stepping, 26 percent of
card-time busy over three cycles.  A card server runs the members' jobs in
order in one process; these tests hold its protocol and the scheduler's
use of it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from gpuwm.da import member_transport as transport
from gpuwm.da import member_wave as wave
from test_da_member_wave import FAKE_WORKER, jobs, uuid

FAKE_SERVER = r'''
import json, os, sys, time, runpy
from pathlib import Path
queue = Path(sys.argv[-1]); index = 0
while True:
    job = queue / f"job-{index:05d}.json"
    if (queue / "stop").exists() and not job.exists(): sys.exit(0)
    if not job.exists(): time.sleep(0.02); continue
    doc = json.loads(job.read_text())
    task = json.loads(Path(doc["request"]).read_text())
    with open(doc["log"], "a") as log: log.write(f"served by {os.getpid()}\n")
    sys.argv = ["fake_worker", doc["request"]]
    rc = 0
    try:
        runpy.run_path(SCRIPT, run_name="__main__")
    except SystemExit as stop:
        rc = int(stop.code or 0)
    (queue / f"job-{index:05d}.done").write_text(json.dumps({"rc": rc}))
    index += 1
    if rc: sys.exit(1)
'''


def server_jobs(tmp_path, behaviors):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER)
    server = tmp_path / "fake_server.py"
    server.write_text(FAKE_SERVER.replace("SCRIPT", repr(str(worker))))
    roster = jobs(tmp_path, len(behaviors))
    for job, behavior in zip(roster, behaviors):
        spec = tmp_path / f"task{job['member']}.json"
        spec.write_text(json.dumps(dict(job, behavior=behavior)))
        job["argv"] = [sys.executable, "-m", "x", "--request", str(spec),
                       "--request-sha256", job["request_hash"],
                       "--result", job["result_path"]]
    return roster, [sys.executable, str(server)]


def test_members_run_in_one_server_per_card_and_clear_the_barrier(tmp_path):
    roster, argv = server_jobs(tmp_path, ["ok"] * 5)
    servers = {u: wave.CardServer(u, tmp_path / "servers" / u, argv=argv)
               for u in (uuid(), uuid(1))}
    try:
        results = wave.run_wave(roster, [uuid(), uuid(1)], 1, tmp_path / "wave",
                                30, host_available_bytes=251 * wave.GIB,
                                servers=servers)
    finally:
        for server in servers.values():
            server.close(timeout_seconds=10)
    assert [row["member"] for row in results] == list(range(5))
    pids = {Path(tmp_path / "wave" / f"member-{m:03d}.log").read_text().split()[-1]
            for m in range(5)}
    assert pids == {str(s.process.pid) for s in servers.values()}
    assert all(s.process.returncode == 0 for s in servers.values())


def test_a_failed_server_job_fails_the_barrier_and_ends_its_server(tmp_path):
    roster, argv = server_jobs(tmp_path, ["ok", "fail"])
    server = wave.CardServer(uuid(), tmp_path / "servers", argv=argv)
    try:
        with pytest.raises(wave.MemberWaveFailed, match="member forecast failed"):
            wave.run_wave(roster, [uuid()], 1, tmp_path / "wave", 30,
                          host_available_bytes=251 * wave.GIB,
                          servers={uuid(): server})
    finally:
        server.close(timeout_seconds=10)
    assert server.process.poll() is not None


def test_serve_runs_jobs_in_order_with_their_own_logs(tmp_path, monkeypatch):
    queue = tmp_path / "queue"
    queue.mkdir()
    seen = []

    def fake_run_job(request, digest, result, *, served_before=0, prefetched=None):
        os.write(1, ("job %s after %d" % (request, served_before)).encode() + bytes([10]))
        seen.append((request, served_before))
        if request == "bad":
            raise RuntimeError("member leg failed")
        return 0

    monkeypatch.setattr(transport, "run_job", fake_run_job)
    for index, name in enumerate(["a", "b", "bad", "never"]):
        log = tmp_path / f"{name}.log"
        log.write_text("")
        (queue / f"job-{index:05d}.json").write_text(json.dumps({
            "schema": transport.SERVER_JOB_SCHEMA, "request": name,
            "request_sha256": "0" * 64, "result": "r", "log": str(log)}))
    assert transport.serve(queue) == 1      # ends at the failed job
    assert seen == [("a", 0), ("b", 1), ("bad", 2)]
    assert json.loads((queue / "job-00001.done").read_text())["rc"] == 0
    bad = json.loads((queue / "job-00002.done").read_text())
    assert bad["rc"] == 1 and "member leg failed" in bad["error"]
    assert not (queue / "job-00003.done").exists()
    assert "job b after 1" in (tmp_path / "b.log").read_text()


def test_a_packed_card_runs_one_server_per_slot(tmp_path):
    """Two members per card: two servers, so the two run side by side.

    THE BREAKAGE: one server per card queued a packed card's members on one
    process, so the slot scheduler counted two running while they ran one
    after another."""
    roster, argv = server_jobs(tmp_path, ["ok"] * 4)
    servers = wave.start_card_servers([uuid()], tmp_path / "servers",
                                      per_card=2, argv=argv)
    pool = servers[uuid()]
    try:
        results = wave.run_wave(roster, [uuid()], 2, tmp_path / "wave", 30,
                                host_available_bytes=251 * wave.GIB,
                                servers=servers)
    finally:
        pool.close(timeout_seconds=10)
    assert [row["member"] for row in results] == list(range(4))
    pids = {Path(tmp_path / "wave" / f"member-{m:03d}.log").read_text().split()[-1]
            for m in range(4)}
    assert pids == {str(s.process.pid) for s in pool.servers} and len(pids) == 2


def test_the_occupying_control_takes_its_own_server_slot(tmp_path):
    """The control occupies one slot of the last card; the members go to the
    card's other server, not into a queue behind the control."""
    roster, argv = server_jobs(tmp_path, ["ok"] * 3)
    servers = wave.start_card_servers([uuid()], tmp_path / "servers",
                                      per_card=2, argv=argv)
    pool = servers[uuid()]
    control = dict(roster[0], result_path=str(tmp_path / "control" / "result.json"),
                   output_root=str(tmp_path / "control"))
    spec = tmp_path / "taskcontrol.json"
    spec.write_text(json.dumps(dict(control, behavior="ok")))
    control["argv"] = [sys.executable, "-m", "x", "--request", str(spec),
                       "--request-sha256", control["request_hash"],
                       "--result", control["result_path"]]
    try:
        owned = wave.OwnedJob(control, uuid(), tmp_path / "control.log", server=pool)
        busy = [s for s in pool.servers if not s.idle()]
        assert len(busy) == 1
        wave.run_wave(roster, [uuid()], 2, tmp_path / "wave", 30,
                      host_available_bytes=251 * wave.GIB,
                      occupied={uuid(): owned}, servers=servers)
        assert owned.poll() == 0
    finally:
        pool.close(timeout_seconds=10)
    member_pids = {Path(tmp_path / "wave" / f"member-{m:03d}.log").read_text().split()[-1]
                   for m in range(3)}
    control_pid = (tmp_path / "control.log").read_text().split()[-1]
    assert str(busy[0].process.pid) == control_pid
    assert member_pids <= {str(s.process.pid) for s in pool.servers}


def test_start_card_servers_refuses_a_card_without_a_server(tmp_path):
    with pytest.raises(ValueError, match="at least one server"):
        wave.start_card_servers([uuid()], tmp_path / "servers", per_card=0)


class _Server:
    def __init__(self):
        self.jobs = 0
        self.process = type("P", (), {"poll": lambda self: None})()

    def load(self):
        return self.jobs

    def submit(self, job, log_path):
        self.jobs += 1
        return self.jobs


def test_a_server_holds_the_running_job_and_the_next():
    """THE BREAKAGE: a member's host start (request, restart set, preflight)
    ran with the card idle between two members' stepping (box N fullN3,
    12-17 s of each 40-48 s leg).  Each server now holds the next job beside
    the running one and no more."""
    server = _Server()
    pool = wave.CardServerPool([server])
    pool.submit({}, "a")
    pool.submit({}, "b")
    with pytest.raises(wave.MemberWaveFailed, match="busy"):
        pool.submit({}, "c")
    two = wave.CardServerPool([_Server(), _Server()])
    for name in "abcd":
        two.submit({}, name)
    assert [s.jobs for s in two.servers] == [2, 2]


def test_the_prefetch_reads_exactly_the_next_job(tmp_path, monkeypatch):
    import threading

    queue = tmp_path / "queue"
    queue.mkdir()
    reads = []

    def fake_read(path, *, expected_hash=None):
        reads.append((Path(path).name, expected_hash))
        return "context", {"restart_files": []}

    monkeypatch.setattr(transport, "read_context", fake_read)
    ahead = transport._Prefetch(queue, 1, threading.Event())
    (queue / "job-00001.json").write_text(json.dumps({
        "schema": transport.SERVER_JOB_SCHEMA, "request": str(tmp_path / "r1.json"),
        "request_sha256": "1" * 64, "result": "x", "log": ""}))
    assert ahead.result_for(tmp_path / "r1.json", "1" * 64) == ("context", {"restart_files": []})
    assert ahead.result_for(tmp_path / "r1.json", "2" * 64) is None
    assert ahead.result_for(tmp_path / "other.json", "1" * 64) is None
    assert reads == [("r1.json", "1" * 64)]
    idle = transport._Prefetch(queue, 2, threading.Event())
    (queue / "stop").touch()
    idle.done.wait(5)
    assert idle.result_for(tmp_path / "r2.json", "3" * 64) is None
