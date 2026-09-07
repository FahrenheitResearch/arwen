"""Real Linux processes prove durable jobs and ownership-bounded cleanup.

Only the forecast command is replaced by a harmless CPU process using the
private callable seam. Production RPC has no arbitrary-command parameter.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from gpuwm import remote_worker as rw

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc job ownership")
REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "gpuwm" / "remote_worker.py"


def wait_for(predicate, *, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.03)
    raise AssertionError("bounded wait expired")


def write_script(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def config(tmp_path):
    from gpuwm import domain_wizard as dw
    path = tmp_path / "science's 日本語.toml"
    path.write_text(dw.render_config(name="remote-worker-cpu", start_time=datetime(2026, 9, 5),
        hours=3, projection=dw._projection_entries(40, -100, "auto"),
        dims=dw._dims_for_scale(1, ()), ratios=(),
        fetch_hints=dict(source="gfs", cycle="2026-09-05T00", hours=3,
                         out="data/cache", cadence=3), case_data=None), encoding="utf-8")
    return path


@pytest.fixture
def case(tmp_path):
    source = config(tmp_path)
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(tmp_path / "new output"), "products": "none"}
    directory = None

    def launch(script):
        nonlocal directory
        result = rw._launch(request, tmp_path,
            command_factory=lambda _review, _snapshot: [sys.executable, "-u", str(script)],
            worker_command=[sys.executable, "-u", str(WORKER)])
        directory = tmp_path / ".arwen-jobs" / result["job"]["id"]
        assert result["job"]["state"] in {"running", "completed", "failed"}
        return directory

    yield tmp_path, request, launch
    # A failed assertion must not strand a real worker or descendant.
    directories = [directory] if directory else list((tmp_path / ".arwen-jobs").glob("*"))
    for current in directories:
        if not current or not (current / "job.json").exists():
            continue
        if rw._status(current)["state"] == "running":
            rw._stop(current)
        record = rw._record(current)
        owner = rw._json(current / "started.json")["identity"]
        assert not rw._owned_processes(record["token"], owner_pid=owner["pid"])


def hold_script(root):
    return write_script(root / "harmless worker.py", """
        from pathlib import Path
        import os, time
        root = Path(__file__).resolve().parent
        print("ready: 日本語 🌦 café", flush=True)
        (root / "ready").write_text(str(os.getpid()))
        while not (root / "release").exists():
            (root / "heartbeat").write_text(str(time.monotonic_ns()))
            time.sleep(.03)
        print("finished: 雨 🌧", flush=True)
    """)


@pytest.mark.parametrize("disconnect", ["normal", "hangup"])
def test_job_survives_controller_disconnect_and_reconnects(case, disconnect):
    root, request, _launch = case
    script = hold_script(root)
    (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
    controller = write_script(root / "controller.py", """
        import json, sys, time
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        from gpuwm import remote_worker as rw
        root = Path(__file__).resolve().parent
        request = json.loads((root / "request.json").read_text())
        result = rw._launch(request, root,
            command_factory=lambda review, snapshot: [sys.executable, '-u', str(root / 'harmless worker.py')],
            worker_command=[sys.executable, '-u', str(Path(sys.argv[1]) / 'gpuwm/remote_worker.py')])
        (root / 'launched.json').write_text(json.dumps(result))
        if sys.argv[2] == 'hangup':
            time.sleep(60)
    """)
    process = subprocess.Popen([sys.executable, "-u", str(controller), str(REPO), disconnect],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        wait_for(lambda: (root / "launched.json").exists())
        wait_for(lambda: (root / "ready").exists())
        result = json.loads((root / "launched.json").read_text())
        identifier = result["job"]["id"]
        directory = root / ".arwen-jobs" / identifier
        owner = rw._json(directory / "started.json")["identity"]
        assert owner["pgid"] == owner["pid"]
        assert owner["pgid"] != process.pid
        if disconnect == "hangup":
            os.killpg(process.pid, signal.SIGHUP)
        process.wait(timeout=10)
        assert process.returncode == (0 if disconnect == "normal" else -signal.SIGHUP)
        reconnect = {"schema": "gpuwm.remote.request.v1", "workspace": str(root), "job": identifier}
        assert rw.dispatch({**reconnect, "action": "status"})["job"]["state"] == "running"
        assert rw.dispatch({**reconnect, "action": "list"})["jobs"][0]["id"] == identifier
        assert rw._process(owner["pid"]) == owner
        (root / "release").touch()
        finished = wait_for(lambda: (state if (state := rw._status(directory))["state"] == "completed" else None))
        assert finished["exit_code"] == 0
        assert rw.dispatch({**reconnect, "action": "logs"})["text"] == "ready: 日本語 🌦 café\nfinished: 雨 🌧\n"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=10)


@pytest.mark.parametrize("descendant_mode", ["same-session", "new-session", "orphan-cleared-environment"])
def test_stop_reaps_owned_descendants_and_spares_unrelated_process(case, descendant_mode):
    root, _request, launch = case
    child = write_script(root / "descendant.py", """
        from pathlib import Path
        import os, signal, time
        root = Path(__file__).resolve().parent
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (root / 'descendant-ready').write_text(str(os.getpid()))
        while True:
            time.sleep(.03)
    """)
    intermediary = write_script(root / "intermediary.py", """
        import os, subprocess, sys
        from pathlib import Path
        subprocess.Popen([sys.executable, '-u', str(Path(__file__).parent / 'descendant.py')],
                         start_new_session=True, env={'PATH': os.environ.get('PATH', '')})
    """)
    parent = write_script(root / "parent.py", f"""
        import subprocess, sys, time
        from pathlib import Path
        root = Path(__file__).parent
        subprocess.Popen([sys.executable, '-u', {str(intermediary if descendant_mode == 'orphan-cleared-environment' else child)!r}],
                         start_new_session={descendant_mode != 'same-session'!r})
        while True:
            time.sleep(.03)
    """)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        directory = launch(parent)
        wait_for(lambda: (root / "descendant-ready").exists())
        pid = int((root / "descendant-ready").read_text())
        descendant = rw._process(pid)
        assert descendant
        record = rw._record(directory)
        owner = rw._json(directory / "started.json")["identity"]
        if descendant_mode == "orphan-cleared-environment":
            assert not rw._has_token(pid, record["token"])
            wait_for(lambda: descendant in rw._owned_processes(record["token"], owner_pid=owner["pid"]))
        stopped = rw._stop(directory)["job"]
        assert stopped["state"] == "stopped"
        assert rw._process(pid) is None
        assert not rw._owned_processes(record["token"], owner_pid=owner["pid"])
        assert unrelated.poll() is None
        assert rw._stop(directory)["job"] == stopped
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_wrong_owner_refuses_without_signalling_real_job_or_unrelated_process(case):
    root, _request, launch = case
    directory = launch(hold_script(root))
    wait_for(lambda: (root / "ready").exists())
    started = rw._json(directory / "started.json")
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        fake = dict(started, identity=rw._process(unrelated.pid))
        rw._write(directory / "started.json", fake)
        assert rw._status(directory)["state"] == "ownership_mismatch"
        with pytest.raises(ValueError, match="cannot prove ownership.*no signal was sent"):
            rw._stop(directory)
        assert not (directory / "stop.json").exists()
        assert unrelated.poll() is None
        assert rw._process(started["identity"]["pid"]) == started["identity"]
    finally:
        rw._write(directory / "started.json", started)
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_stop_ignores_marker_for_another_job(case):
    root, _request, launch = case
    directory = launch(hold_script(root))
    wait_for(lambda: (root / "heartbeat").exists())
    before = (root / "heartbeat").read_text()
    rw._write(directory / "stop.json", {"token": "0" * 64, "requested_at": rw._now()})
    wait_for(lambda: (root / "heartbeat").read_text() != before)
    assert rw._status(directory)["state"] == "running"
    assert rw._stop(directory)["job"]["state"] == "stopped"


def test_logs_preserve_unicode_at_chunk_boundaries_and_byte_cursors(case):
    root, _request, launch = case
    expected = ("a" * 16383 + "🌦日本語\n") * 3
    script = write_script(root / "unicode.py", f"""
        import sys
        sys.stdout.buffer.write({expected.encode()!r})
        sys.stdout.buffer.flush()
    """)
    directory = launch(script)
    wait_for(lambda: rw._status(directory)["state"] == "completed")
    cursor, chunks = 0, []
    while True:
        result = rw._logs(directory, {"cursor": cursor, "limit": 16384})
        assert result["cursor"] > cursor
        assert result["cursor"] - cursor == len(result["text"].encode())
        cursor = result["cursor"]
        chunks.append(result["text"])
        if result["eof"]:
            break
    assert "".join(chunks) == expected
    assert cursor == len(expected.encode())
    empty = rw._logs(directory, {"cursor": cursor})
    assert empty["text"] == "" and empty["cursor"] == cursor and empty["eof"]
    with pytest.raises(ValueError, match="past the end"):
        rw._logs(directory, {"cursor": cursor + 1})
