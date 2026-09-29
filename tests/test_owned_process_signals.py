"""A stop reaches the process that was recorded and nothing else, and the assistant's server is never lost.

The defects this file holds shut:

- Stop, its escalation a minute later and the assistant's unload checked a
  recorded process and then signalled its number: a process that ended in
  between could have its number handed to another program, which the
  signal reached.  The signal now goes through the checked process's own
  handle (a Linux pidfd, or a Windows handle held across taskkill).
- The assistant's server record was deleted before the server was
  stopped, so a stop that did not take left a model on the card that
  nothing could find again; a server that stopped answering was replaced
  while it still ran; a server that could not be started raised a
  traceback; and a failed unload failed the forecast start that asked
  for it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from gpuwm import proc_identity
from gpuwm.gui.assistant import local
from gpuwm.gui.assistant.llm import LlmError

FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
POPEN = subprocess.Popen


def _sleeper(*, group: bool = False, child: bool = False) -> subprocess.Popen:
    code = "import time; time.sleep(60)"
    if child:
        code = ("import subprocess, sys, time\n"
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\ntime.sleep(60)\n")
    return POPEN([sys.executable, "-c", code], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, creationflags=FLAGS,
                            start_new_session=group and os.name != "nt")


def test_a_signal_reaches_the_recorded_process_and_never_a_stale_record(tmp_path):
    process = _sleeper()
    try:
        record = proc_identity.identify(process.pid)
        assert not proc_identity.signal_process({**record, "start": "1"}, signal.SIGTERM)
        assert not proc_identity.signal_process({"pid": process.pid}, signal.SIGTERM)
        assert process.poll() is None
        assert proc_identity.signal_process(record, signal.SIGTERM)
        process.wait(timeout=10)
        # Once it has ended, the same record signals nothing.
        assert not proc_identity.signal_process(record, signal.SIGTERM)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux process handles")
def test_a_tree_signal_reaches_every_member_of_the_group_through_handles(monkeypatch):
    process = _sleeper(group=True, child=True)
    try:
        import time

        members = []
        for _ in range(100):
            members = [int(p.name) for p in Path("/proc").iterdir() if p.name.isdecimal()
                       and int(p.name) != process.pid and _pgid(int(p.name)) == process.pid]
            if members:
                break
            time.sleep(0.05)
        assert members, "the leader's child did not start"

        def numeric(*_args):
            raise AssertionError("a numeric signal after a check can reach a program given the number since")

        monkeypatch.setattr(os, "kill", numeric)
        monkeypatch.setattr(os, "killpg", numeric)
        assert proc_identity.signal_process(proc_identity.identify(process.pid), signal.SIGTERM, tree=True)
        monkeypatch.undo()
        process.wait(timeout=10)
        for _ in range(100):
            if not any(proc_identity.running(pid) for pid in members):
                break
            time.sleep(0.05)
        assert not any(proc_identity.running(pid) for pid in members)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def _pgid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except OSError:
        return None


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux process handles")
def test_a_handle_opened_after_the_process_changed_signals_nothing(monkeypatch):
    process = _sleeper()
    try:
        record = proc_identity.identify(process.pid)
        checks = iter([True, False])
        # The process passes the first check and is another process by the time its handle is checked.
        monkeypatch.setattr(proc_identity, "alive", lambda *_a, **_k: next(checks, False))
        assert not proc_identity.signal_process(record, signal.SIGTERM)
        monkeypatch.undo()
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=10)


# ---------------------------------------------------------------- the assistant's own server

def _record(home: Path, process: subprocess.Popen, port: int = 8089) -> None:
    (home / "server.json").write_text(json.dumps({"pid": process.pid, "process": proc_identity.identify(process.pid),
                                                  "port": port, "model": "chosen"}), encoding="utf-8")


def test_unload_ends_the_real_server_and_then_forgets_it(tmp_path):
    process = _sleeper()
    instance = local.Local(tmp_path)
    _record(tmp_path, process)
    try:
        assert instance.stop() is True
        process.wait(timeout=10)
        assert not (tmp_path / "server.json").exists()
        assert instance.stop() is False
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_a_stop_that_does_not_take_keeps_the_record_to_try_again(tmp_path, monkeypatch):
    process = _sleeper()
    instance = local.Local(tmp_path)
    _record(tmp_path, process)

    def denied(*_args, **_kwargs):
        raise OSError("Access is denied.")

    monkeypatch.setattr(proc_identity, "signal_process", denied)
    try:
        with pytest.raises(LlmError, match="could not stop"):
            instance.stop()
        assert (tmp_path / "server.json").exists()
        monkeypatch.undo()
        assert instance.stop() is True
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


def test_a_server_that_stopped_answering_is_stopped_before_another_starts(tmp_path, monkeypatch):
    process = _sleeper()
    instance = local.Local(tmp_path)
    _record(tmp_path, process)
    monkeypatch.setattr(instance, "server_binary", lambda *a: tmp_path / "llama-server")
    monkeypatch.setattr(instance, "model_ready", lambda row: True)
    monkeypatch.setattr(local, "_free_port", lambda preferred: 9001)

    def health(url, **_kwargs):
        if ":8089/" in url:
            raise LlmError("The server is not answering.")
        return {"status": "ok"}

    started = []

    def launch(*_args, **_kwargs):
        assert process.poll() is not None, "the unanswering server still held the card when its replacement began"
        replacement = _sleeper()
        started.append(replacement)
        return replacement

    monkeypatch.setattr(local, "get_json", health)
    monkeypatch.setattr(local.subprocess, "Popen", launch)
    try:
        record = instance.start({"id": "chosen", "file": "model.gguf"})
        assert record["pid"] == started[0].pid and record["port"] == 9001
    finally:
        for item in [process, *started]:
            if item.poll() is None:
                item.kill()
            item.wait(timeout=10)


def test_a_server_that_cannot_start_says_so_and_leaves_no_record(tmp_path, monkeypatch):
    instance = local.Local(tmp_path)
    monkeypatch.setattr(instance, "server_binary", lambda *a: tmp_path / "deleted-llama-server")
    monkeypatch.setattr(instance, "model_ready", lambda row: True)
    with pytest.raises(LlmError, match="could not start"):
        instance.start({"id": "chosen", "file": "model.gguf"})
    assert not (tmp_path / "server.json").exists()


def test_an_unload_never_waits_behind_a_download(tmp_path):
    instance = local.Local(tmp_path)
    done = threading.Event()
    with instance._lock:  # a download holds this for its whole transfer
        worker = threading.Thread(target=lambda: (instance.stop(), done.set()), daemon=True)
        worker.start()
        assert done.wait(2), "an accepted forecast's unload waited on a model download"
    worker.join(timeout=2)


def test_a_failed_unload_keeps_the_forecast_start_and_says_why(tmp_path):
    from gpuwm.gui.api import ApiError
    from gpuwm.gui.assistant.service import Assistant

    def cannot_stop():
        raise LlmError("The model server did not stop and may still hold card memory. Try Unload again.")

    assistant = Assistant(SimpleNamespace(root=tmp_path), local=SimpleNamespace(stop=cannot_stop))
    assistant.settings = lambda: {"enabled": True, "backend": "bundled"}
    assistant.card_warning = lambda: None
    assert assistant.make_room() == {"unloaded": False, "warning": assistant.note}
    assert "may still hold card memory" in assistant.note
    with pytest.raises(ApiError) as caught:
        assistant.post(["unload"], {}, False)
    assert caught.value.status == 502
