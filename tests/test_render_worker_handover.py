"""A Draw again handed to a render worker that is about to draw keeps what it asked for.

The defect: the worker wrote its job file outside the job lock when it
began a batch (and when it started), reading the file first and writing it
back.  A fresh start landing between that read and that write was written
over: the new attempt's number and products went back to the old ones, the
picture asked for was never drawn, and the worker ended as if nothing had
been asked.
"""

from __future__ import annotations

import io
import json
import os
import threading
from types import SimpleNamespace

from gpuwm import machine_agent as agent


def _args(workspace):
    return SimpleNamespace(workspace=str(workspace), job="job1", python="python")


def test_a_fresh_start_that_lands_as_a_batch_begins_is_not_written_over(tmp_path, monkeypatch):
    workspace = tmp_path / "w"
    folder = agent.render_dir(workspace, "job1")
    monkeypatch.setattr(agent, "spawn", lambda *a, **k: os.getpid())
    passes = []

    def sleep(_seconds):
        passes.append(1)
        assert len(passes) < 2000, "the worker waits for an end that belongs to an attempt it no longer knows"

    monkeypatch.setattr(agent.time, "sleep", sleep)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"run": "r1", "fresh": False})))
    agent.cmd_render_start(_args(workspace))
    frame = tmp_path / "wrfout_d01_2026-09-24_12_00_00"
    frame.write_bytes(b"frame")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps([str(frame)])))
    agent.cmd_render_feed(_args(workspace))

    drawn: list[list[str]] = []

    def renderer(argv, **_kwargs):
        drawn.append(list(argv))
        if "--products" in argv and argv[argv.index("--products") + 1] == "radar":
            agent.cmd_render_end(_args(workspace))  # the fresh attempt's own end, once its picture is drawn
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(agent.subprocess, "run", renderer)
    real_write = agent.write_json
    started = []

    def update(path, **fields):
        # The worker's read and its write, with a fresh start landing between the two.
        document = agent.read_json(path, default={}) or {}
        if fields.get("state") == "rendering" and not started:
            monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"run": "r1", "fresh": True,
                                                                      "products": "radar"})))
            thread = threading.Thread(target=lambda: started.append(agent.cmd_render_start(_args(workspace))))
            thread.start()
            thread.join(0.5)
            started.append(thread)
        document.update(fields)
        real_write(path, document)
        return document

    monkeypatch.setattr(agent, "update_json", update)
    assert agent.cmd_render_loop(_args(workspace)) == 0
    for item in started:
        if isinstance(item, threading.Thread):
            item.join(10.0)
    job = agent.read_json(folder / "render-job.json")
    assert job["attempt"] == 2 and job["products"] == "radar", job
    assert job["state"] == "finished"
    assert any("--products" in argv and argv[argv.index("--products") + 1] == "radar" for argv in drawn), drawn
