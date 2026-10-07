"""Controller barrier, source retention and serial resource lifetime on CPU."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.da.member_roster import forecast_roster


@dataclass
class Context:
    t_end: float = 60.0
    absolute_leg_number: int = 1
    restart: Path | None = None
    setup_arrays: dict | None = None
    thb_host: object = None


class Stage:
    def __init__(self, root):
        self.root, self.consumed = root, []

    def directory(self, leg, name):
        return self.root/str(leg)/str(name)

    def consume(self, path):
        self.consumed.append(path)
        path.unlink()
        return True


def fixture(monkeypatch, tmp_path):
    from gpuwm.io import restart
    stage = Stage(tmp_path/"accepted")
    contexts = {}
    for name in ["control", 0, 1]:
        source = tmp_path/f"source-{name}.npz"
        source.write_bytes(b"original accepted tree")
        contexts[name] = Context(restart=source)
    monkeypatch.setattr(restart, "tree_restart_members", lambda p: {1: Path(p)})
    monkeypatch.setattr(restart, "read_restart_header", lambda p: {"elapsed_seconds": 60.0})
    seen, events = {}, []

    def run(context, name):
        assert all(c.restart.is_file() for c in contexts.values())
        seen[name] = context
        events.append(name)
        directory = stage.directory(1, name)
        directory.mkdir(parents=True)
        output = directory/"gpuwmrst_d01_000060.npz"
        output.write_bytes(b"new complete tree")
        return SimpleNamespace(restart=output,
            record={"elapsed_seconds": 60.0, "restart": {}},
            setup_arrays={"geometry": np.array([1.0], np.float32)},
            thb_host=np.array([300.0], np.float32))
    return stage, contexts, run, seen, events


def invoke(stage, contexts, run, release, tmp_path):
    return forecast_roster(["control", 0, 1], context_for=contexts.__getitem__,
        packed=False, devices=[], members_per_card=1, member_peak_bytes=0,
        workdir=tmp_path/"wave", stage=stage, run_leg=run,
        release=release, timeout_seconds=1)


def test_complete_roster_consumes_inputs_after_last_result_and_shares_geometry(monkeypatch, tmp_path):
    stage, contexts, run, seen, events = fixture(monkeypatch, tmp_path)
    results, receipt = invoke(stage, contexts, run, lambda: events.append("release"), tmp_path)
    assert events == ["control", "release", 0, "release", 1, "release"]
    assert list(results) == ["control", 0, 1]
    assert receipt["analysis_permitted"] is True
    assert len(stage.consumed) == 3
    assert seen[0].setup_arrays is results["control"].setup_arrays
    assert seen[1].thb_host is results["control"].thb_host


def test_member_failure_keeps_every_input_and_releases_failed_scope(monkeypatch, tmp_path):
    stage, contexts, run, _, events = fixture(monkeypatch, tmp_path)
    def fail(context, name):
        if name == 1:
            raise RuntimeError("member 1 failed before publication")
        return run(context, name)
    with pytest.raises(RuntimeError, match="member 1"):
        invoke(stage, contexts, fail, lambda: events.append("release"), tmp_path)
    assert events[-1] == "release"
    assert stage.consumed == []
    assert all(c.restart.read_bytes() == b"original accepted tree" for c in contexts.values())


def test_wrong_last_member_clock_does_not_consume_any_input(monkeypatch, tmp_path):
    stage, contexts, run, _, _ = fixture(monkeypatch, tmp_path)
    def wrong_clock(context, name):
        result = run(context, name)
        if name == 1:
            result.record["elapsed_seconds"] = 59.0
        return result
    with pytest.raises(ValueError, match="analysis clock"):
        invoke(stage, contexts, wrong_clock, lambda: None, tmp_path)
    assert stage.consumed == []
    assert all(c.restart.is_file() for c in contexts.values())


def test_incomplete_roster_never_starts_control(tmp_path):
    with pytest.raises(ValueError, match="every member"):
        forecast_roster(["control", 1], context_for=lambda _: pytest.fail("started"),
            packed=False, devices=[], members_per_card=1, member_peak_bytes=0,
            workdir=tmp_path/"wave", stage=None, run_leg=None, release=None,
            timeout_seconds=1)
