"""A forecast runner with no host draws every frame as it lands.

The breakage these pin: ``gpuwm go`` runs its forecast runner as a
subprocess and asks it, on its command line, to draw the first frame
early.  That was all it drew while the forecast ran: every later frame,
and every nest's frames, waited for the finalize stage, and a nested
``gpuwm go`` drew nothing at all until the end because the tree runner
was never given the flags.  The run-plan route, which hosts the runner
in its own process, already drew each frame as it landed.

A stand-in renderer writes what the real one writes, so the arming, the
queue and the hand-over to the finalize stage are pinned without a real
render.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from gpuwm import first_products, live_products, render_georef
from gpuwm.first_products import FirstProducts

_START = datetime(2026, 9, 26, 12)


def _frame(root: Path, domain: int, hour: int) -> Path:
    valid = _START + timedelta(hours=hour)
    path = root / "wrfout" / f"wrfout_d{domain:02d}_{valid:%Y-%m-%d_%H_%M_%S}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"frame d{domain} h{hour}".encode())
    return path


class _Renderer:
    """One picture per frame, in the layout, with the map record."""

    def __init__(self):
        self.frames = []
        self._lock = threading.Lock()

    def __call__(self, command, **options):
        frame = Path(command[command.index("--series") - 1])
        with self._lock:
            self.frames.append(frame.name)
        out = Path(command[command.index("--out") + 1])
        domain = frame.name.split("_")[1]
        key = f"{domain}/2m_temperature/20260926/{frame.name[7:]}.png"
        (out / key).parent.mkdir(parents=True, exist_ok=True)
        (out / key).write_bytes(b"\x89PNG " + frame.name.encode())
        (out / render_georef.GEOREF_FILENAME).write_text(json.dumps({
            "schema": render_georef.GEOREF_SCHEMA,
            "generated_utc": "2026-09-26T12:00:00Z",
            "panels": {key: {"projection": "lambert", "frame": frame.name}},
            "without_georeference": []}), encoding="utf-8")
        return subprocess.CompletedProcess(list(command), 0, "", "")


def _armed(tmp_path, products="all"):
    from gpuwm import prepared_single_domain_forecast as runner

    return runner._route_owned_first_products(
        SimpleNamespace(render_products=products, render_dir=None),
        outdir=tmp_path / "out", observer=None, started=time.perf_counter())


def test_a_runner_with_no_host_draws_every_frame_as_it_lands(
        tmp_path, monkeypatch, capsys):
    """Root and nest, every frame; the finalize stage has nothing left.

    Fails before this fix: the runner's own arming drew the first root
    frame and nothing else, so three of these four frames reached the
    finalize stage undrawn.
    """

    renderer = _Renderer()
    monkeypatch.setattr(first_products, "_run_render", renderer)
    renders = _armed(tmp_path)
    # Every caller that armed the early render takes this unchanged.
    assert isinstance(renders, FirstProducts)
    assert renders.render_dir == tmp_path / "out" / "png"

    frames = []
    for domain, hour in ((1, 0), (2, 0), (1, 1), (2, 1)):
        frame = _frame(tmp_path / "out", domain, hour)
        frames.append(frame)
        renders.frame_committed(domain=domain,
                                valid_time=_START + timedelta(hours=hour),
                                path=frame)
    # What the runner calls before it writes its report.
    receipt = renders.wait(timeout=60.0)

    assert receipt is not None and receipt["frame"] == str(frames[0])
    assert sorted(renderer.frames) == sorted(frame.name for frame in frames)
    record = live_products.read_receipt(tmp_path / "out" / "png")
    assert sorted(Path(entry["frame"]).name for entry in record["frames"]) == \
        sorted(frame.name for frame in frames[1:])
    # The finalize stage in `gpuwm go` reads both records off disk and
    # finds every frame drawn.
    plan = {"run": tmp_path / "out", "render": tmp_path / "out" / "png",
            "render_products": "all"}
    remaining, _already, _note = first_products.published_frames(frames, plan)
    remaining, drawn, _note = live_products.published_frames(remaining, plan)
    assert remaining == []
    assert sorted(frame.name for frame in drawn) == sorted(
        frame.name for frame in frames[1:])
    out = capsys.readouterr().out
    assert out.count("prepared forecast: pictures ready for") == 3


def test_a_stopped_runner_draws_nothing_more(tmp_path, monkeypatch):
    renderer = _Renderer()
    monkeypatch.setattr(first_products, "_run_render", renderer)
    renders = _armed(tmp_path)
    renders.halt()
    renders.frame_committed(domain=1, valid_time=_START,
                            path=_frame(tmp_path / "out", 1, 0))
    renders.frame_committed(domain=1, valid_time=_START + timedelta(hours=1),
                            path=_frame(tmp_path / "out", 1, 1))
    renders.wait(timeout=30.0)
    # The halt reaches the early render as well as the every-frame one,
    # so nothing committed after it is drawn, the analysis frame included.
    assert renderer.frames == []
    assert first_products.read_receipt(tmp_path / "out" / "png") is None
    assert live_products.read_receipt(tmp_path / "out" / "png") is None


def test_a_stop_during_the_analysis_render_ends_that_render(tmp_path,
                                                            monkeypatch):
    """The stop ends the first-frame render too, not the every-frame one alone.

    Fails before this fix: the runner routes' halt reached only the
    every-frame render, so a stop that landed while the analysis frame
    was being drawn (early in every run) left that render running past
    the stop, free to publish after it.
    """

    import sys

    import gpuwm.go_cli as go_cli

    monkeypatch.setattr(
        go_cli, "render_command",
        lambda plan, frames=None, **_: [sys.executable, "-c",
                                        "import time; time.sleep(120)"])
    renders = _armed(tmp_path)
    renders.frame_committed(domain=1, valid_time=_START,
                            path=_frame(tmp_path / "out", 1, 0))
    deadline = time.monotonic() + 30.0
    while not first_products._RUNNING and time.monotonic() < deadline:
        time.sleep(0.02)
    assert first_products._RUNNING, "the analysis render never started"
    process = next(iter(first_products._RUNNING.values()))

    started = time.monotonic()
    summary = renders.halt()
    assert time.monotonic() - started < 5.0
    assert process.poll() is not None
    assert summary["ended"] is True
    assert first_products.read_receipt(tmp_path / "out" / "png") is None
    # Nothing after the stop is drawn either, and a second halt is safe.
    renders.frame_committed(domain=1, valid_time=_START + timedelta(hours=1),
                            path=_frame(tmp_path / "out", 1, 1))
    assert renders.live.pending == 0
    renders.halt()


def test_no_products_still_arms_nothing(tmp_path):
    assert _armed(tmp_path, products="none") is None
    assert _armed(tmp_path, products=None) is None


def test_gpuwm_go_asks_a_nested_runner_to_draw_as_it_goes(tmp_path,
                                                          monkeypatch):
    """The tree runner takes --render-products; `gpuwm go` never passed it."""

    import gpuwm.go_cli as go_cli

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    receipt = prepared / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    (tmp_path / "authority").mkdir()
    (tmp_path / "authority" / "experiment.toml").write_text(
        "# experiment\n", encoding="utf-8")
    plan = {"render": tmp_path / "case", "run": tmp_path / "run",
            "runner": "gpuwm.prepared_domain_tree_forecast",
            "source": "hrrr", "prepared": prepared,
            "authority": tmp_path / "authority", "domains": 3}
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(go_cli, "_profile_flags", lambda _plan: [])
    monkeypatch.setattr(go_cli, "_hierarchy_document", lambda _root: receipt)
    seen = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **_: seen.append(command))

    go_cli._run_forecast(plan, {}, explain=False,
                         observer=SimpleNamespace(hosts_forecast=False))

    (command,) = seen
    assert command[command.index("--render-products") + 1] == "all"
    assert command[command.index("--render-dir") + 1] == str(tmp_path / "case")
    # Composed without it (a host, run-plan, draws for the runner itself,
    # and --dry-run prints this form), the command is unchanged.
    assert "--render-products" not in go_cli.tree_forecast_command(plan)


def test_a_host_that_draws_every_frame_is_not_doubled(tmp_path):
    """run-plan draws every frame itself whenever it draws at all, so a
    runner it hosts arms only the first-frame render its flag asked for."""

    from gpuwm import prepared_single_domain_forecast as runner
    from gpuwm.live_products import LandingRenders
    from gpuwm.runplan import EventStream, RunObserver

    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    host = RunObserver(events, root_domain=1)
    host.arm_first_products({"run": tmp_path, "render": tmp_path / "png",
                             "render_products": None})
    assert host.first_products is None and host.live_products is not None
    armed = runner._route_owned_first_products(
        SimpleNamespace(render_products="all", render_dir=None),
        outdir=tmp_path / "out", observer=host, started=0.0)
    events.close()
    assert isinstance(armed, FirstProducts)
    assert not isinstance(armed, LandingRenders)


def test_the_wrfinput_and_met_em_doors_stop_their_renders_with_the_run():
    """A stop halts; a failure finishes the frames it wrote; nothing raises."""

    from gpuwm.wrfinput_forecast import stop_door_renders

    class _Renders:
        def __init__(self):
            self.calls = []

        def halt(self):
            self.calls.append("halt")

        def wait(self):
            self.calls.append("wait")
            raise RuntimeError("a wedged render")

    stopped, failed = _Renders(), _Renders()
    stop_door_renders(stopped, KeyboardInterrupt())
    stop_door_renders(failed, RuntimeError("non-finite"))
    stop_door_renders(None, KeyboardInterrupt())
    assert stopped.calls == ["halt"]
    assert failed.calls == ["wait"]
