"""``gpuwm cycle`` draws every boundary it keeps, as it lands.

The breakage these pin (sweep finding RS-7, a known limit carried by the
2.7.5 to 2.7.7 notes): the cycle drew no picture at any point of a run.
Its parent engines publish anchors, nothing under ``gpuwm/cycle/`` wrote
a frame, and the draw-as-it-lands render every other route arms was
never armed here, so a cycle's ``events.jsonl`` held no
``live_products_ready`` because the cycle wrote no events at all.

A stand-in renderer writes what the real one writes (a picture in the
render layout and a ``render-georef.json``), so the queue, the finalize
skip and the stop are pinned without spending a real render; the real
render is proved on a real cycle separately.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from gpuwm import first_products, go_cli, render_georef
from gpuwm.cli import build_parser
from gpuwm.cycle import pictures
from gpuwm.cycle.cli import cycle_main
from gpuwm.cycle.clock import CycleClock
from gpuwm.cycle.ledger import CycleLedger
from gpuwm.cycle.supervisor import CycleSupervisor
from gpuwm.runplan import read_events

NY = NX = 24
_EPOCH = "2026-08-14T18:00:00Z"


def _grid():
    lat = np.linspace(35.0, 35.0 + 0.027 * (NY - 1), NY)[:, None] * np.ones(
        (1, NX))
    lon = np.ones((NY, 1)) * np.linspace(-97.5, -97.5 + 0.033 * (NX - 1),
                                         NX)[None, :]
    return lat, lon


def _fields(k, *, shape=(NY, NX)):
    """One boundary's planes, consistent enough for the replay engine."""

    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    blob = np.exp(-(((yy - (shape[0] // 2 + k)) ** 2
                     + (xx - (shape[1] // 2 + k)) ** 2) / (2 * 3.0 ** 2)))
    rho = 1.10 - 0.02 * blob
    rho_theta = rho * (300.0 + 12.0 * blob)
    return dict(
        rho=rho, rho_theta=rho_theta, rho_u=rho * (8.0 + 14.0 * blob),
        rho_w=rho * 3.5 * blob, scalars=rho * 0.012 * blob,
        exner=np.power(np.maximum(rho_theta, 1e-12) * (287.0 / 100000.0),
                       287.0 / (1004.5 - 287.0)),
        composite_reflectivity=(14.0 + 52.0 * blob))


def _series(tmp_path, n=4, *, coordinates=False, shape=(NY, NX)):
    directory = tmp_path / "state"
    directory.mkdir(parents=True, exist_ok=True)
    lat, lon = _grid()
    for k in range(n):
        extra = dict(XLAT=lat, XLONG=lon) if coordinates else {}
        np.savez(directory / f"frame_{k:03d}.npz",
                 time_seconds=np.asarray(float(k) * 960.0),
                 **_fields(k, shape=shape), **extra)
    return str(directory / "*.npz")


def _geo(tmp_path):
    lat, lon = _grid()
    path = tmp_path / "geo.npz"
    np.savez(path, XLAT=lat, XLONG=lon)
    return str(path)


def _args(tmp_path, argv, *, root):
    base = ["cycle", "--root", str(root), "--epoch-anchor", _EPOCH,
            "--parent-kind", "replay", "--cycle-seconds", "960",
            "--parent-mesh-id", "t24x24"]
    return build_parser().parse_args(base + argv)


class _Renderer:
    """Draws one picture per frame into the layout, as ``gpuwm render`` does.

    ``hold`` names frames whose render waits on ``gate`` (for at most
    ``hold_seconds``) before it returns: a render still in flight.
    ``fail`` names frames whose render exits 1 and draws nothing.
    """

    def __init__(self, *, hold=(), hold_seconds=5.0, fail=()):
        self.frames: list[str] = []
        self.gate = threading.Event()
        self.hold = set(hold)
        self.fail = set(fail)
        self.hold_seconds = hold_seconds
        self._lock = threading.Lock()

    def __call__(self, command, **_options):
        frame = Path(command[command.index("--series") - 1])
        with self._lock:
            self.frames.append(frame.name)
        if frame.name in self.fail:
            return subprocess.CompletedProcess(list(command), 1, "", "boom")
        out = Path(command[command.index("--out") + 1])
        key = (f"d01-3km/composite_reflectivity/20260814/"
               f"{frame.name.replace('wrfout_', 'picture_')}.png")
        (out / key).parent.mkdir(parents=True, exist_ok=True)
        (out / key).write_bytes(b"\x89PNG " + frame.name.encode())
        (out / render_georef.GEOREF_FILENAME).write_text(json.dumps({
            "schema": render_georef.GEOREF_SCHEMA,
            "generated_utc": "2026-08-14T18:00:00Z",
            "panels": {key: {"projection": "lambert", "frame": frame.name}},
            "without_georeference": []}), encoding="utf-8")
        if frame.name in self.hold:
            self.gate.wait(self.hold_seconds)
        return subprocess.CompletedProcess(list(command), 0, "", "")


def _events(root: Path, name: str | None = None) -> list[dict]:
    path = root / pictures.EVENTS_FILENAME
    if not path.exists():
        return []
    events = read_events(path, allow_partial_tail=True)
    return [event for event in events
            if name is None or event["event"] == name]


def _frame_name(k: int) -> str:
    valid = datetime(2026, 8, 14, 18) + (k * 960) * (
        datetime(2026, 1, 1, 0, 0, 1) - datetime(2026, 1, 1))
    return valid.strftime("wrfout_d01_%Y-%m-%d_%H_%M_%S")


def _wait_for(predicate, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.fixture
def renderer(monkeypatch):
    stand_in = _Renderer()
    monkeypatch.setattr(first_products, "_run_render", stand_in)
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    return stand_in


# ---------------------------------------------------------------------------
# 1. the door draws every boundary, by default
# ---------------------------------------------------------------------------
def test_every_boundary_the_cycle_keeps_is_drawn(tmp_path, renderer):
    """Three boundaries, three frames, three ``live_products_ready``.

    Fails before this fix: the door wrote no frame and no event stream,
    so a cycle had nothing to draw and nothing that said so.  No
    ``--render-products`` is passed: drawing is the default.
    """

    root = tmp_path / "run"
    args = _args(tmp_path, ["--cycles", "3",
                            "--parent-state", _series(tmp_path),
                            "--parent-geo-file", _geo(tmp_path),
                            "--parent-dx-m", "3000"], root=root)
    assert cycle_main(args) == 0

    frames = sorted(path.name for path in (root / "wrfout").iterdir())
    assert frames == [_frame_name(k) for k in (1, 2, 3)], frames
    ready = _events(root, "live_products_ready")
    assert [Path(event["frame"]).name for event in ready] == frames
    assert all(event["pictures"] == 1 and event["domain"] == 1
               for event in ready)
    # Each drawn as it landed: the finalize stage had nothing left.
    assert renderer.frames == frames
    pngs = sorted((root / "png").rglob("*.png"))
    assert len(pngs) == 3


def test_each_boundary_is_drawn_while_the_next_leg_runs(tmp_path, renderer):
    """The stub cycle loop: leg k waits to see boundary k-1 drawn.

    A leg of a real parent runs for minutes, so a boundary drawn only at
    the end of the cycle is a boundary nobody saw while it mattered.
    Each stand-in leg records whether the boundary before it had been
    drawn by the time it started stepping.
    """

    root = tmp_path / "run"
    clock = CycleClock.build(
        epoch_anchor=datetime(2026, 8, 14, 18, tzinfo=timezone.utc),
        parent_dt_seconds=120.0, cycle_seconds=960.0, n_cycles=3)
    lat, lon = _grid()
    drawing = pictures.BoundaryPictures(root=root, clock=clock)
    drawn_before_leg: dict[int, bool] = {}

    def drawn(k):
        return any(Path(event["frame"]).name == _frame_name(k)
                   for event in _events(root, "live_products_ready"))

    def advance_parent(cycle_index, anchor_in):
        if cycle_index > 1:
            drawn_before_leg[cycle_index] = _wait_for(
                lambda: drawn(cycle_index - 1))
        ticks = clock.boundary_ticks(cycle_index)
        derived = {"XLAT": lat, "XLONG": lon,
                   **_fields(cycle_index)}
        return {"kind": "replay", "parent_ticks": ticks,
                "anchor_ticks": ticks, "derived_arrays": derived,
                "prognostic_arrays": {}}

    supervisor = CycleSupervisor(
        clock=clock, ledger=CycleLedger(root), root=root,
        advance_parent=advance_parent,
        on_cycle_completed=drawing.boundary_completed)
    supervisor.run(resume=False)
    assert drawing.finish() is True

    assert drawn_before_leg == {2: True, 3: True}
    assert [Path(e["frame"]).name
            for e in _events(root, "live_products_ready")] == [
        _frame_name(k) for k in (1, 2, 3)]
    committed = _events(root, "output_committed")
    assert [event["cycle"] for event in committed] == [1, 2, 3]
    assert drawing.grid.dx_source == "measured off the coordinates"
    assert 2500.0 < drawing.grid.dx_m < 3500.0


def test_the_frame_is_a_wrfout_the_renderer_can_place(tmp_path, renderer):
    """Coordinates, the composite as reflectivity, every plane by name."""

    netCDF4 = pytest.importorskip("netCDF4")
    root = tmp_path / "run"
    args = _args(tmp_path, ["--cycles", "1",
                            "--parent-state",
                            _series(tmp_path, coordinates=True)], root=root)
    assert cycle_main(args) == 0

    frame = root / "wrfout" / _frame_name(1)
    with netCDF4.Dataset(frame) as data:
        names = set(data.variables)
        assert {"XLAT", "XLONG", "REFL_10CM", "rho", "rho_theta",
                "exner"} <= names
        assert data.variables["REFL_10CM"].shape[-2:] == (NY, NX)
        assert float(np.max(data.variables["REFL_10CM"][:])) > 60.0
        assert data.getncattr("SIMULATION_START_DATE") == \
            "2026-08-14_18:00:00"
        assert data.getncattr("GPUWM_CYCLE_GRID_SOURCE") == \
            "the parent's own XLAT/XLONG"
        # A parent that carries no terrain and no column mass gets no
        # zero planes standing in for them: the renderer would draw them.
        assert "HGT" not in names and "MU" not in names, sorted(names)
    assert not list((root / "wrfout").glob(".*")), "a staging file was left"


def _real_renderer_missing() -> str | None:
    from gpuwm import rustwx

    if rustwx.find_renderer() is None:
        return "no built rw_wrfbatch"
    return go_cli.render_extra_missing()


@pytest.mark.skipif(_real_renderer_missing() is not None,
                    reason="needs the real renderer (cargo build in "
                           "tools/rustwx)")
def test_a_boundary_with_no_terrain_draws_no_terrain_picture(tmp_path):
    """Only the planes the boundary carries are drawn, by the real renderer.

    Fails before this fix: the frame writer added zero ``HGT`` and ``MU``
    planes and the default product set drew them, so each boundary of a
    replayed 3 km parent published a Terrain Height map reading 0 m and
    flat-zero ``wrf_hgt``, ``wrf_terrain`` and ``wrf_mu`` pictures.
    """

    root = tmp_path / "run"
    args = _args(tmp_path, ["--cycles", "1",
                            "--parent-state",
                            _series(tmp_path, coordinates=True)], root=root)
    assert cycle_main(args) == 0

    products = {path.parent.parent.name
                for path in (root / "png").rglob("*.png")}
    # `all` draws the named products the frame can draw and no stored variable beside them (the default product
    # set's own rule), so reflectivity is the plane this boundary carries that a picture shows.
    assert "composite_reflectivity" in products, sorted(products)
    assert not any(name.startswith("var_wrf_") for name in products), sorted(products)
    invented = {name for name in products
                if name == "terrain_height"
                or name.startswith(("var_wrf_hgt_", "var_wrf_terrain_",
                                    "var_wrf_mu_"))}
    assert not invented, sorted(invented)


# ---------------------------------------------------------------------------
# 2. what is not drawn says why, and the cycle is unaffected
# ---------------------------------------------------------------------------
def test_a_parent_with_no_coordinates_names_the_flag(tmp_path, renderer,
                                                     capsys):
    root = tmp_path / "run"
    args = _args(tmp_path, ["--cycles", "2",
                            "--parent-state", _series(tmp_path)], root=root)
    assert cycle_main(args) == 0

    out = capsys.readouterr().out
    assert "cycle: pictures: none:" in out
    assert "--parent-geo-file" in out
    assert not (root / "wrfout").exists()
    assert renderer.frames == []
    warnings = _events(root, "warning")
    assert [event["code"] for event in warnings] == ["cycle_pictures_none"]
    assert len(sorted((root / "anchors").glob("*"))) == 2


def test_an_mpas_mesh_parent_says_it_cannot_be_placed(tmp_path, renderer,
                                                      capsys):
    """A boundary stamped as MPAS output is cells, not a grid."""

    root = tmp_path / "run"
    clock = CycleClock.build(
        epoch_anchor=datetime(2026, 8, 14, 18, tzinfo=timezone.utc),
        parent_dt_seconds=120.0, cycle_seconds=960.0, n_cycles=1)
    drawing = pictures.BoundaryPictures(root=root, clock=clock,
                                        geo_file=_geo(tmp_path))
    record = {"kind": "mpas-cuda-frames",
              "prognostic_arrays": _fields(0, shape=(300, 6)),
              "derived_arrays": {}}
    assert drawing.boundary_completed(1, record) is None
    assert drawing.finish() is False
    assert pictures.MESH_SENTENCE in capsys.readouterr().out
    assert renderer.frames == []


# ---------------------------------------------------------------------------
# 3. the end of the run: finalize fills gaps, a stop stops
# ---------------------------------------------------------------------------
def test_finalize_draws_only_the_boundary_the_live_pass_missed(
        tmp_path, monkeypatch):
    renderer = _Renderer(fail={_frame_name(2)})
    monkeypatch.setattr(first_products, "_run_render", renderer)
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    staged: list[list[str]] = []

    def run_stage(label, command, **_kwargs):
        staged.append([Path(item).name for item in command
                       if Path(item).name.startswith("wrfout_d01")])

    monkeypatch.setattr(go_cli, "_run_stage", run_stage)
    root = tmp_path / "run"
    args = _args(tmp_path, ["--cycles", "3",
                            "--parent-state",
                            _series(tmp_path, coordinates=True)], root=root)
    assert cycle_main(args) == 0

    assert staged == [[_frame_name(2)]], staged
    ready = [Path(event["frame"]).name
             for event in _events(root, "live_products_ready")]
    assert ready == [_frame_name(1), _frame_name(3)]


def _hourly_series(tmp_path, n=4):
    """A replayed parent with coordinates, one boundary an hour."""

    directory = tmp_path / "state"
    directory.mkdir(parents=True, exist_ok=True)
    lat, lon = _grid()
    for k in range(n):
        np.savez(directory / f"frame_{k:03d}.npz",
                 time_seconds=np.asarray(float(k) * 3600.0),
                 XLAT=lat, XLONG=lon, **_fields(k))
    return str(directory / "*.npz")


def _hour_name(k: int) -> str:
    return f"wrfout_d01_2026-08-14_{18 + k:02d}_00_00"


def _finalize_on_an_hourly_cycle(tmp_path, monkeypatch, *, window_status):
    """An hourly 3-cycle run, every boundary drawn as it landed.

    The engine's catalog is stood in for: every windowed row carries
    ``window_status``.  Returns the commands the finalize stage ran, the
    frames the catalog was asked about, and what the run printed.
    """

    from gpuwm import rustwx

    renderer = _Renderer()
    monkeypatch.setattr(first_products, "_run_render", renderer)
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    staged: list[list[str]] = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **_kwargs:
                        staged.append(list(command)))
    asked: list[list[str]] = []

    def catalog_rows(renderer_path, wrfouts, *, store_root, heavy=False):
        asked.append([Path(frame).name for frame in wrfouts])
        code = ("renderable" if window_status == "renderable"
                else "windowed-blocked")
        return ([("composite_reflectivity", "direct", "renderable", "",
                  "renderable"),
                 ("qpf_1h", "windowed", window_status, "", code),
                 ("10m_wind_1h_max", "windowed", window_status, "", code)],
                "CATALOG total=3")

    monkeypatch.setattr(rustwx, "find_renderer", lambda: Path("rw_wrfbatch"))
    monkeypatch.setattr(rustwx, "catalog_rows", catalog_rows)
    root = tmp_path / "run"
    args = _args(tmp_path, ["--cycle-seconds", "3600", "--cycles", "3",
                            "--parent-state", _hourly_series(tmp_path)],
                 root=root)
    assert cycle_main(args) == 0
    assert renderer.frames == [_hour_name(k) for k in (1, 2, 3)]
    return staged, asked


def _windowed(command: list[str]) -> bool:
    return ("--products" in command
            and command[command.index("--products") + 1] == "windowed")


def test_an_hourly_cycle_asks_for_no_window_its_boundaries_cannot_close(
        tmp_path, monkeypatch, capsys):
    """Every boundary drawn as it landed, and no windowed pass after.

    Fails before this fix: the finalize stage ran a windowed pass over
    boundaries 2 and 3, which on a real hourly cycle drew nothing, exited
    1 ("batch render incomplete: rendered=0 skipped=49") and ended every
    run on "FAILED render" and "cycle: stopped at render".
    """

    staged, asked = _finalize_on_an_hourly_cycle(
        tmp_path, monkeypatch, window_status="blocked")

    assert not [command for command in staged if _windowed(command)], staged
    assert staged == []
    assert asked == [[_hour_name(k) for k in (1, 2, 3)]]
    out = capsys.readouterr().out
    assert "no windowed picture is drawn" in out
    assert "stopped at" not in out


def test_an_hourly_cycle_whose_boundaries_close_a_window_draws_it(
        tmp_path, monkeypatch, capsys):
    """The engine decides: a boundary series that holds a window keeps its pass."""

    staged, _asked = _finalize_on_an_hourly_cycle(
        tmp_path, monkeypatch, window_status="renderable")

    passes = [command for command in staged if _windowed(command)]
    assert len(passes) == 1, staged
    frames = [Path(item).name for item in passes[0]
              if Path(item).name.startswith("wrfout_d01")]
    assert _hour_name(2) in frames and _hour_name(3) in frames
    assert "no windowed picture is drawn" not in capsys.readouterr().out


def test_a_stopped_cycle_draws_nothing_more(tmp_path, monkeypatch):
    """Ctrl-C during leg 3, with boundary 2 still being drawn.

    The render in flight publishes nothing, nothing queued is drawn, and
    the stop is not held up by the render.
    """

    renderer = _Renderer(hold={_frame_name(2)}, hold_seconds=30.0)
    monkeypatch.setattr(first_products, "_run_render", renderer)
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    ended: list[object] = []
    # The live pass hands its worker's end marker too (``ended=``).
    monkeypatch.setattr(first_products, "end_render",
                        lambda thread, **_: (ended.append(thread),
                                             renderer.gate.set()))
    root = tmp_path / "run"
    clock = CycleClock.build(
        epoch_anchor=datetime(2026, 8, 14, 18, tzinfo=timezone.utc),
        parent_dt_seconds=120.0, cycle_seconds=960.0, n_cycles=4)
    lat, lon = _grid()
    drawing = pictures.BoundaryPictures(root=root, clock=clock)

    def advance_parent(cycle_index, anchor_in):
        if cycle_index == 3:
            assert _wait_for(lambda: _frame_name(2) in renderer.frames)
            raise KeyboardInterrupt
        ticks = clock.boundary_ticks(cycle_index)
        return {"kind": "replay", "parent_ticks": ticks,
                "anchor_ticks": ticks,
                "derived_arrays": {"XLAT": lat, "XLONG": lon,
                                   **_fields(cycle_index)},
                "prognostic_arrays": {}}

    supervisor = CycleSupervisor(
        clock=clock, ledger=CycleLedger(root), root=root,
        advance_parent=advance_parent,
        on_cycle_completed=drawing.boundary_completed)
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        try:
            supervisor.run(resume=False)
        except BaseException:
            drawing.close(stopped=True)
            raise
    assert time.monotonic() - started < 20.0
    assert ended, "the render in flight was not ended"
    ready = [Path(event["frame"]).name
             for event in _events(root, "live_products_ready")]
    assert ready == [_frame_name(1)]
    pictures_on_disk = sorted(path.name
                              for path in (root / "png").rglob("*.png"))
    assert pictures_on_disk == [
        _frame_name(1).replace("wrfout_", "picture_") + ".png"]


def test_render_products_none_draws_nothing(tmp_path, renderer):
    root = tmp_path / "run"
    args = _args(tmp_path, ["--cycles", "1", "--render-products", "none",
                            "--parent-state",
                            _series(tmp_path, coordinates=True)], root=root)
    assert cycle_main(args) == 0
    assert not (root / "wrfout").exists()
    assert not (root / pictures.EVENTS_FILENAME).exists()
    assert renderer.frames == []


def test_an_install_with_no_renderer_says_so_and_cycles(tmp_path, monkeypatch,
                                                        capsys):
    monkeypatch.setattr(go_cli, "render_extra_missing",
                        lambda: "no renderer is installed")
    root = tmp_path / "run"
    args = _args(tmp_path, ["--cycles", "1",
                            "--parent-state",
                            _series(tmp_path, coordinates=True)], root=root)
    assert cycle_main(args) == 0
    err = capsys.readouterr().err
    assert "cycle: pictures: none: no renderer is installed" in err
    assert "remedy: gpuwm setup" in err
    assert not (root / "wrfout").exists()
    assert len(sorted((root / "anchors").glob("*"))) == 1
