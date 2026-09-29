"""A resumed forecast draws its first new hour's rainfall from the saved history.

Breakage prevented: a 120 hour GFS forecast resumed at hour 24 published 96
temperature pictures but 95 qpf_1h and 95 qpf_total, because the hour-24
frame saved beside the checkpoint was never given to the render as context
(and a context frame from another folder was not joined to the new series);
the first-products receipt still said complete.
"""
from datetime import timedelta
import json
from pathlib import Path

from gpuwm.first_products import FirstProducts
from gpuwm.io.restart import write_tree_restart
from gpuwm.live_products import LiveProducts, windowed_passes
from gpuwm.restart_render import history_before_restart
from test_live_products import _Recorder, _Renderer, _context
from test_restart import _sealed_tree_fixture
from test_first_products import _stand_in_renderer


def _saved(tmp_path, monkeypatch):
    model, start = _sealed_tree_fixture(monkeypatch, forcing_count=3,
                                       run_seconds=7200, payload_seed=91)
    checkpoint = write_tree_restart(tmp_path / "old", model, start + timedelta(hours=1))
    frames = {}
    for grid in (1, 2):
        for hour in (0, 1, 2):
            path = checkpoint.parent / "wrfout" / f"wrfout_d{grid:02d}_{start + timedelta(hours=hour):%Y-%m-%d_%H_%M_%S}"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"grid {grid}, hour {hour}".encode())
            frames[grid, hour] = path
    return checkpoint, start, frames


def _new(tmp_path, start, grid, hour):
    valid = start + timedelta(hours=hour)
    frame = tmp_path / "new" / "wrfout" / f"wrfout_d{grid:02d}_{valid:%Y-%m-%d_%H_%M_%S}"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"new frame")
    return frame, valid


def test_restart_context_excludes_history_written_after_the_checkpoint(tmp_path, monkeypatch):
    checkpoint, _, old = _saved(tmp_path, monkeypatch)
    assert set(history_before_restart(checkpoint)) == {old[g, h] for g in (1, 2) for h in (0, 1)}
    # A renamed checkpoint still takes its clock from its real header.
    renamed = checkpoint.with_name("saved.npz")
    renamed.write_bytes(checkpoint.read_bytes())
    assert history_before_restart(renamed) == history_before_restart(checkpoint)


def test_first_resumed_frame_gets_saved_context_only_for_its_grid(tmp_path, monkeypatch):
    checkpoint, start, old = _saved(tmp_path, monkeypatch)
    frame, valid = _new(tmp_path, start, 1, 2)
    renderer, recorder = _Renderer(), _Recorder()
    trigger = FirstProducts({"run": tmp_path / "new", "render": tmp_path / "png",
        "restart": checkpoint, "render_products": "t2,qpf_1h,qpf_total"},
        report=recorder.report, warn=recorder.warn, runner=renderer)
    trigger.frame_committed(domain=1, valid_time=valid, path=frame)
    assert trigger.wait(10) is not None
    # The hour the frame closes, as the unbroken run's per-frame render
    # holds it; the longer windows are the end-of-run pass's.
    assert _context(renderer.calls[0]) == [old[1, 1]]


def _windows_listed(monkeypatch):
    """The renderer's listing of windowed slugs, the same on every box."""
    from gpuwm import live_products
    monkeypatch.setattr(live_products, "catalog_windowed_slugs",
                        lambda: frozenset({"qpf_1h", "qpf_total"}))


def test_first_resumed_frame_imports_no_saved_context_without_a_window(tmp_path, monkeypatch):
    """A baseline buys only windows.  A 12 km GFS run resumed from its
    hour-1 checkpoint and asked for composite reflectivity alone drew its
    first new hour with the saved hour-1 frame imported beside it (the
    renderer's receipt listed it as a context input)."""
    _windows_listed(monkeypatch)
    checkpoint, start, _old = _saved(tmp_path, monkeypatch)
    frame, valid = _new(tmp_path, start, 1, 2)
    renderer, recorder = _Renderer(), _Recorder()
    trigger = FirstProducts({"run": tmp_path / "new", "render": tmp_path / "png",
        "restart": checkpoint, "render_products": "composite_reflectivity"},
        report=recorder.report, warn=recorder.warn, runner=renderer)
    trigger.frame_committed(domain=1, valid_time=valid, path=frame)
    assert trigger.wait(10) is not None
    assert _context(renderer.calls[0]) == []


def test_first_resumed_frame_keeps_its_saved_hour_for_a_listed_window(tmp_path, monkeypatch):
    _windows_listed(monkeypatch)
    checkpoint, start, old = _saved(tmp_path, monkeypatch)
    frame, valid = _new(tmp_path, start, 1, 2)
    renderer, recorder = _Renderer(), _Recorder()
    trigger = FirstProducts({"run": tmp_path / "new", "render": tmp_path / "png",
        "restart": checkpoint, "render_products": "composite_reflectivity,qpf_1h"},
        report=recorder.report, warn=recorder.warn, runner=renderer)
    trigger.frame_committed(domain=1, valid_time=valid, path=frame)
    assert trigger.wait(10) is not None
    assert _context(renderer.calls[0]) == [old[1, 1]]


def test_resumed_live_frame_imports_no_saved_context_without_a_window(tmp_path, monkeypatch):
    """The live pass notes the saved history and still imports none of it
    for a request with no window: both halves of LiveProducts.__init__."""
    _windows_listed(monkeypatch)
    checkpoint, start, _old = _saved(tmp_path, monkeypatch)
    frame, valid = _new(tmp_path, start, 2, 2)
    renderer, recorder = _Renderer(), _Recorder()
    live = LiveProducts({"run": tmp_path / "new", "render": tmp_path / "png",
        "restart": checkpoint, "render_products": "composite_reflectivity"},
        report=recorder.report, warn=recorder.warn, runner=renderer)
    live.frame_committed(domain=2, valid_time=valid, path=frame)
    live.stop(10)
    assert len(renderer.calls) == 1
    assert _context(renderer.calls[0]) == []


def test_finalize_of_a_resumed_run_imports_saved_history_only_for_a_window(
        tmp_path, monkeypatch):
    """The end-of-run batch of a resumed run: the saved frames before the
    checkpoint are baselines of its grid for a request with a window, and
    nothing is imported for one without."""
    from gpuwm import go_cli
    from test_first_products import _a_box_that_can_draw
    _a_box_that_can_draw(monkeypatch)
    _windows_listed(monkeypatch)
    checkpoint, start, old = _saved(tmp_path, monkeypatch)
    current = [_new(tmp_path, start, 1, hour)[0] for hour in (2, 3)]
    for products, expected in (("qpf_1h", [old[1, 0], old[1, 1]]),
                               ("composite_reflectivity", [])):
        commands = []
        monkeypatch.setattr(go_cli, "_run_stage",
                            lambda label, command, **kw: commands.append(list(command)))
        plan = {"run": tmp_path / "new", "render": tmp_path / f"png-{products}",
                "restart": checkpoint, "render_products": products}
        assert go_cli._render_stage(plan, explain=False, observer=None)
        assert len(commands) == 1
        assert all(str(frame) in commands[0] for frame in current)
        assert _context(commands[0]) == expected, products


def test_first_resumed_child_gets_its_own_last_whole_hour(tmp_path, monkeypatch):
    checkpoint, start, old = _saved(tmp_path, monkeypatch)
    frame, valid = _new(tmp_path, start, 2, 2)
    renderer, recorder = _Renderer(), _Recorder()
    live = LiveProducts({"run": tmp_path / "new", "render": tmp_path / "png",
        "restart": checkpoint, "render_products": "t2,qpf_1h,qpf_total"},
        report=recorder.report, warn=recorder.warn, runner=renderer)
    live.frame_committed(domain=2, valid_time=valid, path=frame)
    live.stop(10)
    assert _context(renderer.calls[0]) == [old[2, 1]]


def test_long_restart_context_uses_the_existing_input_file_door(tmp_path, monkeypatch):
    from gpuwm import rustwx
    checkpoint, start, old = _saved(tmp_path, monkeypatch)
    frame, valid = _new(tmp_path, start, 1, 2)
    monkeypatch.setattr(rustwx, "COMMAND_LINE_BUDGET", 1)
    seen = []
    draw = _stand_in_renderer()
    def renderer(command):
        inputs = Path(command[command.index("--inputs-from") + 1])
        seen.append(json.loads(inputs.read_text()))
        return draw(command)
    recorder = _Recorder()
    trigger = FirstProducts({"run": tmp_path / "new", "render": tmp_path / "png",
        "restart": checkpoint, "render_products": "t2,qpf_1h,qpf_total"},
        report=recorder.report, warn=recorder.warn, runner=renderer)
    trigger.frame_committed(domain=1, valid_time=valid, path=frame)
    assert trigger.wait(10) is not None
    assert seen[0]["wrfout"] == [str(frame)]
    assert seen[0]["context_wrfout"] == [str(old[1, 1])]


def test_finalize_uses_prior_frames_as_context_without_drawing_them(tmp_path, monkeypatch):
    checkpoint, start, old = _saved(tmp_path, monkeypatch)
    current = [_new(tmp_path, start, 1, hour)[0] for hour in (2, 3)]
    passes = windowed_passes(current, current, "qpf_1h",
        windowed_slugs=lambda _: {"qpf_1h"}, prior_frames=history_before_restart(checkpoint))
    assert len(passes) == 1
    wanted, context, products = passes[0]
    assert wanted == current
    assert context == [old[1, 0], old[1, 1]]
    assert products == "qpf_1h"


def test_explicit_saved_context_joins_its_compatible_continuation(tmp_path):
    from gpuwm.render import group_history_series
    from test_render_rust import _write_wrfout
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    before = _write_wrfout(old / "before.nc", ("2026-09-27_00:00:00",))
    after = _write_wrfout(new / "after.nc", ("2026-09-27_01:00:00",))
    assert len(group_history_series([before, after])) == 2
    assert group_history_series([after, before], context_paths=[before]) == [[before, after]]


def test_cross_directory_context_never_bridges_another_grid_or_overlapping_time(tmp_path):
    from gpuwm.render import group_history_series
    from test_render_rust import _write_wrfout
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    after = _write_wrfout(new / "after.nc", ("2026-09-27_01:00:00",))
    other_grid = _write_wrfout(old / "other.nc", ("2026-09-27_00:00:00",), grid_id=3)
    overlap = _write_wrfout(old / "overlap.nc", ("2026-09-27_01:00:00",))
    for context in (other_grid, overlap):
        assert len(group_history_series([after, context], context_paths=[context])) == 2


def test_cross_directory_context_refuses_ambiguous_independent_targets(tmp_path):
    import pytest
    from gpuwm.render import group_history_series
    from test_render_rust import _write_wrfout
    paths = []
    for name, stamp in (("old", "2026-09-27_00:00:00"),
                        ("new-a", "2026-09-27_01:00:00"),
                        ("new-b", "2026-09-27_01:00:00")):
        folder = tmp_path / name
        folder.mkdir()
        paths.append(_write_wrfout(folder / "frame.nc", (stamp,)))
    with pytest.raises(ValueError, match="cannot say which one it continues"):
        group_history_series(paths, context_paths=paths[:1])


def test_the_hour_before_a_frame_holds_the_saved_frames_between_hours(tmp_path):
    from gpuwm.restart_render import hour_before
    saved = [tmp_path / f"wrfout_d02_2026-09-27_{stamp}" for stamp in
             ("01_00_00", "02_00_00", "02_15_00", "02_30_00", "02_45_00")]
    other = tmp_path / "wrfout_d01_2026-09-27_02_00_00"
    history = [*saved, other]
    assert hour_before(tmp_path / "wrfout_d02_2026-09-27_03_00_00",
                       history) == saved[1:]
    # A frame between hours closes no window and is drawn alone.
    assert hour_before(tmp_path / "wrfout_d02_2026-09-27_03_15_00", history) == []
    # A grid with no saved whole hour has nothing to close.
    assert hour_before(tmp_path / "wrfout_d03_2026-09-27_03_00_00", history) == []
