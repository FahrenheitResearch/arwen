"""A frame whose renderer failed partway is not a finished frame.

The breakage: the renderer draws temperature, fails on wind and exits 1.
The early render and the as-it-lands render both published the one
picture it drew and recorded the WHOLE frame as done, with no warning, so
the end-of-run render skipped the frame and wind was never drawn.  Each
frame now carries the renderer's exit and the products it drew; a nonzero
exit keeps the good pictures, warns, and leaves the frame for finalize,
which draws it again.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from gpuwm import first_products, live_products
from gpuwm.first_products import FirstProducts
from gpuwm.live_products import LiveProducts
from gpuwm.runplan import WARNING_CODES

_VALID = datetime(2026, 9, 25, 12)
_NAME = "wrfout_d01_2026-09-25_12_00_00"


def _frame(tmp_path):
    path = tmp_path / "run" / "wrfout" / _NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"one durable history frame")
    return path


def _plan(tmp_path):
    return {"run": tmp_path / "run", "render": tmp_path / "png", "render_products": "t2,wind10m"}


def _renderer(returncode):
    """Draws temperature and files it; wind fails, and the process exits ``returncode``.

    As ``gpuwm render`` does, the picture it finished is filed into the layout and named in its
    invocation receipt.  A failed render also leaves the wind picture it was writing under the
    renderer's flat staging name, reported by nothing, as a renderer ended partway does.
    """

    from gpuwm import render_receipts

    def run(command):
        out = Path(command[command.index("--out") + 1])
        picture = out / "d01-3km" / "2m_temperature" / "20260925" / f"{_NAME}.png"
        picture.parent.mkdir(parents=True, exist_ok=True)
        picture.write_bytes(b"\x89PNG temperature")
        failures = []
        if returncode != 0:
            (out / "rustwx_wrf_20260925_12z_f000_d01-3km_10m_winds.png").write_bytes(b"\x89PNG cut sh")
            failures = [f"{_NAME}: exit -9"]
        render_receipts.publish_invocation(root=out, engine="rust", requested_spec="t2,wind10m",
                                           written=[picture], failures=failures, skipped=[], layout="nested")
        stderr = "" if returncode == 0 else "render FAIL: the renderer ended before wind10m was drawn\n"
        return subprocess.CompletedProcess(list(command), returncode, "", stderr)

    return run


class _Recorder:
    def __init__(self):
        self.reports, self.warnings = [], []

    def report(self, entry):
        self.reports.append(entry)

    def warn(self, code, message, **fields):
        self.warnings.append((code, message, fields))


def _early(tmp_path, returncode):
    recorder = _Recorder()
    trigger = FirstProducts(_plan(tmp_path), report=recorder.report, warn=recorder.warn,
                            runner=_renderer(returncode))
    frame = _frame(tmp_path)
    trigger.frame_committed(domain=1, valid_time=_VALID, path=frame)
    trigger.wait(timeout=30.0)
    return recorder, trigger, frame


def _live(tmp_path, returncode):
    recorder = _Recorder()
    live = LiveProducts(_plan(tmp_path), report=recorder.report, warn=recorder.warn,
                        runner=_renderer(returncode))
    frame = _frame(tmp_path)
    live.frame_committed(domain=1, valid_time=_VALID, path=frame)
    live.stop(timeout=30.0)
    return recorder, live, frame


def test_an_early_render_that_exits_1_keeps_its_picture_and_leaves_the_frame_to_finalize(tmp_path):
    recorder, trigger, frame = _early(tmp_path, returncode=1)
    picture = tmp_path / "png" / "d01-3km" / "2m_temperature" / "20260925" / f"{_NAME}.png"
    assert picture.is_file()                     # the good picture stays
    receipt = trigger.receipt
    assert receipt["exit_code"] == 1 and receipt["complete"] is False
    assert receipt["products"] == ["2m_temperature"]
    assert "ended before wind10m" in receipt["diagnostics"]
    # the staging file of the picture it was writing is not a picture of the run
    assert [entry["name"] for entry in receipt["written"]] == [f"d01-3km/2m_temperature/20260925/{_NAME}.png"]
    assert not list((tmp_path / "png").glob("rustwx_wrf_*.png"))
    assert recorder.reports and recorder.reports[0]["complete"] is False
    assert [code for code, _, _ in recorder.warnings] == ["first_products_incomplete"]
    remaining, already, note = first_products.published_frames([frame], _plan(tmp_path))
    assert remaining == [frame] and already == []
    assert "exited 1" in note


def test_a_live_render_that_exits_1_keeps_its_picture_and_leaves_the_frame_to_finalize(tmp_path):
    recorder, live, frame = _live(tmp_path, returncode=1)
    picture = tmp_path / "png" / "d01-3km" / "2m_temperature" / "20260925" / f"{_NAME}.png"
    assert picture.is_file()
    entry = live.published[0]
    assert entry["exit_code"] == 1 and entry["complete"] is False and entry["products"] == ["2m_temperature"]
    assert not list((tmp_path / "png").glob("rustwx_wrf_*.png"))
    assert recorder.reports[0]["complete"] is False
    assert [code for code, _, _ in recorder.warnings] == ["live_products_incomplete"]
    remaining, already, note = live_products.published_frames([frame], _plan(tmp_path))
    assert remaining == [frame] and already == []
    assert "drawn again" in note


def test_a_clean_render_is_still_skipped_by_finalize(tmp_path):
    _, trigger, frame = _early(tmp_path, returncode=0)
    assert trigger.receipt["complete"] is True
    remaining, already, _ = first_products.published_frames([frame], _plan(tmp_path))
    assert remaining == [] and already == [frame]
    (tmp_path / "png" / first_products.FIRST_PRODUCTS_RECEIPT).unlink()
    _, live, frame = _live(tmp_path, returncode=0)
    remaining, already, _ = live_products.published_frames([frame], _plan(tmp_path))
    assert remaining == [] and already == [frame]


def test_the_new_warnings_are_registered():
    assert "first_products_incomplete" in WARNING_CODES
    assert "live_products_incomplete" in WARNING_CODES


def test_finalize_draws_again_the_frame_the_renderer_left_incomplete(tmp_path, monkeypatch):
    # The whole path the finding names: the early render draws temperature, fails on wind and exits 1; the
    # end-of-run render stage then hands that frame to the renderer again, so wind is drawn, and says why.
    from gpuwm import go_cli

    _, _, frame = _early(tmp_path, returncode=1)
    plan = {**_plan(tmp_path), "wrfout_dir": frame.parent}
    ran = []
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(go_cli, "wrfout_frames", lambda _plan: [frame])
    monkeypatch.setattr(go_cli, "_run_stage", lambda stage, command, **_: ran.append([str(p) for p in command]))
    monkeypatch.setattr(live_products, "engine_windowed_slugs", lambda _frame, **_where: frozenset())
    assert go_cli._render_stage(plan, explain=False) is True
    assert len(ran) == 1 and str(frame) in ran[0]


def test_finalize_skips_the_frame_a_clean_early_render_drew(tmp_path, monkeypatch):
    from gpuwm import go_cli

    _, _, frame = _early(tmp_path, returncode=0)
    plan = {**_plan(tmp_path), "wrfout_dir": frame.parent}
    ran = []
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(go_cli, "wrfout_frames", lambda _plan: [frame])
    monkeypatch.setattr(go_cli, "_run_stage", lambda stage, command, **_: ran.append([str(p) for p in command]))
    monkeypatch.setattr(live_products, "engine_windowed_slugs", lambda _frame, **_where: frozenset())
    assert go_cli._render_stage(plan, explain=False) is True
    assert ran == []


def test_a_record_from_before_renders_recorded_their_exit_is_not_trusted(tmp_path):
    # Such a record said "done" for a render that failed partway too, so it cannot show every product was drawn.
    import json

    _, trigger, frame = _early(tmp_path, returncode=0)
    path = tmp_path / "png" / first_products.FIRST_PRODUCTS_RECEIPT
    legacy = json.loads(path.read_text(encoding="utf-8"))
    for key in ("complete", "exit_code"):
        legacy.pop(key, None)
    path.write_text(json.dumps(legacy), encoding="utf-8")
    remaining, already, note = first_products.published_frames([frame], _plan(tmp_path))
    assert remaining == [frame] and already == [] and "does not record" in note
    path.unlink()
    _, live, frame = _live(tmp_path, returncode=0)
    record = tmp_path / "png" / live_products.LIVE_PRODUCTS_RECEIPT
    legacy = json.loads(record.read_text(encoding="utf-8"))
    for entry in legacy["frames"]:
        entry.pop("complete", None)
        entry.pop("exit_code", None)
    record.write_text(json.dumps(legacy), encoding="utf-8")
    remaining, already, _ = live_products.published_frames([frame], _plan(tmp_path))
    assert remaining == [frame] and already == []
