"""A sub-hourly grid's closing frame is judged by its own render's exit code.

Breakage these prevent: a render that fails partway can leave an
unfinished picture under a staging name in the scratch, and publishing
what the scratch held published that PNG as one of the frame's pictures.
On a 15-minute grid the closing frame was drawn alone and again, for its
rainfall windows, in a second pass into the same scratch whose exit code
was dropped.  It is now drawn in one pass beside every frame of its
hour; what that pass leaves is published whole only when it exited 0,
and otherwise only where a render receipt names it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from gpuwm import render_receipts

from test_live_products import (_Renderer, _context, _drawn_frame, _nest_hour)


def _closing(command):
    """The launch for the frame that closes the nest's first hour."""
    return _drawn_frame(command).name.endswith("13_00_00")


class _UnfinishedClosing(_Renderer):
    """The closing frame's pass dies partway: a staging PNG, no receipt."""

    def __call__(self, command):
        if _closing(command):
            with self._lock:
                self.calls.append(list(command))
            out = Path(command[command.index("--out") + 1])
            (out / "unfinished-qpf.png").write_bytes(b"partial-png")
            return subprocess.CompletedProcess(list(command), 1, "",
                                               "render failed")
        return super().__call__(command)


def test_a_closing_frame_that_finished_nothing_publishes_nothing(tmp_path):
    launches, recorder, nest = _nest_hour(tmp_path, "all",
                                          _UnfinishedClosing())
    [closing] = launches[nest[60].name]
    assert _context(closing) == [nest[m] for m in (0, 15, 30, 45)]
    assert nest[60].name not in {Path(r["frame"]).name
                                 for r in recorder.reports}
    assert any(code == "live_products_empty" and nest[60].name in message
               for code, message, _ in recorder.warnings)
    assert not list((tmp_path / "png").rglob("unfinished-qpf.png"))


def _file(out, frame, product):
    """File one picture and its render receipt, as ``gpuwm render`` does."""
    path = out / "d02-3km" / product / "20260925" / f"{frame.name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG " + product.encode() + b" " + frame.name.encode())
    render_receipts.publish_invocation(
        root=out, engine="rust", requested_spec=product, written=[path],
        failures=[], skipped=[], layout="nested")


class _ReceiptedPasses:
    """Every pass files receipted pictures; the closing frame's pass also
    files its windows, leaves an unfinished staging PNG and exits 1."""

    def __init__(self):
        self.calls = []

    def __call__(self, command):
        self.calls.append(list(command))
        out = Path(command[command.index("--out") + 1])
        frame = _drawn_frame(command)
        _file(out, frame, "2m_temperature")
        if _closing(command):
            for product in ("qpf_1h", "qpf_total"):
                _file(out, frame, product)
            (out / "unfinished-wind.png").write_bytes(b"partial-png")
            return subprocess.CompletedProcess(list(command), 1, "", "died")
        return subprocess.CompletedProcess(list(command), 0, "", "")


def test_a_failed_closing_pass_publishes_only_what_its_receipts_name(tmp_path):
    launches, recorder, nest = _nest_hour(tmp_path, "all", _ReceiptedPasses())
    records = {Path(record["frame"]).name: record for record in recorder.reports}
    record = records[nest[60].name]
    assert sorted(entry["name"] for entry in record["written"]) == [
        f"d02-3km/{product}/20260925/{nest[60].name}.png"
        for product in ("2m_temperature", "qpf_1h", "qpf_total")]
    assert not list((tmp_path / "png").rglob("unfinished-wind.png"))
    # The pass failed, so the frame is incomplete and finalize draws it
    # again, beside the same frames of its hour.
    assert record["complete"] is False
    assert record["context"] == [str(nest[m]) for m in (0, 15, 30, 45)]
    assert len(launches[nest[60].name]) == 1
