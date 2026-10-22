"""The run's ledger of written history frames (gpuwm.io.wrfout.HISTORY_LEDGER).

THE BREAKAGE: a host removed earlier history frames to save disk (an 8x5090 host,
2026-10-21), and the renderer could not tell a removed frame from a cadence
that never wrote one; a Kuchera snowfall total summed across the hole takes
one end temperature for every interval it spans.  The writer names every
history frame it publishes in a ledger beside them, and the renderer reads
the same file name.
"""

from pathlib import Path

import numpy as np

from gpuwm.io import wrfout


def _write(folder: Path, stamp: str) -> Path:
    path = folder / f"wrfout_d01_{stamp.replace(':', '_')}"
    with wrfout.WrfoutWriter(path, nx=4, ny=3, nz=2, dx=1000.0, dy=1000.0,
                             global_attrs={}) as writer:
        writer.write_frame(stamp, {"T2": np.full((3, 4), 290.0, np.float32)})
    return path


def test_every_published_history_frame_is_in_the_ledger(tmp_path):
    first = _write(tmp_path, "2026-01-01_00:00:00")
    second = _write(tmp_path, "2026-01-01_01:00:00")
    lines = (tmp_path / wrfout.HISTORY_LEDGER).read_text(encoding="ascii").split()
    assert set(lines) == {first.name, second.name}


def test_only_history_frames_are_recorded(tmp_path):
    wrfout.record_history_written(tmp_path / "wrfinput_d01")
    wrfout.record_history_written(tmp_path / "wrfout_d01_2026-01-01_00_00_00.partial")
    assert not (tmp_path / wrfout.HISTORY_LEDGER).exists()
    wrfout.record_history_written(tmp_path / "wrfout_d02_2026-01-01_00_15_00")
    assert (tmp_path / wrfout.HISTORY_LEDGER).read_text(encoding="ascii") == (
        "wrfout_d02_2026-01-01_00_15_00\n")


def test_the_renderer_reads_the_same_ledger():
    snowfall = (Path(__file__).resolve().parents[1] / "tools" / "rustwx" / "crates"
                / "rw-wrfbatch" / "src" / "snowfall.rs").read_text(encoding="utf-8")
    assert f'pub const HISTORY_LEDGER: &str = "{wrfout.HISTORY_LEDGER}";' in snowfall
