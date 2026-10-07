"""History frames written beside the step loop are the frames written in it.

THE BREAKAGE THIS PREVENTS: on the 9 km CONUS free leg the member workers
sat at about 175 % CPU compressing and copying the two-minute rain history
inside the step loop while the cards idled.  The frame is now read at the
alarm and written by a background thread (tools.da_member_leg
.HistoryWriter); the files must not change, and a failed write must still
fail the leg.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import da_cycle_prepared as dcp
from tools.da_member_leg import HistoryWriter


class Driver:
    def __init__(self, rng):
        self.fields = {}
        self._out = {"RAINC": rng.random((7, 9)).astype(np.float32),
                     "RAINNC": rng.random((7, 9)).astype(np.float32)}

    def output_fields(self):
        return self._out


def frame(path, reflectivity, writer=None):
    rng = np.random.default_rng(1)
    return dcp._write_rain_composite(
        Path(path), state=None, driver=Driver(rng), grid=SimpleNamespace(),
        cfg=SimpleNamespace(dt=30.0, dx=9000.0, dy=9000.0),
        elapsed_seconds=10920.0, exp=SimpleNamespace(), label="test",
        reflectivity=reflectivity, writer=writer)


def test_a_background_frame_is_the_inline_frame(tmp_path):
    rng = np.random.default_rng(0)
    refl = (rng.standard_normal((5, 7, 9)) * 20).astype(np.float32)
    frame(tmp_path/"inline"/"f.npz", refl)
    writer = HistoryWriter()
    future = frame(tmp_path/"bg"/"f.npz", refl, writer=writer)
    writer.close()
    assert future.done()
    a = np.load(tmp_path/"inline"/"f.npz")
    b = np.load(tmp_path/"bg"/"f.npz")
    assert a.files == b.files
    for key in a.files:
        assert a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes()
    assert np.array_equal(a["refl_colmax"], refl.max(axis=0))


def test_a_failed_background_frame_fails_the_leg(tmp_path):
    writer = HistoryWriter()
    blocker = tmp_path/"file"
    blocker.write_text("not a directory")
    frame(blocker/"f.npz", np.zeros((2, 3, 4), np.float32), writer=writer)
    with pytest.raises(OSError):
        writer.close()


def test_the_column_maximum_on_the_card_is_the_host_maximum():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(2)
    for dtype in (np.float32, np.float64):
        field = (rng.standard_normal((6, 11, 13)) * 30).astype(dtype)
        host = dcp.to_host(field).astype(np.float32).max(axis=0)
        card = dcp._column_max_float32(cp.asarray(field))
        assert card.dtype == host.dtype and card.tobytes() == host.tobytes()
