"""The ordinary one-domain writer observes the live store at every output."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import sys

from gpuwm import runtime
from gpuwm.io import wrfout


@pytest.mark.parametrize("expect_refl", [False, True])
def test_single_output_uses_current_store_without_reading_stale_device_fields(monkeypatch, tmp_path, expect_refl):
    monkeypatch.setitem(sys.modules, "cupy", None)
    live = {"T": np.array([[[3.0]]], dtype=np.float32),
            "RAINNC": np.array([[2.0]], dtype=np.float32),
            "REFL_10CM": np.array([[[15.0]]], dtype=np.float32)}
    observed = []
    closed = []

    class Store:
        def history_fields(self):
            return dict(live)

    class ForbiddenState:
        _streamed_domain = Store()

        @property
        def physics(self):
            raise AssertionError("output read the stale prepared driver")

        @property
        def qv(self):
            raise AssertionError("output read stale resident reflectivity operands")

    class Writer:
        def __init__(self, path, **kwargs):
            self.path = path

        def __enter__(self):
            return self

        def write_frame(self, time, frame):
            observed.append((time, {key: value.copy() for key, value in frame.items()}))

        def __exit__(self, *args):
            closed.append(self.path)

    def forbidden(*args, **kwargs):
        raise AssertionError("output read the stale prepared state")

    monkeypatch.setattr(wrfout, "state_frame", forbidden)
    monkeypatch.setattr(wrfout, "WrfoutWriter", Writer)
    monkeypatch.setattr(runtime, "_metadata_frame", lambda *args: {"HGT": np.array([[7.0]])})
    monkeypatch.setattr(runtime, "_global_wrf_attrs", lambda *args, **kwargs: {})
    monkeypatch.setattr(runtime, "soil_layer_count", lambda cfg: 4)
    prepared = SimpleNamespace(
        initial_result=SimpleNamespace(state=ForbiddenState(), coord=object()),
        cfg=SimpleNamespace(nx=1, ny=1, nz=1, dx=1.0, dy=1.0, mp_physics=10),
        grid=object(), static_fields={})
    start = datetime(2026, 5, 29, 18)
    for index in range(2):
        runtime.write_case_output(prepared, tmp_path, start + timedelta(hours=index),
                                  start_time=start, title="live store", expect_refl_10cm=expect_refl)
        assert len(closed) == index + 1
        live["T"] += np.float32(10.0)
        live["RAINNC"] += np.float32(1.0)
    assert [float(frame["T"].item()) for _, frame in observed] == [3.0, 13.0]
    assert [float(frame["RAINNC"].item()) for _, frame in observed] == [2.0, 3.0]
    assert [float(frame["REFL_10CM"].item()) for _, frame in observed] == [15.0, 15.0]
    assert observed[0][0] != observed[1][0]
