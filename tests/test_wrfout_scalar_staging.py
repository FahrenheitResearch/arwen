"""Scalar history fields retain dimensions and words through async staging."""
from datetime import datetime

import netCDF4
import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.io.wrfout import WrfoutWriter, _contiguous_preserving_shape


def _values(xp=np):
    # Non-integer clocks exercise their float32 words, including subnormals.
    return {
        "FMOIST_LASTTIME": xp.asarray(np.float32(751.719)),
        "FMOIST_NEXTTIME": xp.asarray(np.float32(1e-38)),
        "P_TOP": xp.asarray(np.float32(5000)),
        "ZNU": xp.asarray(np.array([0.75, 0.25], dtype=np.float32)),
    }


def _verify(path, expected):
    with netCDF4.Dataset(path) as ds:
        for name, value in expected.items():
            var = ds.variables[name]
            assert var.dimensions == (("Time",) if value.ndim == 0 else
                                      ("Time", "bottom_top"))
            actual = np.asarray(var[0], dtype=np.float32)
            assert actual.shape == value.shape
            assert actual.tobytes() == value.tobytes()


def test_scalar_history_keeps_zero_dimensional_axis(tmp_path):
    expected = _values()
    # Reproduced pre-fix conversion: a scalar becomes a one-element vector,
    # which the complete writer dimension contract correctly refuses.
    bad = np.ascontiguousarray(expected["FMOIST_LASTTIME"])
    assert bad.shape == (1,)
    with WrfoutWriter(tmp_path / "bad", nx=3, ny=4, nz=2, dx=1, dy=1) as writer:
        with pytest.raises(KeyError, match="1"):
            writer._dims_for("FMOIST_LASTTIME", bad.shape)
        writer.abort()
    fields = {name: _contiguous_preserving_shape(value)
              for name, value in expected.items()}
    path = tmp_path / "scalar"
    with WrfoutWriter(path, nx=3, ny=4, nz=2, dx=1, dy=1,
                      field_schema=fields) as writer:
        writer.write_frame("2026-07-16_00:00:00", fields)
    _verify(path, expected)


@requires_gpu
@pytest.mark.parametrize("mode", ("resident", "host_frame"))
def test_async_scalar_device_and_extra_fields_keep_words(monkeypatch, tmp_path, mode):
    import cupy as cp
    import gpuwm.io.wrfout as wrfout

    expected = _values()
    device = _values(cp)
    extras = {"P_TOP": expected["P_TOP"]}
    device.pop("P_TOP")
    # The whole production resident staging loop consumes these actual GPU
    # scalars. The prepared-frame path exercises its independent helper.
    monkeypatch.setattr(wrfout, "_device_state_frame", lambda *a, **k: device)
    path = tmp_path / mode
    writer = wrfout.AsyncDomainWrfoutWriter(nx=3, ny=4, nz=2, dx=1, dy=1,
                                           title="scalar staging", global_attrs={})
    try:
        if mode == "resident":
            writer.submit(path, datetime(2026, 7, 16), None, extra_fields=extras)
        else:
            writer.submit(path, datetime(2026, 7, 16), None,
                          frame=device, extra_fields=extras)
        writer.drain()
    finally:
        writer.close()
    _verify(path, expected)
