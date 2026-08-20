"""The writer side of ``[output]``: what reaches the tape, and the header.

Two things have to be true and neither is provable from the grammar
tests: a dropped variable must not be STAGED (the D2H it would cost is
most of what the feature buys on a big domain), and the file it lands in
must SAY what was dropped so the render front door can name the cause
instead of reporting an absence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import netCDF4
import pytest

from gpuwm.io import history_selection as hs
from gpuwm.io.wrfout import AsyncDomainWrfoutWriter, WrfoutWriter


NX, NY, NZ = 12, 10, 8


def _frame() -> dict[str, np.ndarray]:
    """A small but structurally real history frame, in writer order."""
    def mass(): return np.zeros((NZ, NY, NX), np.float32)

    return {
        "T": mass(), "U": np.zeros((NZ, NY, NX + 1), np.float32),
        "V": np.zeros((NZ, NY + 1, NX), np.float32),
        "W": np.zeros((NZ + 1, NY, NX), np.float32),
        "PH": np.zeros((NZ + 1, NY, NX), np.float32),
        "PHB": np.zeros((NZ + 1, NY, NX), np.float32),
        "MU": np.zeros((NY, NX), np.float32),
        "MUB": np.zeros((NY, NX), np.float32),
        "HGT": np.zeros((NY, NX), np.float32),
        "QVAPOR": mass(), "QCLOUD": mass(), "QRAIN": mass(),
        "REFL_10CM": mass(),
        "T2": np.zeros((NY, NX), np.float32),
        "U10": np.zeros((NY, NX), np.float32),
        "V10": np.zeros((NY, NX), np.float32),
        "XLAT": np.zeros((NY, NX), np.float32),
        "XLONG": np.zeros((NY, NX), np.float32),
        "ZNU": np.zeros((NZ,), np.float32),
        "ZNW": np.zeros((NZ + 1,), np.float32),
        "P_TOP": np.asarray(5000.0, np.float32),
    }


def _write(path, frame, *, global_attrs=None):
    with WrfoutWriter(path, nx=NX, ny=NY, nz=NZ, dx=3000.0, dy=3000.0,
                      title="gpuwm test",
                      global_attrs=global_attrs) as writer:
        writer.write_frame("1974-04-03_12:00:00", frame)
    return Path(path)


def test_a_trimmed_tape_is_smaller_and_the_variable_is_simply_absent(tmp_path):
    """The storage claim, measured on real bytes rather than asserted."""
    selection = hs.HistorySelection.from_mapping(
        {"history_drop": ["QCLOUD", "QRAIN", "REFL_10CM"]})
    frame = _frame()
    produced = tuple(frame)

    full = _write(tmp_path / "full.nc", frame)
    trimmed_frame = {name: frame[name] for name in selection.select(produced)}
    trimmed = _write(tmp_path / "trim.nc", trimmed_frame,
                     global_attrs=selection.wrfout_attrs(produced))

    assert trimmed.stat().st_size < full.stat().st_size
    with netCDF4.Dataset(trimmed) as ds:
        names = set(ds.variables)
        assert not ({"QCLOUD", "QRAIN", "REFL_10CM"} & names)
        # Structure survives: the file is still a wrfout.
        assert {"Times", "XLAT", "XLONG", "ZNU", "ZNW", "T2"} <= names
        assert ds.GPUWM_HISTORY_PRESET == "full"
        assert "REFL_10CM" in ds.GPUWM_HISTORY_DROPPED


def test_a_default_run_stamps_no_history_attribute(tmp_path):
    """Every wrfout written before this surface existed keeps its header."""
    path = _write(tmp_path / "full.nc", _frame(),
                  global_attrs=hs.FULL.wrfout_attrs(tuple(_frame())))
    with netCDF4.Dataset(path) as ds:
        assert not [name for name in ds.ncattrs()
                    if name.startswith("GPUWM_HISTORY_")]


# ------------------------------------------- the async writer's own plan

def _writer_shell(selection):
    """CPU-only AsyncDomainWrfoutWriter shell: no CuPy stream allocation."""
    writer = object.__new__(AsyncDomainWrfoutWriter)
    writer.global_attrs = {"TITLE": "gpuwm test"}
    writer.history_selection = selection
    return writer


def test_the_writer_plan_names_what_it_will_write_and_what_it_shed():
    writer = _writer_shell(hs.HistorySelection.from_mapping(
        {"history_drop": ["QCLOUD"]}))
    produced = ("T", "QCLOUD", "T2", "XLAT")
    keep, attrs = writer._history_plan(produced)
    assert "QCLOUD" not in keep
    assert {"T", "T2", "XLAT"} <= keep
    assert attrs["GPUWM_HISTORY_DROPPED"] == "QCLOUD"


def test_the_writer_plan_of_a_full_default_keeps_everything_and_stamps_none():
    writer = _writer_shell(None)
    produced = ("T", "QCLOUD", "T2")
    keep, attrs = writer._history_plan(produced)
    assert set(produced) <= keep
    assert attrs == {}


@pytest.mark.gpu
def test_a_dropped_field_is_never_staged_off_the_device(tmp_path):
    """The D2H a dropped 3-D field would cost is what the feature buys.

    Filtering at write time would still copy every dropped volume host-ward
    first, which on a real domain is the majority of the cost the user is
    trying to avoid.  The assertion is against the TICKET the writer
    builds: a dropped name may not appear in the staged host frame at all.
    """
    import cupy as cp

    writer = object.__new__(AsyncDomainWrfoutWriter)
    writer.global_attrs = {}
    writer.history_selection = hs.HistorySelection.from_mapping(
        {"history_drop": ["QCLOUD"]})
    writer.stream = cp.cuda.Stream(non_blocking=True)
    admitted = []
    writer._admit = admitted.append

    frame = {"T": cp.zeros((NZ, NY, NX), cp.float32),
             "QCLOUD": cp.zeros((NZ, NY, NX), cp.float32),
             "T2": cp.zeros((NY, NX), cp.float32)}
    writer._admit_host_frame(
        tmp_path / "x.nc", __import__("datetime").datetime(1974, 4, 3, 12),
        frame, extra_fields={"XLAT": np.zeros((NY, NX), np.float32)},
        refl_field=None, producer=cp.cuda.get_current_stream())

    assert len(admitted) == 1
    staged = admitted[0].fields
    assert "QCLOUD" not in staged
    assert {"T", "T2", "XLAT"} <= set(staged)
    assert "QCLOUD" in admitted[0].global_attrs["GPUWM_HISTORY_DROPPED"]
