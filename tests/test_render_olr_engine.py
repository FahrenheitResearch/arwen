"""The real ``rw_wrfbatch`` draws ``--products olr`` from a history's OLR.

The CPU half (tests/test_render_olr_alias.py) pins the spelling; this asks
the engine this checkout resolves, over a history that carries ``OLR``,
and skips with the reason where no usable engine resolves.
"""
from __future__ import annotations

import datetime
from types import SimpleNamespace

import numpy as np

import gpuwm.cli as cli
from gpuwm.io.wrfout import WrfoutWriter, wrf_global_attrs
from test_render_rust import _NX, _NY, _NZ, _delivered, _frame, needs_renderer


def _olr_history(path):
    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(
        grid, datetime.datetime(1974, 4, 3, 18), grid_id=2, parent_id=1,
        i_parent_start=5, j_parent_start=5, parent_grid_ratio=3, dt=6.0)
    frame = _frame(seed=11)
    frame["OLR"] = np.random.default_rng(3).uniform(
        100.0, 300.0, (_NY, _NX)).astype(np.float32)
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0,
                      global_attrs=attrs) as writer:
        writer.write_frame("1974-04-03_18:00:00", frame)
    return path


@needs_renderer
def test_render_olr_draws_the_stored_olr_plane(tmp_path, capsys):
    wrfout = _olr_history(tmp_path / "wrfout_d02_1974-04-03_18-00-00.nc")
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "olr", "--timeidx", "0",
                   "--size", "800x600", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    produced = _delivered(out)
    assert len(produced) == 1, produced
    assert "var_wrf_olr" in produced[0], produced
    assert "unknown product" not in captured.err
