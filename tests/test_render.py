"""``gpuwm render`` matplotlib-engine product tests -- CPU-only.

These pin the matplotlib engine explicitly: they are the fallback
engine's product contract, and must hold identically whether or not
the rust renderer is built (tests/test_render_rust.py covers that
engine and the auto selection).

The fixture is written by the project's own ``WrfoutWriter`` (WRF
dimensions, attributes, and Times records), so the file under test is
what ``gpuwm run`` produces, and the wrf package (pip ``wrf-rust``) reads
it exactly as it reads a production frame.  Assertions check that the
requested PNGs exist, are non-empty, and decode to plausible image
dimensions -- not that any pixel is pretty.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "wrf", reason="gpuwm render requires the wrf package (wrf-rust)")

import gpuwm.cli as cli
from gpuwm.io.wrfout import WrfoutWriter
from gpuwm.render import parse_products, parse_timeidx

_NZ, _NY, _NX = 4, 12, 16
_STAMPS = ("1974-04-03_18:00:00", "1974-04-03_19:00:00")


def _frame(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    lat = np.tile(np.linspace(38.0, 40.0, _NY)[:, None], (1, _NX))
    lon = np.tile(np.linspace(-98.0, -95.0, _NX)[None, :], (_NY, 1))
    return {
        "T": np.zeros((_NZ, _NY, _NX), np.float32),
        "MU": np.zeros((_NY, _NX), np.float32),
        "REFL_10CM": rng.uniform(-20.0, 65.0,
                                 (_NZ, _NY, _NX)).astype(np.float32),
        "T2": rng.uniform(280.0, 300.0, (_NY, _NX)).astype(np.float32),
        "U10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "V10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "RAINC": rng.uniform(0.0, 5.0, (_NY, _NX)).astype(np.float32),
        "RAINNC": rng.uniform(0.0, 30.0, (_NY, _NX)).astype(np.float32),
        "XLAT": lat.astype(np.float32),
        "XLONG": lon.astype(np.float32),
        "HGT": np.zeros((_NY, _NX), np.float32),
        "SINALPHA": np.zeros((_NY, _NX), np.float32),
        "COSALPHA": np.ones((_NY, _NX), np.float32),
    }


@pytest.fixture(scope="module")
def wrfout(tmp_path_factory):
    path = tmp_path_factory.mktemp("render") / \
        "wrfout_d02_1974-04-03_18-00-00.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ,
                      dx=1000.0, dy=1000.0) as writer:
        for index, stamp in enumerate(_STAMPS):
            writer.write_frame(stamp, _frame(seed=7 + index))
    return path


def _png_shape(path):
    import matplotlib.image
    return matplotlib.image.imread(path).shape


def test_render_all_products_all_frames(wrfout, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout), "--out", str(out)])
    assert rc == 0
    produced = sorted(p.name for p in out.glob("*.png"))
    expected = sorted(
        f"{product}_d02_{stamp.replace(':', '-')}.png"
        for product in ("refl", "t2", "wind10", "precip")
        for stamp in _STAMPS)
    assert produced == expected
    for png in out.glob("*.png"):
        assert png.stat().st_size > 5_000, png.name
        height, width, _ = _png_shape(png)
        assert height >= 400 and width >= 600, (png.name, height, width)


def test_render_product_and_frame_selection(wrfout, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout), "--products", "refl,t2",
                   "--timeidx", "1", "--out", str(out), "--dpi", "72"])
    assert rc == 0
    produced = sorted(p.name for p in out.glob("*.png"))
    stamp = _STAMPS[1].replace(":", "-")
    assert produced == sorted(
        [f"refl_d02_{stamp}.png", f"t2_d02_{stamp}.png"])


def test_render_missing_variable_fails_that_product_only(tmp_path):
    """A file with no REFL_10CM: refl fails loudly, t2 still renders."""
    path = tmp_path / "wrfout_d01_1974-04-03_18-00-00.nc"
    fields = _frame(seed=3)
    del fields["REFL_10CM"]
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ,
                      dx=1000.0, dy=1000.0) as writer:
        writer.write_frame(_STAMPS[0], fields)
    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(path), "--products", "refl,t2",
                   "--out", str(out)])
    assert rc == 1
    produced = [p.name for p in out.glob("*.png")]
    assert produced == [f"t2_d01_{_STAMPS[0].replace(':', '-')}.png"]


def test_render_timeidx_out_of_range_is_an_error(wrfout, tmp_path):
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout), "--products", "t2",
                   "--timeidx", "9", "--out", str(tmp_path / "png")])
    assert rc == 1
    assert not list((tmp_path / "png").glob("*.png"))


def test_parse_products():
    assert parse_products("all") == ("refl", "t2", "wind10", "precip")
    assert parse_products("precip,refl") == ("precip", "refl")
    assert parse_products("refl,refl") == ("refl",)
    with pytest.raises(ValueError, match="unknown product"):
        parse_products("refl,uh25")
    with pytest.raises(ValueError, match="no products"):
        parse_products(",")


def test_parse_timeidx():
    assert parse_timeidx("all") is None
    assert parse_timeidx("3") == 3
    with pytest.raises(ValueError, match="integer or 'all'"):
        parse_timeidx("last")
    with pytest.raises(ValueError, match="non-negative"):
        parse_timeidx("-1")


def test_unknown_product_is_exit_code_2(wrfout, tmp_path):
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout), "--products", "vorticity",
                   "--out", str(tmp_path / "png")])
    assert rc == 2


def test_wind10_fallback_is_native_units_via_getvar(tmp_path, monkeypatch):
    """No SINALPHA/COSALPHA: the wind panel falls back to raw U10/V10 in
    the model's native m/s -- every retrieval through wrf.getvar with no
    unit arguments and no local conversion factor."""
    import gpuwm.render as render
    import wrf as real_wrf

    path = tmp_path / "wrfout_d01_1974-04-03_18-00-00.nc"
    fields = _frame(seed=11)
    del fields["SINALPHA"]
    del fields["COSALPHA"]
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ,
                      dx=1000.0, dy=1000.0) as writer:
        writer.write_frame(_STAMPS[0], fields)

    calls: list[tuple[str, object]] = []

    class SpyWrf:
        def __getattr__(self, name):
            return getattr(real_wrf, name)

        @staticmethod
        def getvar(wf, name, timeidx=None, **kwargs):
            calls.append((name, kwargs.get("units")))
            return real_wrf.getvar(wf, name, timeidx=timeidx, **kwargs)

    monkeypatch.setattr(render, "_import_wrf", lambda: SpyWrf())
    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(path), "--products", "wind10",
                   "--out", str(out)])
    assert rc == 0
    assert list(out.glob("wind10_*.png"))
    # The earth-rotated primary was attempted (in knots, via getvar) ...
    assert ("uvmet10", "kt") in calls
    # ... and the fallback retrieved raw fields with NO unit arguments:
    # native m/s, labeled as such, no hand-rolled conversion.
    for name in ("U10", "V10", "wspd10"):
        assert (name, None) in calls, calls
        assert (name, "kt") not in calls, calls
    # The kt-per-(m/s) constant must not exist anywhere in the module.
    from pathlib import Path as _Path
    assert "1.9438" not in _Path(render.__file__).read_text()
