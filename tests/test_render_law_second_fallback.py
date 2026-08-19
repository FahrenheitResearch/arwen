"""The render law permits ONE fallback, and `gpuwm render` is not it.

CLAUDE.md, Drew 2026-08-06: "Weather-field product plots come from the
real Rust renderer ``rw_wrfbatch`` driven through ``gpuwm.rustwx`` ...
``da_nowcast_render.py`` is a fallback only.  matplotlib, ``wrf.plot``
and cartopy are not allowed for weather fields".

The concrete breakage these tests prevent (hidden-scope audit F7,
2026-08-18): on a box where bridge staging failed, ``gpuwm render
WRFOUT`` at the default ``--engine auto`` drew composite reflectivity,
2 m temperature, 10 m wind, accumulated precipitation and OLR with
matplotlib and exited 0.  Five weather fields, drawn by a second
fallback the law does not name, reported as success -- so the operator
who ran a forecast and got pictures had no way to learn that the
production catalog had never run.

``da_nowcast_render.py`` does not serve these products (it composes
multi-panel sheets from a ``tools/da_nowcast.py`` CASE DIRECTORY, not
single-panel products from a wrfout), so the lawful answer when the
renderer is unresolvable is a refusal that names the missing artifact
and the staging remedy, at a nonzero exit.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.cli as cli
from gpuwm import render as render_module
from gpuwm.io.wrfout import WrfoutWriter, wrf_global_attrs

_NZ, _NY, _NX = 4, 12, 16


def _frame() -> dict:
    rng = np.random.default_rng(11)
    lat = np.tile(np.linspace(38.0, 40.0, _NY)[:, None], (1, _NX))
    lon = np.tile(np.linspace(-98.0, -95.0, _NX)[None, :], (_NY, 1))
    return {
        "T": np.zeros((_NZ, _NY, _NX), np.float32),
        "MU": np.zeros((_NY, _NX), np.float32),
        "REFL_10CM": rng.uniform(-20.0, 65.0,
                                 (_NZ, _NY, _NX)).astype(np.float32),
        "T2": rng.uniform(280.0, 300.0, (_NY, _NX)).astype(np.float32),
        "Q2": rng.uniform(0.004, 0.012, (_NY, _NX)).astype(np.float32),
        "PSFC": rng.uniform(96000.0, 98000.0, (_NY, _NX)).astype(np.float32),
        "U10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "V10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "RAINC": rng.uniform(0.0, 5.0, (_NY, _NX)).astype(np.float32),
        "RAINNC": rng.uniform(0.0, 30.0, (_NY, _NX)).astype(np.float32),
        "OLR": rng.uniform(120.0, 300.0, (_NY, _NX)).astype(np.float32),
        "XLAT": lat.astype(np.float32),
        "XLONG": lon.astype(np.float32),
        "HGT": np.zeros((_NY, _NX), np.float32),
        "SINALPHA": np.zeros((_NY, _NX), np.float32),
        "COSALPHA": np.ones((_NY, _NX), np.float32),
    }


@pytest.fixture(scope="module")
def wrfout(tmp_path_factory) -> Path:
    """A real wrfout, written by the product's own writer."""

    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(
        grid, datetime.datetime(1974, 4, 3, 18), grid_id=2, parent_id=1,
        i_parent_start=5, j_parent_start=5, parent_grid_ratio=3, dt=6.0)
    path = (tmp_path_factory.mktemp("render-law")
            / "wrfout_d02_1974-04-03_18-00-00.nc")
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0,
                      global_attrs=attrs) as writer:
        writer.write_frame("1974-04-03_18:00:00", _frame())
    return path


def _no_renderer(monkeypatch) -> None:
    """A box where bridge staging failed: nothing resolves."""

    from gpuwm import rustwx

    monkeypatch.setattr(rustwx, "find_renderer", lambda: None)


def _forbid_matplotlib_drawing(monkeypatch) -> None:
    """Any matplotlib weather field drawn from here on is the defect."""

    def refuse(*_args, **_kwargs):
        raise AssertionError(
            "gpuwm render drew weather fields with the matplotlib engine "
            "without an explicit --engine matplotlib; that is the second "
            "fallback the render law forbids")

    monkeypatch.setattr(render_module, "render_wrfouts", refuse)


def test_auto_refuses_when_the_renderer_is_not_staged(
        wrfout, tmp_path, monkeypatch, capsys):
    """The bare default: no renderer, no matplotlib weather fields, rc != 0."""

    _no_renderer(monkeypatch)
    _forbid_matplotlib_drawing(monkeypatch)
    out = tmp_path / "out"
    rc = cli.main(["render", str(wrfout), "--out", str(out)])
    captured = capsys.readouterr()
    assert rc != 0, (
        "a render that drew nothing reported success; the exit code must "
        "reflect what happened")
    text = captured.err + captured.out
    assert "rw_wrfbatch" in text, (
        "the refusal must name the artifact that is missing")
    assert "fetch-bridges" in text, (
        "the refusal must name the remedy that stages it")
    assert not list(out.rglob("*.png")), (
        "a refused render wrote weather-field imagery anyway")


def test_auto_refuses_when_the_renderer_fails_its_contract(
        wrfout, tmp_path, monkeypatch, capsys):
    """A staged binary from another checkout is not a licence to degrade."""

    monkeypatch.setattr(
        render_module, "renderer_refusal",
        lambda _renderer: "--abi does not match the render contract")
    from gpuwm import rustwx

    monkeypatch.setattr(rustwx, "find_renderer",
                        lambda: tmp_path / "rw_wrfbatch.exe")
    _forbid_matplotlib_drawing(monkeypatch)
    rc = cli.main(["render", str(wrfout), "--out", str(tmp_path / "o")])
    captured = capsys.readouterr()
    assert rc != 0
    text = captured.err + captured.out
    assert "--abi does not match the render contract" in text
    assert "fetch-bridges" in text


def test_auto_never_resolves_to_the_matplotlib_engine(monkeypatch, tmp_path):
    """`auto` has exactly two answers now: the rust engine, or a refusal."""

    _no_renderer(monkeypatch)
    with pytest.raises(RuntimeError) as excinfo:
        render_module._resolve_engine("auto")
    assert "rw_wrfbatch" in str(excinfo.value)


def test_the_chain_gate_reports_no_engine_rather_than_matplotlib(monkeypatch):
    """`gpuwm go`'s render leg must not chain into the second fallback."""

    _no_renderer(monkeypatch)
    engine, why = render_module.drawable_engine()
    assert engine is None, (
        f"the chain would have rendered weather fields with {engine!r}")
    assert "rw_wrfbatch" in why


def test_an_explicit_matplotlib_request_announces_itself_as_a_workaround():
    """Opt-in is not silence: `Fixed means default` (Drew 2026-08-10).

    The engine is reachable only by name now, and every run of it says
    so in the project's own workaround voice, so a reader can never
    mistake its five products for the production catalog.
    """

    notice = render_module.matplotlib_workaround_notice("matplotlib")
    assert notice is not None
    assert notice.startswith("WORKAROUND:")
    assert "render law" in notice
    assert "rw_wrfbatch" in notice
    assert render_module.matplotlib_workaround_notice("rust") is None


def test_the_explicit_workaround_line_reaches_stderr(wrfout, tmp_path,
                                                     capsys):
    """Against the front door, not against the function."""

    pytest.importorskip("wrf")
    rc = cli.main(["render", str(wrfout), "--engine", "matplotlib",
                   "--products", "t2", "--out", str(tmp_path / "mpl")])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "WORKAROUND:" in captured.err
