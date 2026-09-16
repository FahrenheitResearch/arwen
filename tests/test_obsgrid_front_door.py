"""The observation-grid lane, finally reached from a door.

``rw_obsgrid`` renders five products from a ``gpuwm-obs.radar-grid.v1``
observation volume and has existed, wrapped and handshaken, with no
caller at all: the two DA render tools printed it as a HINT and drew the
same fields in matplotlib.  These cases stub the binary, so they run with
no build.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gpuwm import rustwx_lanes


def test_the_lane_resolves_its_engine_through_the_shared_contract(monkeypatch):
    monkeypatch.setattr(rustwx_lanes, "find_obsgrid_bin", lambda: None)
    with pytest.raises(RuntimeError) as excinfo:
        rustwx_lanes.resolve_obsgrid_engine("rust")
    message = str(excinfo.value)
    assert rustwx_lanes.OBSGRID_NAME in message
    assert rustwx_lanes.CARGO_BUILD_HINT.split()[0] in message
    with pytest.raises(RuntimeError):
        rustwx_lanes.resolve_obsgrid_engine("auto")
    assert rustwx_lanes.resolve_obsgrid_engine("matplotlib") == (
        "matplotlib", "requested")


def _stage_engine(monkeypatch, tmp_path, written):
    monkeypatch.setattr(rustwx_lanes, "find_obsgrid_bin",
                        lambda: tmp_path / "rw_obsgrid")
    monkeypatch.setattr(rustwx_lanes, "probe_obsgrid_bin",
                        lambda path: (True, "--abi matches the contract"))

    def fake_run(engine, obs, *, out_dir, **kwargs):
        paths = []
        for slug in rustwx_lanes.OBSGRID_PRODUCTS:
            path = Path(out_dir) / f"{slug}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes.fromhex("89504e470d0a1a0a"))
            paths.append(path)
        written.extend(paths)
        return paths, [], [], []

    monkeypatch.setattr(rustwx_lanes, "run_obsgrid_renderer", fake_run)


def test_the_level2_gallery_drives_the_engine_and_stays_quiet_about_fallback(
        tmp_path, monkeypatch, capsys):
    from tools import da_level2_render

    written: list[Path] = []
    _stage_engine(monkeypatch, tmp_path, written)
    gallery = da_level2_render.Level2Gallery(tmp_path / "out", dpi=72)
    drawn, reason = gallery.engine_products(tmp_path / "obs.nc", engine="auto")
    assert len(drawn) == len(rustwx_lanes.OBSGRID_PRODUCTS)
    assert reason == ""
    assert sorted(path.name for path in drawn) == sorted(
        f"{slug}.png" for slug in rustwx_lanes.OBSGRID_PRODUCTS)


def test_the_level2_engine_delivery_leaves_a_render_receipt(
        tmp_path, monkeypatch):
    """Five PNGs and no record of them is a delivery nothing can read."""

    import json

    from tools import da_level2_render

    written: list[Path] = []
    _stage_engine(monkeypatch, tmp_path, written)
    out = tmp_path / "out"
    gallery = da_level2_render.Level2Gallery(out, dpi=72)
    drawn, reason = gallery.engine_products(tmp_path / "obs.nc", engine="auto")
    assert drawn and reason == ""
    receipt = out / "render-summary.json"
    assert receipt.is_file(), sorted(path.name for path in out.iterdir())
    summary = json.loads(receipt.read_text(encoding="utf-8"))
    assert summary["schema"] == "gpuwm.render-summary.v1", summary
    assert summary["rendered_png_count"] == len(
        rustwx_lanes.OBSGRID_PRODUCTS), summary
    assert summary["invocation_count"] == 1, summary


def test_the_level2_gallery_names_the_reason_when_it_falls_back(
        tmp_path, monkeypatch):
    from tools import da_level2_render

    monkeypatch.setattr(rustwx_lanes, "find_obsgrid_bin", lambda: None)
    gallery = da_level2_render.Level2Gallery(tmp_path / "out", dpi=72)
    drawn, reason = gallery.engine_products(tmp_path / "obs.nc", engine="auto")
    assert drawn == []
    assert rustwx_lanes.OBSGRID_NAME in reason


def test_the_nowcast_gallery_reports_the_engine_it_could_not_reach(
        tmp_path, monkeypatch):
    from tools import da_nowcast_render

    gallery = object.__new__(da_nowcast_render.Gallery)
    monkeypatch.setattr(rustwx_lanes, "find_obsgrid_bin", lambda: None)
    assert rustwx_lanes.OBSGRID_NAME in gallery.engine_resolution("auto")
    monkeypatch.setattr(rustwx_lanes, "find_obsgrid_bin",
                        lambda: tmp_path / "rw_obsgrid")
    monkeypatch.setattr(rustwx_lanes, "probe_obsgrid_bin",
                        lambda path: (True, "ok"))
    assert gallery.engine_resolution("auto") == ""
    assert "matplotlib" in gallery.engine_resolution("matplotlib")


def _quiet_gallery(tmp_path):
    """A ``Gallery`` with every figure method stubbed out.

    The sheets themselves are not what is under test here; what the run
    SAYS about how they were drawn is.
    """

    from tools import da_nowcast_render

    gallery = object.__new__(da_nowcast_render.Gallery)
    gallery.out = tmp_path / "gallery"
    gallery.manifest = []
    for name in ("fig_lead", "fig_strip", "fig_numbers", "fig_scorecard",
                 "fig_structure"):
        setattr(gallery, name, lambda *a, **k: None)
    gallery.fig_verify = lambda *a, **k: []
    gallery.write_page = lambda *a, **k: None
    return gallery


def test_the_nowcast_run_says_the_fallback_drew_even_where_the_engine_built(
        tmp_path, monkeypatch, capsys):
    """One lane, one story about one run.

    ``Gallery.render`` composes every weather-field panel itself and
    drives ``rw_obsgrid`` nowhere, so the DEPRECATED FALLBACK sentence is
    true on every box.  Gating it on whether the engine resolved made the
    run SILENT about the fallback exactly where the engine was built,
    which is where a reader is most likely to believe the panels came
    from it.
    """

    monkeypatch.setattr(rustwx_lanes, "find_obsgrid_bin",
                        lambda: tmp_path / "rw_obsgrid")
    monkeypatch.setattr(rustwx_lanes, "probe_obsgrid_bin",
                        lambda path: (True, "--abi matches the contract"))
    _quiet_gallery(tmp_path).render(engine="auto")
    said = capsys.readouterr().out
    assert "DEPRECATED" in said and "FALLBACK" in said, said
    assert "drives no engine" in said, said
    # Nothing to fix on this box, so nothing is claimed to be unreachable.
    assert "not reachable here" not in said, said


def test_the_nowcast_run_names_the_missing_engine_beside_the_same_sentence(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rustwx_lanes, "find_obsgrid_bin", lambda: None)
    _quiet_gallery(tmp_path).render(engine="auto")
    said = capsys.readouterr().out
    assert "DEPRECATED" in said and "FALLBACK" in said, said
    assert "not reachable here" in said, said
    assert rustwx_lanes.OBSGRID_NAME in said, said


def test_both_da_tools_offer_the_engine_at_their_door():
    import inspect

    from tools import da_level2_render, da_nowcast_render

    for module in (da_level2_render, da_nowcast_render):
        source = inspect.getsource(module.main)
        assert '"--engine"' in source, module.__name__
        assert '"auto", "rust", "matplotlib"' in source, module.__name__
