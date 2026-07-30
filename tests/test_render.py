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
from gpuwm.render import (DEFAULT_SOURCE_LABEL, domain_token, parse_products,
                          parse_timeidx, plot_context, resolution_token,
                          spacing_label)

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
        f"{product}_d02-1km_{stamp.replace(':', '-')}.png"
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
        [f"refl_d02-1km_{stamp}.png", f"t2_d02-1km_{stamp}.png"])


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
    assert produced == [f"t2_d01-1km_{_STAMPS[0].replace(':', '-')}.png"]


def test_render_timeidx_out_of_range_is_an_error(wrfout, tmp_path):
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout), "--products", "t2",
                   "--timeidx", "9", "--out", str(tmp_path / "png")])
    assert rc == 1
    assert not list((tmp_path / "png").glob("*.png"))


def test_resolution_tokens_are_trimmed_km_or_integer_metres():
    """Format pin, shared with the rust engine's own unit tests."""

    assert resolution_token(3_000.0) == "3km"
    assert resolution_token(12_000.0) == "12km"
    assert resolution_token(1_000.0) == "1km"
    # A 3:1 nest of a 12 km parent, twice over.
    assert resolution_token(1_333.3333) == "1.333km"
    assert resolution_token(1_500.0) == "1.5km"
    # Sub-kilometre reads as metres, not as 0.111 km.
    assert resolution_token(111.1111) == "111m"
    assert resolution_token(500.0) == "500m"
    # The boundary rounds before it chooses a unit.
    assert resolution_token(999.6) == "1km"
    assert resolution_token(999.4) == "999m"
    # Nothing usable is never guessed at.
    assert resolution_token(None) is None


def test_spacing_labels_spell_the_same_number():
    assert spacing_label(3_000.0) == "Δx 3 km"
    assert spacing_label(1_333.3333) == "Δx 1.333 km"
    assert spacing_label(111.1111) == "Δx 111 m"
    assert spacing_label(None) is None


def test_domain_token_pairs_domain_with_resolution():
    assert domain_token("d02", 3_000.0) == "d02-3km"
    assert domain_token("d05", 111.1111) == "d05-111m"
    # A file declaring no DX still separates its nests by domain.
    assert domain_token("d02", None) == "d02"


def test_plot_context_is_the_second_title_line():
    """Grid identity, spacing, valid time, and who produced it -- the
    matplotlib mirror of the rust engine's subtitle."""

    assert plot_context("d02", 3_000.0, "1974-04-03_18:00:00",
                        DEFAULT_SOURCE_LABEL) == (
        "d02-3km | Δx 3 km | valid 1974-04-03_18:00:00 | ArWen")
    # No DX in the file: the segment is absent, never invented.
    assert plot_context("d02", None, "1974-04-03_18:00:00", "WRF-ARW") == (
        "d02 | valid 1974-04-03_18:00:00 | WRF-ARW")


def test_titles_carry_the_spacing_and_the_source_label(wrfout, tmp_path,
                                                       monkeypatch):
    """The title text reaches matplotlib, not just the filename."""

    import gpuwm.render as render

    titles: list[str] = []
    real_finish = render._finish

    def spy(fig, axis, mappable, *, title, **kwargs):
        titles.append(title)
        return real_finish(fig, axis, mappable, title=title, **kwargs)

    monkeypatch.setattr(render, "_finish", spy)
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout),
                   "--products", "t2", "--timeidx", "0",
                   "--out", str(tmp_path / "png")])
    assert rc == 0
    assert titles == [
        "2 m temperature\nd02-1km | Δx 1 km | "
        "valid 1974-04-03_18:00:00 | ArWen"]

    titles.clear()
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout),
                   "--products", "t2", "--timeidx", "0",
                   "--source-label", "WRF-ARW 4.6.1",
                   "--out", str(tmp_path / "png2")])
    assert rc == 0
    assert titles[0].endswith("| WRF-ARW 4.6.1"), titles


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


def test_the_matplotlib_fallback_announces_itself(monkeypatch, capsys):
    """A fallback that does not say so is indistinguishable from the real
    thing until someone counts the products.

    A pilot spent a whole session on the four-product matplotlib path
    without noticing the 151-product rust catalog was simply not built.
    """
    import argparse
    from pathlib import Path
    from gpuwm import render as render_module

    assert render_module.fallback_notice("rust", "/path/to/rw_wrfbatch") is None
    # An explicit --engine matplotlib is a choice, not a surprise.
    assert render_module.fallback_notice("matplotlib", "requested") is None

    notice = render_module.fallback_notice(
        "matplotlib", "rust renderer not built (gpuwm doctor shows the "
        "build one-liner)")
    assert notice is not None
    # ONE line: a notice that scrolls is a notice that gets skimmed.
    assert "\n" not in notice, notice
    # The engine actually used ...
    assert "engine matplotlib" in notice
    assert "rust render engine not available" in notice
    # ... what that costs, against the rust catalog ...
    assert f"{len(render_module.PRODUCTS)} of the rust catalog's " \
           f"{render_module._RUST_CATALOG_PRODUCTS} products" in notice
    # ... and the exact command that fixes it.
    from gpuwm import rustwx
    assert rustwx.CARGO_BUILD_HINT in notice
    assert "cargo build --release --locked --offline" in notice

    # It reaches stderr on a real invocation, before anything else.
    monkeypatch.setattr(render_module, "_resolve_engine",
                        lambda requested: ("matplotlib", "rust renderer "
                                           "unusable: probe exited 1"))
    args = argparse.Namespace(
        pair=None, wrfout=[Path("nonexistent.nc")], timeidx="0",
        size="1200x900", engine="auto", products="all", list_products=False,
        out=Path("out"), dpi=110, heavy=False,
        source_label=render_module.DEFAULT_SOURCE_LABEL)
    monkeypatch.setattr(render_module, "render_wrfouts",
                        lambda *a, **k: ([], []))
    render_module.render_main(args)
    captured = capsys.readouterr()
    # (the stub writes nothing, so the exit code is the empty-output one;
    # what is under test is that the notice was printed, and printed
    # before any product line.)
    assert "rust render engine not available" in captured.err
    assert "probe exited 1" in captured.err


# ---------------------------------------------------------------------------
# Degradation: identical to the rust engine's, or it is not one behaviour
# ---------------------------------------------------------------------------
# The mirror of tests/test_render_rust.py's
# test_unknown_domain_identity_keeps_the_native_grid_token.  Matplotlib
# guessed `d01` for a file carrying neither GRID_ID nor a wrfout_dNN
# name, which labelled an anonymous input as the parent domain on no
# evidence and -- because the token is what separates one nest's output
# names from another's -- let two such inputs at one valid time
# overwrite each other, the exact failure this release exists to end.

def test_domain_token_degrades_exactly_as_the_rust_engine_does():
    """The three steps, stated once, asserted on the matplotlib side."""

    from gpuwm.render import NATIVE_GRID_SLUG

    # 1. Identity + usable DX -> dNN-resolution.
    assert domain_token("d02", 3_000.0) == "d02-3km"
    # 2. Identity, no usable DX -> bare dNN; the domain alone still
    #    separates the nests, which is the token's whole job.
    assert domain_token("d02", None) == "d02"
    # 3. No identity -> the generic slug, and NO resolution appended: a
    #    resolution is not an identity.
    assert domain_token(None, 3_000.0) == NATIVE_GRID_SLUG
    assert domain_token(None, None) == NATIVE_GRID_SLUG
    # Spelled the same as rust spells it, so one word means one thing.
    assert NATIVE_GRID_SLUG == "native_grid"


def test_unknown_domain_identity_keeps_the_native_grid_token(tmp_path):
    """No GRID_ID, no wrfout_dNN name: the token degrades, never guesses."""

    path = tmp_path / "some_model_output.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ,
                      dx=1000.0, dy=1000.0) as writer:
        writer.write_frame(_STAMPS[0], _frame(seed=21))

    import netCDF4
    with netCDF4.Dataset(path) as ds:
        assert "GRID_ID" not in ds.ncattrs(), (
            "the fixture must be anonymous for this test to mean anything")

    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(path),
                   "--products", "t2", "--out", str(out)])
    assert rc == 0
    produced = [p.name for p in out.glob("*.png")]
    assert produced, "the anonymous-domain file rendered nothing"
    for name in produced:
        assert "native_grid" in name, name
        # The old guess, in either spelling.
        assert "_d01_" not in name and "_d01-" not in name, name


def test_two_anonymous_inputs_refuse_to_overwrite_each_other(tmp_path):
    """The collision the guessed `d01` used to hide.

    Both files are unidentifiable, so both resolve to the same
    `native_grid` token -- and at one valid time, for one product, to
    one output name.  Silently writing the second over the first is the
    v1.0.0 data-loss bug wearing a different token, so the second is
    refused with the reason and the remedy.
    """

    paths = []
    for index in (0, 1):
        path = tmp_path / f"anonymous_{index}.nc"
        with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ,
                          dx=1000.0, dy=1000.0) as writer:
            writer.write_frame(_STAMPS[0], _frame(seed=31 + index))
        paths.append(str(path))

    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", *paths,
                   "--products", "t2", "--out", str(out)])
    # One PNG, and a loud non-zero exit -- never two inputs and one
    # silent survivor.
    assert rc == 1
    assert len(list(out.glob("*.png"))) == 1


def test_a_grid_id_out_of_domain_range_is_not_identity(tmp_path):
    """`GRID_ID = 0` is not domain zero; it is no evidence.

    The rust side accepts 1..=99 and this must agree, or one engine
    invents a `d00` the other refuses.
    """

    from gpuwm.render import _domain_tag

    path = tmp_path / "odd_grid_id.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0,
                      dy=1000.0, global_attrs={"GRID_ID": 0}) as writer:
        writer.write_frame(_STAMPS[0], _frame(seed=41))
    assert _domain_tag(path) is None

    named = tmp_path / "wrfout_d03_1974-04-03_18-00-00.nc"
    with WrfoutWriter(named, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0,
                      dy=1000.0, global_attrs={"GRID_ID": 0}) as writer:
        writer.write_frame(_STAMPS[0], _frame(seed=42))
    # GRID_ID unusable, so the filename is the evidence -- rust's
    # precedence exactly.
    assert _domain_tag(named) == "d03"
