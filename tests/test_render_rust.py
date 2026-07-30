"""``gpuwm render --engine rust`` and ``--pair`` -- CPU-only tests.

The rust-engine tests run against the real vendored renderer binary
(``tools/rustwx``) and honestly skip when it is not built; nothing here
mocks the engine.  The fixture wrfout is written by the project's own
``WrfoutWriter`` with the production global-attribute profile
(``wrf_global_attrs``: Lambert projection, START_DATE, domain
topology), because that is what ``gpuwm run`` writes and what the rust
importer's fail-closed preflight requires.  Its second frame is on the
half hour -- every rust-engine test doubles as the exact-time
(sub-hourly) regression test that the source renderer refused.

PNG dimensions are read from the IHDR chunk directly so the assertions
add no image-library dependency.
"""

from __future__ import annotations

import datetime
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.cli as cli
from gpuwm import rustwx
from gpuwm.io.wrfout import WrfoutWriter, wrf_global_attrs
from gpuwm.render import parse_products_rust, parse_size

_NZ, _NY, _NX = 4, 12, 16
_STAMPS = ("1974-04-03_18:00:00", "1974-04-03_18:30:00")
_LEADS = ("lead_000h00m00s", "lead_000h30m00s")

RENDERER = rustwx.find_renderer()
needs_renderer = pytest.mark.skipif(
    RENDERER is None,
    reason="rust renderer not built (cd tools/rustwx && cargo build "
           "--release --locked --offline)")


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
        "Q2": rng.uniform(0.004, 0.012, (_NY, _NX)).astype(np.float32),
        "PSFC": rng.uniform(96000.0, 98000.0,
                            (_NY, _NX)).astype(np.float32),
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


def _write_wrfout(path, stamps, *, grid_id=2, dx=1000.0):
    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(
        grid, datetime.datetime(1974, 4, 3, 18), grid_id=grid_id,
        parent_id=max(grid_id - 1, 1), i_parent_start=5, j_parent_start=5,
        parent_grid_ratio=3, dt=6.0)
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=dx, dy=dx,
                      global_attrs=attrs) as writer:
        for index, stamp in enumerate(stamps):
            writer.write_frame(stamp, _frame(seed=7 + index))
    return path


@pytest.fixture(scope="module")
def wrfout(tmp_path_factory):
    """Two frames, the second on the half hour (exact-time axis)."""

    return _write_wrfout(
        tmp_path_factory.mktemp("render-rust") /
        "wrfout_d02_1974-04-03_18-00-00.nc", _STAMPS)


@pytest.fixture(scope="module")
def wrfout_hourly(tmp_path_factory):
    """Two WHOLE-hour frames -- the windowed lane's admissible axis."""

    return _write_wrfout(
        tmp_path_factory.mktemp("render-rust-hourly") /
        "wrfout_d02_1974-04-03_18-00-00.nc",
        ("1974-04-03_18:00:00", "1974-04-03_19:00:00"))


@pytest.fixture(scope="module")
def wrfout_hourly_d03(tmp_path_factory):
    """The SAME init and the same whole-hour frames, one nest down.

    Sub-kilometre spacing (333 m), so its resolution token exercises the
    integer-metre branch while its lead/valid times are identical to
    ``wrfout_hourly``'s -- everything the old filename carried.
    """

    return _write_wrfout(
        tmp_path_factory.mktemp("render-rust-hourly-d03") /
        "wrfout_d03_1974-04-03_18-00-00.nc",
        ("1974-04-03_18:00:00", "1974-04-03_19:00:00"),
        grid_id=3, dx=333.3333)


def _png_size(path) -> tuple[int, int]:
    """(width, height) from the PNG IHDR chunk; no image library."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    assert data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


@needs_renderer
def test_rust_engine_renders_every_frame_including_half_hourly(
        wrfout, tmp_path, capsys):
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--out", str(out)])
    assert rc == 0
    assert "render: engine rust" in capsys.readouterr().out
    produced = sorted(p.name for p in out.glob("*.png"))
    assert produced, "the rust engine wrote no PNGs"
    # Both frames rendered -- the :30 frame is the sub-hourly lead the
    # source renderer refused (exact-time ordinal axis).
    for lead in _LEADS:
        matching = [name for name in produced if lead in name]
        assert matching, (lead, produced)
    # 'all' renders the catalog the fixture's fields prove out, which
    # must include the reflectivity composite at minimum.
    assert any("composite_reflectivity" in name for name in produced)
    for png in out.glob("*.png"):
        assert png.stat().st_size > 5_000, png.name
        width, height = _png_size(png)
        assert (width, height) == (1200, 900), png.name


@needs_renderer
def test_rust_engine_timeidx_selects_the_half_hourly_frame(
        wrfout, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "refl", "--timeidx", "1",
                   "--out", str(out)])
    assert rc == 0
    produced = [p.name for p in out.glob("*.png")]
    assert len(produced) == 1, produced
    assert "composite_reflectivity" in produced[0]
    assert _LEADS[1] in produced[0], produced
    assert _LEADS[0] not in produced[0]


@needs_renderer
def test_rust_engine_maps_shared_product_names(wrfout, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "refl,t2", "--timeidx", "0",
                   "--size", "800x600", "--out", str(out)])
    assert rc == 0
    produced = sorted(p.name for p in out.glob("*.png"))
    assert len(produced) == 2, produced
    assert any("composite_reflectivity" in name for name in produced)
    assert any("2m_temperature" in name for name in produced)
    for png in out.glob("*.png"):
        assert _png_size(png) == (800, 600)


# ---------------------------------------------------------------------------
# Domain + resolution tokens: two domains at one lead must not collide
# ---------------------------------------------------------------------------

@needs_renderer
def test_two_domains_at_one_lead_do_not_overwrite_each_other(
        wrfout_hourly, wrfout_hourly_d03, tmp_path):
    """The data-loss bug this token exists to prevent.

    Two nests of one run share model, init cycle, and forecast hour.  On
    the whole-hour axis nothing else distinguished them, so the second
    domain's PNG landed on the first domain's path and one of the two
    forecasts vanished with no error, no warning, and an exit code of 0.
    """

    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout_hourly), str(wrfout_hourly_d03),
                   "--engine", "rust", "--products", "refl",
                   "--timeidx", "0", "--out", str(out)])
    assert rc == 0
    produced = sorted(p.name for p in out.glob("*.png"))
    assert len(produced) == 2, (
        "two domains at one lead collapsed to one file", produced)
    assert any("_d02-1km_" in name for name in produced), produced
    assert any("_d03-333m_" in name for name in produced), produced
    # Both survive as real images, not one truncated overwrite.
    for png in out.glob("*.png"):
        assert png.stat().st_size > 5_000, png.name


@needs_renderer
def test_domain_and_resolution_token_on_the_exact_time_axis(
        wrfout, tmp_path):
    """The sub-hourly axis carries the same token as the whole-hour one."""

    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "refl", "--out", str(out)])
    assert rc == 0
    produced = sorted(p.name for p in out.glob("*.png"))
    assert len(produced) == 2, produced
    for name in produced:
        assert "_d02-1km_" in name, name
        assert "native_grid" not in name, name


@needs_renderer
def test_unknown_domain_identity_keeps_the_native_grid_token(
        tmp_path):
    """No GRID_ID, no wrfout_dNN name: the token degrades, never guesses.

    A file whose domain identity cannot be established keeps the
    renderer's generic ``native_grid`` slug rather than being labelled
    ``d01`` on no evidence.
    """

    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(grid, datetime.datetime(1974, 4, 3, 18),
                             dt=6.0)
    assert "GRID_ID" not in attrs
    path = tmp_path / "some_model_output.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0,
                      dy=1000.0, global_attrs=attrs) as writer:
        writer.write_frame(_STAMPS[0], _frame(seed=21))
    out = tmp_path / "png"
    rc = cli.main(["render", str(path), "--engine", "rust",
                   "--products", "refl", "--out", str(out)])
    assert rc == 0
    produced = [p.name for p in out.glob("*.png")]
    assert produced, "the anonymous-domain file rendered nothing"
    for name in produced:
        assert "native_grid" in name, name


def test_source_label_reaches_the_renderer_invocation(monkeypatch,
                                                      tmp_path):
    """A locally imported run is not a GDEX fetch, and a stock-WRF file
    is not ours: both are one flag away from being labelled honestly."""

    import subprocess

    from gpuwm import render as render_module

    seen: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def spy(command, **kwargs):
        seen.append([str(part) for part in command])
        return Result()

    monkeypatch.setattr(subprocess, "run", spy)
    monkeypatch.setattr("gpuwm.rustwx.find_renderer",
                        lambda: tmp_path / "rw_wrfbatch.exe")
    monkeypatch.setattr("gpuwm.rustwx.probe_renderer",
                        lambda path: (True, "stubbed"))

    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="composite_reflectivity",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600))
    assert seen, "no renderer invocation was made"
    command = seen[-1]
    assert "--source-label" in command
    assert command[command.index("--source-label") + 1] == "ArWen"

    seen.clear()
    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="composite_reflectivity",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600),
        source_label="WRF-ARW 4.6.1")
    command = seen[-1]
    assert command[command.index("--source-label") + 1] == "WRF-ARW 4.6.1"


@needs_renderer
def test_rust_engine_unknown_slug_fails_loudly(wrfout, tmp_path):
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "definitely_not_a_product",
                   "--out", str(tmp_path / "png")])
    assert rc == 1
    assert not list((tmp_path / "png").glob("*.png"))


def test_engine_rust_unbuilt_is_a_documented_refusal(
        wrfout, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("gpuwm.rustwx.find_renderer", lambda: None)
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--out", str(tmp_path / "png")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not built" in err
    assert "cargo build --release --locked --offline" in err


def test_engine_auto_falls_back_to_matplotlib(
        wrfout, tmp_path, monkeypatch, capsys):
    pytest.importorskip(
        "wrf", reason="matplotlib fallback needs the render extra")
    monkeypatch.setattr("gpuwm.rustwx.find_renderer", lambda: None)
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--products", "t2",
                   "--out", str(out)])
    assert rc == 0
    assert "render: engine matplotlib" in capsys.readouterr().out
    assert list(out.glob("t2_*.png"))


def test_parse_products_rust_mapping():
    assert parse_products_rust("all") == "all"
    assert parse_products_rust("refl,t2,wind10,precip") == (
        "composite_reflectivity,2m_temperature,mslp_10m_winds,total_qpf")
    # Raw catalog slugs pass through for the renderer's own validation.
    assert parse_products_rust("sbcape,refl") == (
        "sbcape,composite_reflectivity")
    assert parse_products_rust("refl,refl") == "composite_reflectivity"
    with pytest.raises(ValueError, match="no products"):
        parse_products_rust(",")


def test_parse_size():
    assert parse_size("1200x900") == (1200, 900)
    assert parse_size("800X600") == (800, 600)
    for bad in ("1200", "axb", "100x100", "1200x"):
        with pytest.raises(ValueError):
            parse_size(bad)


# ---------------------------------------------------------------------------
# --list-products: the catalog with per-file availability
# ---------------------------------------------------------------------------

@needs_renderer
def test_list_products_reports_the_full_catalog(wrfout, tmp_path, capsys):
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--list-products", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # The complete catalog is enumerated, not just what renders.
    assert "total=151" in out
    assert "renderable" in out and "excluded" in out
    # The fixture's fields prove out the reflectivity composite ...
    assert any("composite_reflectivity" in line and "renderable" in line
               for line in out.splitlines())
    # ... and its exact-time (half-hourly) axis excludes fixed-hour
    # windowed accumulations with the reason spelled out.
    assert any("qpf_total" in line and "exact-time" in line
               for line in out.splitlines())
    # Every non-renderable row carries a reason string.
    for line in out.splitlines():
        if line.lstrip().startswith("missing-fields"):
            assert "not stored:" in line, line


@needs_renderer
def test_list_products_whole_hour_axis_serves_windowed(
        wrfout_hourly, tmp_path, capsys):
    rc = cli.main(["render", str(wrfout_hourly), "--engine", "rust",
                   "--list-products", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # Run-total QPF folds the wrfout lane's apcp plane; the 1 h wind max
    # realizes through the documented hypot(u_10m, v_10m) fallback.
    assert any("qpf_total" in line and "renderable" in line
               for line in out.splitlines()), out
    assert any("10m_wind_1h_max" in line and "renderable" in line
               for line in out.splitlines()), out
    # 24/48 h windows stay blocked with their window arithmetic.
    assert any("2m_temp_0_24h_max" in line and "blocked" in line
               for line in out.splitlines()), out


@needs_renderer
def test_list_products_rejects_identity_gated_rows(
        wrfout, wrfout_hourly, tmp_path, capsys):
    """Architectural guard: availability rows must justify themselves
    with FIELDS, never with the source model's identity.  A 'gated'
    status, a 'no fetch plan' reason, or any 'for model ...' wording is
    a regression -- a product that cannot render always names the
    fields it is missing."""
    allowed = {"renderable", "missing-fields", "blocked", "excluded"}
    for path in (wrfout, wrfout_hourly):
        rc = cli.main(["render", str(path), "--engine", "rust",
                       "--list-products", "--out", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        rows = [line for line in out.splitlines()
                if line.startswith("  ") and line.split()]
        assert rows, out
        for line in rows:
            status = line.split()[0]
            assert status in allowed, line
            low = line.lower()
            assert "gated" not in low, line
            assert "fetch plan" not in low, line
            assert "for model" not in low, line
        assert "gated=" not in out
        # The families the old model-identity gate hid (smoke, simulated
        # IR, categorical precip) now carry honest field reasons.
        for slug in ("smoke_column", "simulated_ir_satellite",
                     "precipitation_type"):
            assert any(slug in line and "missing-fields" in line
                       and "not stored:" in line for line in rows), \
                (slug, out)


@needs_renderer
def test_windowed_products_render_on_whole_hour_stores(
        wrfout_hourly, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout_hourly), "--engine", "rust",
                   "--products", "qpf_total", "--out", str(out)])
    assert rc == 0
    produced = [p.name for p in out.glob("*.png")]
    assert any("qpf_total" in name for name in produced), produced


def test_list_products_matplotlib_engine(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("gpuwm.rustwx.find_renderer", lambda: None)
    path = tmp_path / "wrfout_d01_1974-04-03_18-00-00.nc"
    fields = _frame(seed=5)
    del fields["REFL_10CM"]
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0) \
            as writer:
        writer.write_frame(_STAMPS[0], fields)
    rc = cli.main(["render", str(path), "--list-products",
                   "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "engine matplotlib" in out
    # The identity-gate guard holds on this engine too: reasons are
    # variable presence, never a model identity.
    assert "gated" not in out
    lines = out.splitlines()
    assert any("refl" in line and "missing-fields" in line
               and "REFL_10CM" in line for line in lines), out
    for product in ("t2", "wind10", "precip"):
        assert any(f" {product} " in f" {line} " and "renderable" in line
                   for line in lines), (product, out)


# ---------------------------------------------------------------------------
# --pair: compose two runs' rendered PNGs into comparison sheets
# ---------------------------------------------------------------------------

def _tiny_png(path, *, width=32, height=24) -> None:
    PIL = pytest.importorskip(
        "PIL", reason="--pair needs Pillow (render extra)")
    from PIL import Image

    Image.new("RGB", (width, height), "#336699").save(path)


def _pair_dirs(tmp_path):
    left = tmp_path / "run-a"
    right = tmp_path / "run-b"
    left.mkdir()
    right.mkdir()
    # Rust-engine naming: domain token + slug after the _fNNN_ lead
    # marker is the pair key, so differing run prefixes still pair.
    _tiny_png(left / "rustwx_wrf_19740403_12z_f003_d02-3km_sbcape.png")
    _tiny_png(right / "rustwx_wrf_19740403_15z_f006_d02-3km_sbcape.png")
    # matplotlib naming pairs by identical stems.
    _tiny_png(left / "t2_d02-3km_1974-04-03_18-00-00.png")
    _tiny_png(right / "t2_d02-3km_1974-04-03_18-00-00.png")
    # Unmatched product must not produce a sheet.
    _tiny_png(left / "rustwx_wrf_19740403_12z_f003_d02-3km_mlcin.png")
    return left, right


def test_pair_composes_common_products(tmp_path, capsys):
    left, right = _pair_dirs(tmp_path)
    out = tmp_path / "pairs"
    rc = cli.main(["render", "--pair", str(left), str(right),
                   "--out", str(out)])
    assert rc == 0
    sheets = sorted(p.name for p in out.glob("*.png"))
    assert sheets == ["d02-3km_sbcape-pair.png",
                      "t2_d02-3km_1974-04-03_18-00-00-pair.png"]
    manifest = (out / "manifest.tsv").read_text(encoding="utf-8")
    assert manifest.startswith("product\tleft\tright\tpair\n")
    assert "sbcape" in manifest
    for sheet in out.glob("*.png"):
        assert sheet.stat().st_size > 500
        width, height = _png_size(sheet)
        assert width > 1_000 and height > 100, sheet.name


def test_pair_keys_keep_the_domain_so_nests_do_not_cross_pair(tmp_path):
    """Two nests of one run in one directory must not pair with each
    other -- a 3 km panel beside a 333 m panel compares nothing."""

    from gpuwm.pair_compose import product_name

    assert product_name(
        Path("rustwx_wrf_19740403_12z_f003_d02-3km_sbcape")
    ) == "d02-3km_sbcape"
    assert product_name(
        Path("rustwx_wrf_19740403_12z_f003_d03-333m_sbcape")
    ) == "d03-333m_sbcape"
    # The pre-token naming still keys the same way it always did.
    assert product_name(
        Path("rustwx_wrf_19740403_12z_f003_native_grid_sbcape")
    ) == "native_grid_sbcape"
    # The exact-time suffix rides along, as before.
    assert product_name(
        Path("rustwx_wrf_19740403_12z_f000_d02-1km_sbcape_valid_"
             "19740403_183000z_lead_000h30m00s")
    ) == "d02-1km_sbcape_valid_19740403_183000z_lead_000h30m00s"
    # A matplotlib name has no lead marker; the stem is the key.
    assert product_name(
        Path("t2_d02-1km_1974-04-03_18-00-00")
    ) == "t2_d02-1km_1974-04-03_18-00-00"


def test_pair_with_no_common_products_is_exit_2(tmp_path, capsys):
    left = tmp_path / "a"
    right = tmp_path / "b"
    left.mkdir()
    right.mkdir()
    _tiny_png(left / "rustwx_wrf_19740403_12z_f000_d02-1km_sbcape.png")
    _tiny_png(right / "rustwx_wrf_19740403_12z_f000_d02-1km_mucape.png")
    rc = cli.main(["render", "--pair", str(left), str(right),
                   "--out", str(tmp_path / "pairs")])
    assert rc == 2
    assert "no matching product PNGs" in capsys.readouterr().err


def test_pair_rejects_wrfout_arguments(tmp_path, capsys):
    rc = cli.main(["render", "some_wrfout.nc", "--pair",
                   str(tmp_path), str(tmp_path),
                   "--out", str(tmp_path / "pairs")])
    assert rc == 2
    assert "do not combine" in capsys.readouterr().err


def test_render_without_wrfouts_or_pair_is_exit_2(capsys):
    rc = cli.main(["render"])
    assert rc == 2
    assert "WRFOUT" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# doctor integration
# ---------------------------------------------------------------------------

def test_doctor_reports_the_rust_renderer():
    from gpuwm.doctor import _rust_renderer_check

    check = _rust_renderer_check()
    assert check.name.startswith("renderer rw_wrfbatch")
    if RENDERER is not None:
        assert check.status == "verified"
        assert str(RENDERER) in check.detail
        assert "basemap" in check.detail
    else:
        # Not built: an info line with the build one-liner, never a gap
        # (matplotlib remains the documented fallback).
        assert check.status in ("info", "missing")
        assert check.remedy and "cargo build" in check.remedy


def test_doctor_env_override_naming_missing_file_is_hard(monkeypatch):
    from gpuwm.doctor import _rust_renderer_check

    monkeypatch.setenv(rustwx.RENDERER_ENV, r"C:\nonexistent\rw.exe")
    check = _rust_renderer_check()
    assert check.status == "missing"
    assert "names a missing file" in check.detail
