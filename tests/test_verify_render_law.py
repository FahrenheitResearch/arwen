"""`gpuwm verify`'s weather fields come from the renderer, or not at all.

CLAUDE.md, Drew 2026-08-06: "Weather-field product plots come from the
real Rust renderer ``rw_wrfbatch`` driven through ``gpuwm.rustwx`` ...
matplotlib, ``wrf.plot`` and cartopy are not allowed for weather
fields".

The concrete breakage these tests prevent (hidden-scope audit F6,
2026-08-18): ``gpuwm verify real74_d01`` drew mean-sea-level pressure,
2 m temperature, 500 hPa height with a wind quiver and 6 h accumulated
precipitation with ``matplotlib.pcolormesh``/``contour``/``quiver``, on
the BARE DEFAULT of the door -- omitting ``--outdir`` sent the PNGs to
a temporary directory rather than skipping them, so there was no way to
run the case without the unlawful drawing happening.  Four of the exact
product classes the law names, with no ``rustwx`` probe, no renderer
and no fallback announcement anywhere under ``gpuwm/verify/``.

The lawful shape is the renderer's own panels, filed under the render
layout, and -- where the renderer cannot be reached -- receipts with NO
weather-field imagery and a line saying so.  Never matplotlib.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from gpuwm.io.wrfout import WrfoutWriter
from gpuwm.verify import metrics

_NZ, _NY, _NX = 8, 12, 16

_ATTRS = {
    "MAP_PROJ": 1, "TRUELAT1": 30.0, "TRUELAT2": 60.0,
    "STAND_LON": -97.0, "MOAD_CEN_LAT": 32.0,
    "CEN_LAT": 32.0, "CEN_LON": -97.0,
    "POLE_LAT": 90.0, "POLE_LON": 0.0,
    "GRID_ID": 1, "PARENT_ID": 1, "DX": 12000.0, "DY": 12000.0,
    "START_DATE": "1974-04-04_00:00:00",
    "SIMULATION_START_DATE": "1974-04-04_00:00:00",
    "TITLE": " OUTPUT FROM GPUWM",
    "I_PARENT_START": 1, "J_PARENT_START": 1, "PARENT_GRID_RATIO": 1,
    "DT": 60.0,
}


def _frame(rain: float) -> dict:
    pressure = np.linspace(92500.0, 30000.0, _NZ, dtype=np.float32)
    height = np.linspace(0.0, 10000.0, _NZ + 1, dtype=np.float32)
    lat = np.broadcast_to(
        np.linspace(30.0, 34.0, _NY, dtype=np.float32)[:, None],
        (_NY, _NX)).copy()
    lon = np.broadcast_to(
        np.linspace(-100.0, -94.0, _NX, dtype=np.float32)[None, :],
        (_NY, _NX)).copy()
    return {
        "U": np.full((_NZ, _NY, _NX + 1), 8.0, np.float32),
        "V": np.full((_NZ, _NY + 1, _NX), 3.0, np.float32),
        "W": np.zeros((_NZ + 1, _NY, _NX), np.float32),
        "T": np.zeros((_NZ, _NY, _NX), np.float32),
        "P": np.zeros((_NZ, _NY, _NX), np.float32),
        "PB": np.broadcast_to(pressure[:, None, None],
                              (_NZ, _NY, _NX)).copy(),
        "PH": np.zeros((_NZ + 1, _NY, _NX), np.float32),
        "PHB": np.broadcast_to((9.81 * height)[:, None, None],
                               (_NZ + 1, _NY, _NX)).copy(),
        "QVAPOR": np.full((_NZ, _NY, _NX), 0.004, np.float32),
        "QCLOUD": np.zeros((_NZ, _NY, _NX), np.float32),
        "QRAIN": np.zeros((_NZ, _NY, _NX), np.float32),
        "MU": np.zeros((_NY, _NX), np.float32),
        "MUB": np.full((_NY, _NX), 82500.0, np.float32),
        "HGT": np.zeros((_NY, _NX), np.float32),
        "PSFC": np.full((_NY, _NX), 100000.0, np.float32),
        "TSK": np.full((_NY, _NX), 291.0, np.float32),
        "T2": np.full((_NY, _NX), 290.0, np.float32),
        "TH2": np.full((_NY, _NX), 290.0, np.float32),
        "Q2": np.full((_NY, _NX), 0.004, np.float32),
        "U10": np.full((_NY, _NX), 7.0, np.float32),
        "V10": np.full((_NY, _NX), 2.0, np.float32),
        "RAINNC": np.full((_NY, _NX), rain, np.float32),
        "RAINC": np.full((_NY, _NX), rain / 4.0, np.float32),
        "XLAT": lat,
        "XLONG": lon,
        "MAPFAC_M": np.ones((_NY, _NX), np.float32),
        "MAPFAC_U": np.ones((_NY, _NX + 1), np.float32),
        "MAPFAC_V": np.ones((_NY + 1, _NX), np.float32),
        "F": np.full((_NY, _NX), 8.0e-5, np.float32),
        "SINALPHA": np.zeros((_NY, _NX), np.float32),
        "COSALPHA": np.ones((_NY, _NX), np.float32),
        "LU_INDEX": np.full((_NY, _NX), 10.0, np.float32),
    }


@pytest.fixture(scope="module")
def series(tmp_path_factory) -> tuple[Path, Path]:
    """(accumulation start, final) -- the +6 h and +12 h verify frames."""

    root = tmp_path_factory.mktemp("verify-render-law")
    written = []
    for hour, rain in ((6, 1.0), (12, 4.5)):
        path = root / f"wrfout_d01_1974-04-04_{hour:02d}_00_00.nc"
        with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=12000.0,
                          dy=12000.0, global_attrs=_ATTRS) as writer:
            writer.write_frame(f"1974-04-04_{hour:02d}:00:00", _frame(rain))
        written.append(path)
    return written[0], written[1]


def _renderer_gate() -> tuple[bool, str]:
    from gpuwm import rustwx
    from gpuwm.render import renderer_refusal

    renderer = rustwx.find_renderer()
    if renderer is None:
        return False, "rust renderer not built"
    reason = renderer_refusal(renderer)
    return (reason is None), (reason or "")


_USABLE, _WHY = _renderer_gate()
needs_renderer = pytest.mark.skipif(
    not _USABLE, reason=f"this tree's renderer is not usable: {_WHY}")


# ---------------------------------------------------------------------------
# The source gate: no weather field in this module may be drawn locally.
# ---------------------------------------------------------------------------

#: Drawing CALLS that put a weather field on a canvas -- the trailing
#: paren is what keeps this off the prose above, which has to be free to
#: name the defect it replaced.  Named calls rather than "does the file
#: import matplotlib" because the drawing is what the law forbids.
_LOCAL_DRAWING = re.compile(
    r"\.(pcolormesh|contourf|contour|quiver|barbs|imshow|savefig)\(")


def test_the_verify_metrics_module_draws_no_weather_field_locally():
    """F6's durable gate, on the file the audit named.

    Breakage prevented: a synoptic panel re-acquiring a local
    ``pcolormesh``.  The four subjects here -- MSLP, 2 m temperature,
    500 hPa height/wind, 6 h QPF -- are exactly the products the render
    law reserves for ``rw_wrfbatch``.
    """

    source = Path(metrics.__file__).read_text(encoding="utf-8")
    hits = sorted(set(_LOCAL_DRAWING.findall(source)))
    assert not hits, (
        f"gpuwm/verify/metrics.py draws with {hits}; weather-field "
        "product plots come from rw_wrfbatch through gpuwm.rustwx")
    assert "import matplotlib" not in source, (
        "gpuwm/verify/metrics.py still imports matplotlib")


@needs_renderer
def test_the_synoptic_panels_are_drawn_by_the_rust_renderer(series, tmp_path):
    """Against the artifact: the real binary, on real wrfout bytes."""

    start, final = series
    spec = metrics.SynopticMapSpec(
        products=("mslp_10m_winds", "2m_temperature", "500mb_height_winds",
                  "qpf_6h"))
    paths = metrics.make_synoptic_maps(final, start, tmp_path / "maps", spec)
    assert paths, "the verify door produced no weather-field imagery"
    # The product slug is read from the PRODUCT FOLDER, not the filename:
    # since 7fdca0d74 (2026-08-23) the delivered name drops the domain and
    # product tokens its own folders already spell, because the repeated
    # pair put a measured delivery past the Windows MAX_PATH ceiling.
    products_drawn = [path.parent.parent.name for path in paths]
    for slug in spec.products:
        assert slug in products_drawn, (
            f"no panel for {slug}; drew {products_drawn}")
    # `qpf_6h` is a WINDOWED product: it exists only because both frames
    # reached one store, which is the half a per-file render cannot do.
    assert "qpf_6h" in products_drawn
    for path in paths:
        assert path.is_file() and path.stat().st_size > 1000
        # The renderer's own filename grammar, filed under the render
        # folder layout (domain -> product -> valid day).
        assert path.name.startswith("arwen_"), path.name
        assert path.parent.parent.parent.parent == tmp_path / "maps", (
            f"{path} is not filed under <maps>/<domain>/<product>/<day>/")


@needs_renderer
def test_the_reflectivity_panel_is_drawn_by_the_rust_renderer(series,
                                                              tmp_path):
    _start, final = series
    path = metrics.make_composite_reflectivity_map(
        final, tmp_path / "refl", metrics.ReflectivityMapSpec())
    assert path.is_file() and path.stat().st_size > 1000
    # The product folder spells the product; the delivered filename does
    # not repeat it since 7fdca0d74 (2026-08-23, the MAX_PATH shortening).
    assert path.parent.parent.name == "composite_reflectivity", str(path)


def test_no_renderer_means_no_imagery_and_a_line_that_says_so(
        series, tmp_path, monkeypatch, capsys):
    """The lawful degradation: receipts without weather fields, announced."""

    from gpuwm import rustwx

    monkeypatch.setattr(rustwx, "find_renderer", lambda: None)
    start, final = series
    spec = metrics.SynopticMapSpec(products=("mslp_10m_winds",))
    out = tmp_path / "maps"
    paths = metrics.make_synoptic_maps(final, start, out, spec)
    captured = capsys.readouterr()
    assert paths == ()
    assert not list(out.rglob("*.png")) if out.exists() else True
    text = captured.err + captured.out
    assert "rw_wrfbatch" in text, "the note must name the missing artifact"
    assert "fetch-bridges" in text, "the note must name the remedy"


def test_no_renderer_refuses_a_reflectivity_panel_by_name(
        series, tmp_path, monkeypatch):
    """A caller who asked for ONE panel gets a refusal, never silence."""

    from gpuwm import rustwx

    monkeypatch.setattr(rustwx, "find_renderer", lambda: None)
    _start, final = series
    with pytest.raises(RuntimeError) as excinfo:
        metrics.make_composite_reflectivity_map(
            final, tmp_path / "refl", metrics.ReflectivityMapSpec())
    assert "rw_wrfbatch" in str(excinfo.value)
    assert "fetch-bridges" in str(excinfo.value)


def test_the_frozen_case_declares_renderer_catalog_slugs():
    """The case's map policy is now a product list, not a plot recipe."""

    pytest.importorskip("cupy")
    from gpuwm.verify.cases import real74_d01

    spec = real74_d01._SYNOPTIC_MAP_SPEC
    assert set(spec.products) == {
        "mslp_10m_winds", "2m_temperature", "500mb_height_winds", "qpf_6h"}
