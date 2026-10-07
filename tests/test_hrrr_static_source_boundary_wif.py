"""A static-source HRRR-cone root builds its boundary aerosol from its own coordinates.

THE BREAKAGE (site canary on public 2.8.6).  A plain HRRR regional run
authored by ``gpuwm domain --source hrrr`` on the HRRR cone names
``[static] source = "hrrr-conus-v4"`` and the mp=28 profile with
``(aer_init_opt, wif_input_opt) = (1, 1)``.  It passed prepare and then
failed at forecast start with "(aer_init_opt, wif_input_opt) = (1, 1)
cannot be honoured: ... could not derive the model mass-point
latitudes/longitudes": the HRRR route's boundary workers ran
:func:`gpuwm.ingest.real.initialize_real` on each strip with no
coordinates, and the static-source field set has no ``XLAT_M``/``XLONG_M``
for a strip to borrow.  The site's whole 2.8.6 rollout stopped on it.

These cells build such a root from a small synthetic geo_em registered
under the ``hrrr-conus-v4`` id (HRRR's own Lambert constants, 3 km), cut
its boundary statics through the same helper the HRRR controller calls,
and run the strip loop with the REAL ``initialize_real`` and the real WIF
step (the climatology is a tiny synthetic IFV=5 dataset, decoded and
interpolated by the module's NumPy oracle so no bridge is needed).  What is
proved is that each strip's own rectangle of the projection grid's
mass-point coordinates reaches the WIF step and fills the strip's boundary
aerosol; the control proves the refusal still stands for a carrier that
truly has no coordinates.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import tools.hrrr_single_domain_benchmark as hrrr_runner
from gpuwm.config import RunConfig
from gpuwm.static import external_source as es
from test_real_init import _analyzed_hrrr_real_init, _ReferencePreprocessBackend
from test_static_sources import (
    LEGEND, SRC, _lambert, _source_values, _write_geo_em)
from test_wif_climatology import _MONTHS, _write_ifv5

SOURCE_ID = "hrrr-conus-v4"
WIDTH = 5
#: The root: a 20 x 10 mass-point sub-window of the synthetic source, so
#: the crop is exact (``_lambert(21, 11)`` crops at (10, 10)).
E_WE, E_SN = 21, 11
#: Bottom-up WIF pressure levels (Pa), deep enough to bracket every dry
#: eta pressure a p_top = 100 hPa column produces.
WIF_PRESSURE = (103000.0, 70000.0, 30000.0, 5000.0)


@pytest.fixture
def hrrr_cone_source(tmp_path, monkeypatch):
    """A synthetic geo_em served as the ``hrrr-conus-v4`` row, or skip."""

    pytest.importorskip("gpuwm.io.nc_writer_bridge")
    from gpuwm.io import nc_writer_bridge
    if nc_writer_bridge.unavailable_reason() is not None:
        pytest.skip(nc_writer_bridge.unavailable_reason())
    from gpuwm.netcdf_bridge import find_netcdf_bin
    if find_netcdf_bin() is None:
        pytest.skip("rw_netcdf is not built")
    values = _source_values()
    folder = tmp_path / "geog" / "static_sources" / SOURCE_ID
    folder.mkdir(parents=True)
    path = folder / "hrrr_geo_em.d01.nc"
    _write_geo_em(path, values)
    data = path.read_bytes()
    grid_rows = "\n".join(f"{k} = {json.dumps(v)}" for k, v in SRC.items())
    legend_rows = "\n".join(f"{k} = {json.dumps(v)}" for k, v in LEGEND.items())
    table = tmp_path / "table.toml"
    table.write_text(f"""schema = "gpuwm-static-sources-v1"
[[source]]
id = "{SOURCE_ID}"
describes = "synthetic stand-in on the HRRR CONUS projection"
format = "wps-geo_em"
filename = "hrrr_geo_em.d01.nc"
url = "https://example.invalid/hrrr_geo_em.d01.nc"
mirrors = []
sha256 = "{hashlib.sha256(data).hexdigest()}"
bytes = {len(data)}
fields = {json.dumps(list(values))}
[source.grid]
{grid_rows}
[source.attrs]
{legend_rows}
[source.rename]
VAR = "VARLS"
""", encoding="utf-8")
    monkeypatch.setattr(es, "TABLE_PATH", table)
    es._load_table.cache_clear()
    yield SimpleNamespace(geog=tmp_path / "geog")
    es._load_table.cache_clear()


def _root_static(source, monkeypatch, tmp_path):
    """The root's statics as the HRRR native static build assembles them.

    ``[static] source = "hrrr-conus-v4"`` parsed the way a configuration
    carries it, built through ``build_static`` (a complete source: the
    WPS_GEOG build must not run), plus the projection's map fields exactly
    as tools/hrrr_build_native_static.py adds them.
    """

    from gpuwm.static import build
    from gpuwm.static.highres_production import parse_static_table

    carrier = parse_static_table({"source": SOURCE_ID}, source="experiment",
                                 base_dir=tmp_path)
    assert carrier.static_source.id == SOURCE_ID
    selection = replace(build.GeogSelection.fallback(source.geog),
                        static_source=carrier.static_source)

    def refuse_baseline(*args, **kwargs):
        raise AssertionError("a complete static source must not build WPS fields")

    monkeypatch.setattr(build, "_build_static_routed", refuse_baseline)
    monkeypatch.setattr(build.GeogSelection, "landuse_global_attrs",
                        lambda self: LEGEND)
    grid = _lambert(E_WE, E_SN)
    report = {}
    fields = build.build_static(grid, source.geog, selection=selection,
                                source_coverage_report=report)
    assert report["static_source"]["status"] == "APPLIED"
    fields.update({"MAPFAC_M": grid.mapfac_m(), "MAPFAC_U": grid.mapfac_u(),
                   "MAPFAC_V": grid.mapfac_v()})
    fields["F"], fields["E"] = grid.coriolis_m()
    fields["SINALPHA"], fields["COSALPHA"] = grid.rotation_m()
    return grid, fields


def _wif_dataset(tmp_path):
    """A full 12-month IFV=5 WIF dataset on a tiny global lat-lon grid."""

    ny, nx = 5, 8
    entries = []
    for month_index, month in enumerate(_MONTHS):
        for level, pressure in enumerate(WIF_PRESSURE):
            entries.append((f"QNWFA_{month}", float(level + 1), np.full(
                (ny, nx), 1.0e9 * (1.0 + 0.1 * month_index) / (level + 1),
                dtype=np.float32)))
            entries.append((f"QNIFA_{month}", float(level + 1), np.full(
                (ny, nx), 1.0e5 / (level + 1), dtype=np.float32)))
            entries.append((f"P_WIF_{month}", float(level + 1), np.full(
                (ny, nx), pressure, dtype=np.float32)))
    path = tmp_path / "QNWFA_QNIFA_SIGMA_MONTHLY.dat"
    _write_ifv5(path, entries)
    return path


def _run_cfg(mp_physics):
    return RunConfig(
        nx=E_WE - 1, ny=E_SN - 1, nz=8, dx=3000.0, dy=3000.0, ztop=18000.0,
        dt=15.0, run_seconds=3600.0, hybrid_opt=2, etac=0.2, moist=True,
        mp_physics=mp_physics, spec_bdy_width=WIDTH,
        **({"aer_init_opt": 1, "wif_input_opt": 1} if mp_physics == 28
           else {}))


def _arm_strip_loop(monkeypatch, tmp_path):
    """Run the strip loop's ``initialize_real`` for real, and watch the WIF step.

    The strip loop hands each side's ``grid=`` carrier to
    ``initialize_real``; the delegate runs the real initialization on a
    decoded-native-HRRR column set of that strip's shape with exactly that
    carrier, so the coordinates the WIF step sees are the ones the strip
    loop shipped and nothing else.
    """

    import gpuwm.ingest.preprocess_backend as backends
    import gpuwm.ingest.real as real
    from gpuwm.ingest import wif_climatology

    monkeypatch.setenv(wif_climatology.WIF_CLIMATOLOGY_PATH_ENV,
                       str(_wif_dataset(tmp_path)))
    monkeypatch.setattr(backends, "resolve_preprocess_backend",
                        lambda *a, **k: _ReferencePreprocessBackend())
    load = wif_climatology.load_wif_climatology
    fields_for_grid = wif_climatology.wif_fields_for_grid
    seen = []

    def load_numpy(path, *, backend="rust"):
        return load(path, backend="numpy")

    def fields_numpy(climatology, lat, lon, date_str, pb, phb, *, backend="rust"):
        seen.append((np.asarray(lat, dtype=np.float64).copy(),
                     np.asarray(lon, dtype=np.float64).copy()))
        return fields_for_grid(climatology, lat, lon, date_str, pb, phb,
                               backend="numpy")

    monkeypatch.setattr(wif_climatology, "load_wif_climatology", load_numpy)
    monkeypatch.setattr(wif_climatology, "wif_fields_for_grid", fields_numpy)

    def initialize_strip(strip_met, strip_cfg, coord, terrain, **kwargs):
        assert (strip_cfg.aer_init_opt, strip_cfg.wif_input_opt) == (1, 1)
        init_kwargs = {"boundary_species": kwargs["boundary_species"]}
        if "grid" in kwargs:
            init_kwargs["grid"] = kwargs["grid"]
        result, _ = _analyzed_hrrr_real_init(
            28, shape=(strip_cfg.ny, strip_cfg.nx), init_kwargs=init_kwargs,
            aer_init_opt=1, wif_input_opt=1)
        return result

    monkeypatch.setattr(real, "initialize_real", initialize_strip)
    return seen


def _strip_loop(static_sides, cfg):
    mets = {side: SimpleNamespace(flag_sh_surface_fallback=False)
            for side in static_sides}
    return hrrr_runner._initialize_boundary_sides(
        mets, cfg, static_sides, list(np.linspace(1.0, 0.0, cfg.nz + 1)),
        p_top=10000.0, width=WIDTH, preprocess_backend="cpu")


def test_a_static_source_root_builds_boundary_aerosol_from_its_own_coordinates(
        hrrr_cone_source, monkeypatch, tmp_path):
    grid, static = _root_static(hrrr_cone_source, monkeypatch, tmp_path)
    # The static-source field set carries no coordinates: the strip's
    # geodesy can only come from the projection grid.
    assert "XLAT_M" not in static and "XLONG_M" not in static
    cfg = _run_cfg(28)
    static_sides = hrrr_runner._boundary_static_sides(
        static, grid, cfg, width=WIDTH)
    seen = _arm_strip_loop(monkeypatch, tmp_path)

    sides, _, _, _ = _strip_loop(static_sides, cfg)

    lat, lon = (np.asarray(value, dtype=np.float64)
                for value in grid.latlon_mass())
    ny, nx = cfg.ny, cfg.nx
    rectangles = (("west", (0, ny, 0, WIDTH)), ("east", (0, ny, nx - WIDTH, nx)),
                  ("south", (0, WIDTH, 0, nx)), ("north", (ny - WIDTH, ny, 0, nx)))
    assert len(seen) == len(rectangles)
    for (side, (y0, y1, x0, x1)), (seen_lat, seen_lon) in zip(rectangles, seen):
        np.testing.assert_allclose(seen_lat, lat[y0:y1, x0:x1], rtol=0, atol=1e-5,
                                   err_msg=side)
        np.testing.assert_allclose(seen_lon, lon[y0:y1, x0:x1], rtol=0, atol=1e-5,
                                   err_msg=side)
        # The climatology filled the strip's boundary aerosol.
        for name in ("nwfa", "nifa"):
            assert name in sides[side], (side, sorted(sides[side]))
            value = np.asarray(sides[side][name])
            assert np.isfinite(value).all() and (value > 0.0).all(), (side, name)


def test_a_strip_with_no_coordinates_still_refuses_the_named_climatology(
        hrrr_cone_source, monkeypatch, tmp_path):
    """The control: the 2.8.6 payload (no geodesy) refuses, by name.

    The refusal is kept for a carrier that truly has no coordinates: an
    explicit climatology request cannot be honoured by guessing a grid.
    """

    grid, static = _root_static(hrrr_cone_source, monkeypatch, tmp_path)
    cfg = _run_cfg(28)
    bare = hrrr_runner._compact_boundary_static(static, cfg, width=WIDTH)
    assert all("XLAT_M" not in fields for fields in bare.values())
    seen = _arm_strip_loop(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=(
            r"\(aer_init_opt, wif_input_opt\) = \(1, 1\) cannot be honoured.*"
            r"could not derive the model mass-point latitudes/longitudes")):
        _strip_loop(bare, cfg)
    assert seen == []


def test_only_an_mp28_root_ships_strip_geodesy(hrrr_cone_source, monkeypatch,
                                               tmp_path):
    grid, static = _root_static(hrrr_cone_source, monkeypatch, tmp_path)
    plain = hrrr_runner._boundary_static_sides(
        static, grid, _run_cfg(8), width=WIDTH)
    assert all("XLAT_M" not in fields and "XLONG_M" not in fields
               for fields in plain.values())


def test_the_controller_cuts_its_strips_through_the_geodesy_helper():
    """The HRRR controller builds its strip statics only through the helper
    the cells above prove, so dropping the coordinates there cannot pass."""

    source = inspect.getsource(hrrr_runner.run)
    assert "static_sides = _boundary_static_sides(" in source
    assert "_compact_boundary_static(" not in source
