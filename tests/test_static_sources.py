"""Static fields taken from a published static file (gpuwm.static.external_source).

The packaged table parses and its rows are complete; a configuration's
``[static] source`` rides on the static carrier and through a sealed echo;
the crop is computed from grid identity and refuses a grid that is not a
sub-window; the overlay takes exactly the file's values (cut in Rust) for
the fields the build produced, renames the drag fields, recomputes TMN,
and refuses another land-use legend.  A configuration that names no
source never builds a selection that carries one.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.static import external_source as es


def test_packaged_table_rows_are_complete():
    rows = es.static_source_rows()
    row = rows["hrrr-conus-v4"]
    assert row.format == "wps-geo_em"
    assert row.bytes == 1112724420
    assert row.sha256.startswith("a785b53f")
    served = row.served()
    # The drag rename: the large-scale set takes the LS names, the small-
    # scale set keeps its own; nothing maps onto the WPS gwd_opt=1 names.
    assert served["VARLS"] == "VAR" and served["CONLS"] == "CON"
    assert served["OL4LS"] == "OL4" and served["VARSS"] == "VARSS"
    assert "VAR" not in served and "CON" not in served
    from gpuwm.static.orographic import GSL_LS_FIELDS, GSL_SS_FIELDS
    assert set(GSL_LS_FIELDS + GSL_SS_FIELDS) <= set(served)
    for name in ("LANDUSEF", "LU_INDEX", "LANDMASK", "SCT_DOM", "SCB_DOM",
                 "SOILCTOP", "SOILCBOT", "HGT_M", "SLOPECAT", "GREENFRAC", "LAI12M",
                 "LAKE_DEPTH"):
        assert name in served
    assert row.grid_map["e_we"] == 1800 and row.grid_map["e_sn"] == 1060


#: An NCO production-tree URL under a versioned package folder
#: (``.../nwprod/hrrr.v4.1.21/...``).  NCO deletes the folder when the
#: package version moves on.
_NCO_VERSION_FOLDER = re.compile(r"/nwprod/(?:[^/]+/)*[A-Za-z0-9_-]+\.v\d+(?:\.\d+)+/")


def test_every_row_has_a_source_outside_an_nco_version_folder():
    """The breakage: row hrrr-conus-v4 had only NCO's hrrr.v4.1.21 folder
    (the v4.1.19 one already returned 404), and since 2.8.6 every HRRR go
    route and native chain stages that row, so a rotated folder would make
    every HRRR run on a fresh install refuse.  Each row keeps at least one
    URL that a version bump at the publisher cannot delete."""
    assert _NCO_VERSION_FOLDER.search(
        "https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/hrrr.v4.1.21/fix/conus/hrrr_geo_em.d01.nc")
    for row in es.static_source_rows().values():
        durable = [url for url in (row.url, *row.mirrors) if not _NCO_VERSION_FOLDER.search(url)]
        assert durable, f"{row.id}: every source is an NCO version folder: {(row.url, *row.mirrors)}"
    row = es.static_source_row("hrrr-conus-v4")
    assert row.mirrors == (
        "https://github.com/FahrenheitResearch/arwen/releases/download/v2.8.7/"
        "hrrr-conus-v4-hrrr_geo_em.d01.nc",)


def test_fetch_takes_the_release_mirror_when_the_nco_folder_is_gone(tmp_path, monkeypatch):
    """The packaged row's own URLs, network stubbed: NCO answers 404 (the
    rotated folder), the fetch walks on to the release mirror, and the
    mirror's bytes are installed only after they verify against the pin."""
    from dataclasses import replace
    from io import BytesIO
    from urllib.error import HTTPError
    from gpuwm import fetch_guard
    data = b"stand-in for the pinned geo_em bytes\n" * 64
    packaged = es.static_source_row("hrrr-conus-v4")
    row = replace(packaged, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(es, "static_source_row", lambda source_id: row)
    monkeypatch.setenv(fetch_guard.LOCK_ROOT_ENV, str(tmp_path / "locks"))
    attempted = []

    class Response(BytesIO):
        status = 200
        headers = {}

    def open_url(request):
        attempted.append(request.full_url)
        if request.full_url == packaged.url:
            raise HTTPError(request.full_url, 404, "Not Found", {}, None)
        return Response(data)

    lines = []
    final = es.fetch_static_source("hrrr-conus-v4", tmp_path / "geog",
                                    progress=lines.append, urlopen_fn=open_url)
    assert attempted == [packaged.url, packaged.mirrors[0]]
    assert final == tmp_path / "geog" / "static_sources" / "hrrr-conus-v4" / "hrrr_geo_em.d01.nc"
    assert final.read_bytes() == data
    es.verify_local_file(final, row)

    # Other bytes from the mirror are refused, naming both URLs.
    final.unlink()
    attempted.clear()
    row = replace(row, sha256="0" * 64)
    from gpuwm import geog_assets
    with pytest.raises(geog_assets.GeogFetchError, match="could not be fetched from any of its 2 URL"):
        es.fetch_static_source("hrrr-conus-v4", tmp_path / "geog",
                               progress=lines.append, urlopen_fn=open_url)
    assert attempted == [packaged.url, packaged.mirrors[0]]
    assert not final.exists()


def test_setting_groups_and_refusals():
    every = es.parse_static_source({"source": "hrrr-conus-v4"}, source="t")
    assert every.groups == ("all",)
    soil = es.parse_static_source(
        {"source": "hrrr-conus-v4", "source_fields": ["soil"]}, source="t")
    assert set(soil.engine_names()) == set(es.FIELD_GROUPS["soil"])
    land = es.parse_static_source(
        {"source": "hrrr-conus-v4", "source_fields": ["landuse", "drag"]},
        source="t")
    assert "LANDMASK" in land.engine_names() and "VARSS" in land.engine_names()
    assert "SCT_DOM" not in land.engine_names()
    assert es.parse_static_source({}, source="t") is None
    for bad, words in (
            ({"source": "no-such-row"}, "no static source"),
            ({"source": "hrrr-conus-v4", "source_fields": ["soils"]}, "unknown group"),
            ({"source": "hrrr-conus-v4", "source_fields": ["all", "soil"]}, "not both"),
            ({"source_fields": ["soil"]}, "without a source")):
        with pytest.raises(es.StaticSourceError, match=words):
            es.parse_static_source(bad, source="t")
    with pytest.raises(es.StaticSourceError, match="table now pins"):
        es.StaticSourceSetting(id="hrrr-conus-v4", sha256="0" * 64)
    echo = every.echo()
    for groups in (["bad-group"], [], ["all", "soil"], [42]):
        with pytest.raises(es.StaticSourceError, match="groups"):
            es.setting_from_echo({**echo, "fields": groups}, source="mutated seal")


def test_static_table_carrier_echo_and_seal(tmp_path):
    from gpuwm.static.highres_production import (
        parse_sealed_static_highres, parse_static_table, resolve_static_highres,
        static_highres_identity)
    carrier = parse_static_table({"source": "hrrr-conus-v4"}, source="t",
                                 base_dir=tmp_path)
    assert carrier.enabled is False
    assert carrier.static_source.id == "hrrr-conus-v4"
    echo = static_highres_identity(carrier)
    assert echo["static_source"] == {
        "id": "hrrr-conus-v4", "sha256": es.static_source_row("hrrr-conus-v4").sha256,
        "fields": ["all"]}
    sealed = parse_sealed_static_highres(echo, source="t", base_dir=tmp_path)
    assert sealed.static_source == carrier.static_source
    # A whole configuration: the source rides; a 3 km root takes no
    # high-resolution default beside it.
    raw = {"static": {"source": "hrrr-conus-v4", "source_fields": ["soil"]},
           "domain": [{"dx": 3000.0}]}
    resolved = resolve_static_highres(raw, source="t", base_dir=tmp_path)
    assert resolved.static_source.groups == ("soil",)
    with pytest.raises(ValueError, match="does not have a key or table"):
        parse_static_table({"sauce": "x"}, source="t", base_dir=tmp_path)
    with pytest.raises(ValueError, match="declares nothing"):
        parse_static_table({}, source="t", base_dir=tmp_path)
    # Without a source nothing changes: no key in the echo.
    plain = parse_static_table({"highres": {"enabled": False,
                                            "cache_root": str(tmp_path)}},
                               source="t", base_dir=tmp_path)
    assert "static_source" not in static_highres_identity(plain)


def test_carrier_reaches_the_selection_only_when_named(tmp_path):
    from gpuwm.static.build import GeogSelection
    from gpuwm.static.highres_production import parse_static_table
    from gpuwm.static.terrain_smoothing import selection_carrier
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share\n max_dom = 1,\n/\n&geogrid\n geog_data_res = 'default',\n/\n")
    plain = GeogSelection.from_case_data(
        SimpleNamespace(wps_namelist=wps, geog_root=tmp_path), 1)
    assert plain.static_source is None
    assert plain == GeogSelection.fallback(tmp_path)
    carrier = parse_static_table({"source": "hrrr-conus-v4"}, source="t",
                                 base_dir=tmp_path)
    assert selection_carrier(carrier) is carrier
    named = GeogSelection.from_case_data(
        SimpleNamespace(wps_namelist=wps, geog_root=tmp_path,
                        static_highres=carrier), 1)
    assert named.static_source is carrier.static_source


def test_root_seam_requires_attestation(tmp_path):
    from gpuwm.static.highres_production import parse_static_table
    carrier = parse_static_table({"source": "hrrr-conus-v4"}, source="t",
                                 base_dir=tmp_path)
    with pytest.raises(es.StaticSourceError, match="static-source root seam"):
        es.require_root_static_source(carrier, 1, None)
    es.require_root_static_source(
        carrier, 1, {"baseline": {"static_source": es.static_source_receipt(carrier)}})
    es.require_root_static_source(None, 1, None)


# ---------------------------------------------------------------------------
# A small synthetic geo_em and its own table row.
# ---------------------------------------------------------------------------

SRC = dict(map_proj="lambert", ref_lat=38.5, ref_lon=-97.5, truelat1=38.5,
           truelat2=38.5, stand_lon=-97.5, dx=3000.0, dy=3000.0,
           e_we=41, e_sn=31)
LEGEND = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "NUM_LAND_CAT": 21,
          "ISWATER": 17, "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13,
          "ISOILWATER": 14}


def _lambert(e_we, e_sn, dx=3000.0, ref_lat=38.5, ref_lon=-97.5):
    from gpuwm.static.projection import projection_class
    return projection_class("lambert")(ref_lat, ref_lon, 38.5, 38.5, -97.5,
                                       dx, dx, e_we, e_sn)


def _source_values():
    ny, nx = SRC["e_sn"] - 1, SRC["e_we"] - 1
    j, i = np.mgrid[0:ny, 0:nx].astype(np.float32)
    return {
        "LANDMASK": ((i + j) % 3 != 0).astype(np.float32),
        "LU_INDEX": (1 + (i * 7 + j) % 21).astype(np.float32),
        "LANDUSEF": np.stack([(i + k) / 100.0 for k in range(21)]).astype(np.float32),
        "SCT_DOM": (1 + (i + 2 * j) % 16).astype(np.float32),
        "SCB_DOM": (1 + (i + 3 * j) % 16).astype(np.float32),
        "SOILCTOP": np.stack([(j + k) / 50.0 for k in range(16)]).astype(np.float32),
        "SOILCBOT": np.stack([(i + k) / 50.0 for k in range(16)]).astype(np.float32),
        "HGT_M": (100.0 * i + j).astype(np.float32),
        "SLOPECAT": (1 + (i + j) % 9).astype(np.float32),
        "GREENFRAC": np.stack([(i + k) / 100.0 for k in range(12)]).astype(np.float32),
        "LAI12M": np.stack([(j + k) / 10.0 for k in range(12)]).astype(np.float32),
        "ALBEDO12M": np.stack([10.0 + (i + k) / 10.0 for k in range(12)]).astype(np.float32),
        "SNOALB": (20.0 + i).astype(np.float32),
        "SOILTEMP": (280.0 + i / 10.0).astype(np.float32),
        "LAKE_DEPTH": (2.0 + j).astype(np.float32),
        "VAR": (i * j).astype(np.float32),
        "VARSS": (i * j).astype(np.float32),
    }


def _write_geo_em(path: Path, values) -> None:
    from gpuwm.io.nc_writer_bridge import ClassicSchema
    ny, nx = SRC["e_sn"] - 1, SRC["e_we"] - 1
    lat, lon = _lambert(SRC["e_we"], SRC["e_sn"]).latlon_mass()
    schema = ClassicSchema()
    t = schema.def_dim("Time", 1)
    sn = schema.def_dim("south_north", ny)
    we = schema.def_dim("west_east", nx)
    cat = schema.def_dim("land_cat", 21)
    soil = schema.def_dim("soil_cat", 16)
    month = schema.def_dim("month", 12)
    for key, value in {**LEGEND, "WEST-EAST_GRID_DIMENSION": SRC["e_we"],
                       "SOUTH-NORTH_GRID_DIMENSION": SRC["e_sn"]}.items():
        schema.put_global_attr(key, value, "i4" if isinstance(value, int) else None)
    for key, value in (("DX", 3000.0), ("DY", 3000.0), ("TRUELAT1", 38.5),
                       ("TRUELAT2", 38.5), ("STAND_LON", -97.5),
                       ("CEN_LAT", float(np.nextafter(np.float32(38.5),
                                                     np.float32(39.0)))),
                       ("CEN_LON", -97.5)):
        schema.put_global_attr(key, value, "f4")
    ids = {"XLAT_M": schema.def_var("XLAT_M", "f4", (t, sn, we)),
           "XLONG_M": schema.def_var("XLONG_M", "f4", (t, sn, we))}
    for name, array in values.items():
        dims = (t, {21: cat, 16: soil, 12: month}[array.shape[0]], sn, we) \
            if array.ndim == 3 else (t, sn, we)
        ids[name] = schema.def_var(name, "f4", dims)
    with schema.create(path) as writer:
        writer.write_var(ids["XLAT_M"], np.asarray(lat, np.float32)[None])
        writer.write_var(ids["XLONG_M"], np.asarray(lon, np.float32)[None])
        for name, array in values.items():
            writer.write_var(ids[name], np.ascontiguousarray(array[None]))


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    pytest.importorskip("gpuwm.io.nc_writer_bridge")
    from gpuwm.io import nc_writer_bridge
    if nc_writer_bridge.unavailable_reason() is not None:
        pytest.skip(nc_writer_bridge.unavailable_reason())
    from gpuwm.netcdf_bridge import find_netcdf_bin
    if find_netcdf_bin() is None:
        pytest.skip("rw_netcdf is not built")
    values = _source_values()
    folder = tmp_path / "geog" / "static_sources" / "test-src"
    folder.mkdir(parents=True)
    path = folder / "geo_em.test.nc"
    _write_geo_em(path, values)
    data = path.read_bytes()
    grid_rows = "\n".join(f"{k} = {json.dumps(v)}" for k, v in SRC.items())
    legend_rows = "\n".join(f"{k} = {json.dumps(v)}" for k, v in LEGEND.items())
    table = tmp_path / "table.toml"
    table.write_text(f"""schema = "gpuwm-static-sources-v1"
[[source]]
id = "test-src"
describes = "synthetic"
format = "wps-geo_em"
filename = "geo_em.test.nc"
url = "https://example.invalid/geo_em.test.nc"
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
    yield SimpleNamespace(values=values, geog=tmp_path / "geog", path=path)
    es._load_table.cache_clear()


def _base_fields(ny, nx):
    return {"LANDMASK": np.zeros((ny, nx)), "LU_INDEX": np.ones((ny, nx)),
            "LANDUSEF": np.zeros((21, ny, nx)), "SCT_DOM": np.ones((ny, nx)),
            "SOILCTOP": np.zeros((16, ny, nx)), "HGT_M": np.zeros((ny, nx)),
            "SOILTEMP": np.full((ny, nx), 270.0), "TMN": np.full((ny, nx), 270.0)}


def test_crop_window_is_computed_from_identity(synthetic):
    row = es.static_source_row("test-src")
    assert es.crop_window(row, _lambert(21, 11)) == (10, 10)
    assert es.crop_window(row, _lambert(21, 11, dx=1000.0)) is None
    with pytest.raises(es.StaticSourceError, match="off the source's mass points"):
        es.crop_window(row, _lambert(22, 11))
    with pytest.raises(es.StaticSourceError, match="leaves the source"):
        es.crop_window(row, _lambert(61, 11))


def test_overlay_takes_the_file_exactly(synthetic):
    grid = _lambert(21, 11)
    ny, nx = 10, 20
    setting = es.StaticSourceSetting(id="test-src")
    fields = es.overlay_static_source(
        _base_fields(ny, nx), grid, synthetic.geog, setting,
        requested=("VARLS", "VARSS", "SLOPECAT"), landuse_attrs=LEGEND,
        report=(report := {}))
    window = (Ellipsis, slice(10, 10 + ny), slice(10, 10 + nx))
    for engine, file_name in (("LANDMASK", "LANDMASK"), ("LANDUSEF", "LANDUSEF"),
                               ("SOILCTOP", "SOILCTOP"), ("HGT_M", "HGT_M"),
                               ("SLOPECAT", "SLOPECAT"),
                              ("VARLS", "VAR"), ("VARSS", "VARSS")):
        expected = synthetic.values[file_name][window].astype(np.float64)
        assert fields[engine].dtype == np.float64
        assert np.array_equal(fields[engine], expected), engine
    land = fields["LANDMASK"] > 0.5
    tmn = np.where(land, fields["SOILTEMP"] - 0.0065 * fields["HGT_M"],
                   fields["SOILTEMP"])
    assert np.array_equal(fields["TMN"], tmn)
    applied = report["static_source"]
    assert applied["status"] == "APPLIED"
    assert applied["window"] == {"i0": 10, "j0": 10, "ni": 20, "nj": 10}
    assert applied["fields"]["VARLS"] == "VAR"


def test_overlay_groups_and_legend(synthetic):
    grid = _lambert(21, 11)
    base = _base_fields(10, 20)
    soil = es.StaticSourceSetting(id="test-src", groups=("soil",))
    fields = es.overlay_static_source(dict(base), grid, synthetic.geog, soil,
                                      landuse_attrs=LEGEND)
    assert np.array_equal(fields["LANDMASK"], base["LANDMASK"])
    assert np.array_equal(fields["HGT_M"], base["HGT_M"])
    assert not np.array_equal(fields["SCT_DOM"], base["SCT_DOM"])
    every = es.StaticSourceSetting(id="test-src")
    with pytest.raises(es.StaticSourceError, match="land-use legend"):
        es.overlay_static_source(dict(base), grid, synthetic.geog, every,
                                 landuse_attrs={**LEGEND, "ISLAKE": 28})


def test_missing_and_altered_file_are_refused(synthetic, tmp_path):
    row = es.static_source_row("test-src")
    with pytest.raises(FileNotFoundError, match="fetch-geog --static-source test-src"):
        es.resolve_local_file(row, tmp_path / "elsewhere")
    data = bytearray(synthetic.path.read_bytes())
    data[-1] ^= 1
    synthetic.path.write_bytes(bytes(data))
    with pytest.raises(es.StaticSourceError, match="sha256"):
        es.resolve_local_file(row, synthetic.geog)


def test_integrity_refuses_mutation_with_same_size_and_timestamp(synthetic):
    row = es.static_source_row("test-src")
    es.verify_local_file(synthetic.path, row)
    stat = synthetic.path.stat()
    # A stale stamp from an older reader must not attest different bytes.
    stamp = synthetic.path.with_name(synthetic.path.name + ".verified.json")
    stamp.write_text(json.dumps({"sha256": row.sha256, "bytes": stat.st_size,
                                 "mtime_ns": stat.st_mtime_ns}), encoding="utf-8")
    data = bytearray(synthetic.path.read_bytes())
    data[-1] ^= 1
    synthetic.path.write_bytes(bytes(data))
    os.utime(synthetic.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert synthetic.path.stat().st_size == stat.st_size
    assert synthetic.path.stat().st_mtime_ns == stat.st_mtime_ns
    with pytest.raises(es.StaticSourceError, match="sha256"):
        es.verify_local_file(synthetic.path, row)


def test_terrain_survey_reads_the_same_source_as_full_statics(synthetic):
    from gpuwm.static.build import GeogSelection, build_terrain
    grid = _lambert(21, 11)
    setting = es.StaticSourceSetting(id="test-src")
    selection = GeogSelection.fallback(synthetic.geog)
    from dataclasses import replace
    selection = replace(selection, static_source=setting)
    terrain = build_terrain(grid, synthetic.geog, selection=selection)
    fields = es.overlay_static_source(
        _base_fields(10, 20), grid, synthetic.geog, setting,
        requested=("SLOPECAT",), landuse_attrs=LEGEND)
    assert np.array_equal(terrain, fields["HGT_M"])
    assert np.array_equal(terrain, synthetic.values["HGT_M"][10:20, 10:30])


def test_complete_source_build_reads_its_full_inventory(synthetic, monkeypatch):
    from dataclasses import replace
    from gpuwm.static import build
    grid = _lambert(21, 11)
    selection = replace(build.GeogSelection.fallback(synthetic.geog),
                        static_source=es.StaticSourceSetting(id="test-src"))
    def refuse_baseline(*args, **kwargs):
        raise AssertionError("a complete static source must not build WPS fields")
    monkeypatch.setattr(build, "_build_static_routed", refuse_baseline)
    monkeypatch.setattr(build.GeogSelection, "landuse_global_attrs",
                        lambda self: LEGEND)
    report = {}
    fields = build.build_static(grid, synthetic.geog, selection=selection,
                                source_coverage_report=report)
    for name, file_name in selection.static_source.row.served().items():
        assert np.array_equal(fields[name],
                              synthetic.values[file_name][..., 10:20, 10:30]), name
    assert np.array_equal(fields["TMN"], build.deep_soil_temperature_at_terrain(
        fields["SOILTEMP"], fields["HGT_M"], fields["LANDMASK"]))
    assert report["static_source"]["status"] == "APPLIED"
    assert report["static_source"]["sha256"] == selection.static_source.sha256


@pytest.mark.parametrize("group", ["drag", "lake_depth", "terrain"])
def test_selected_optional_group_survives_inactive_physics_and_prepared_cache(
        synthetic, monkeypatch, tmp_path, group):
    from dataclasses import replace
    from gpuwm.static import build
    from gpuwm.native_wrf_contract import (
        load_native_static_cache, native_static_export_fields,
        verify_native_static_receipt, write_native_geometry_receipt,
        write_native_static_cache)
    from test_native_wrf_contract import _complete_native_static
    grid = _lambert(21, 11)
    setting = es.StaticSourceSetting(id="test-src", groups=(group,))
    selection = replace(build.GeogSelection.fallback(synthetic.geog),
                        static_source=setting)
    assert selection.orographic == () and not selection.lake_depth
    baseline = _complete_native_static(grid)
    monkeypatch.setattr(build, "_build_static_routed", lambda *a, **k: dict(baseline))
    monkeypatch.setattr(build.GeogSelection, "landuse_global_attrs",
                        lambda self: LEGEND)
    evidence = {}
    fields = build.build_static(grid, synthetic.geog, selection=selection,
                                source_coverage_report=evidence)
    wanted = set(setting.engine_names())
    assert set(evidence["static_source"]["fields"]) == wanted
    for name in wanted:
        file_name = setting.row.served()[name]
        np.testing.assert_array_equal(fields[name],
            synthetic.values[file_name][..., 10:20, 10:30])
    for name in set(baseline) - wanted - {"TMN"}:
        assert fields[name].tobytes() == baseline[name].tobytes()
    cache = tmp_path / "prepared-static.npz"
    cfg = SimpleNamespace(nx=20, ny=10, nz=2, dx=3000.0, dy=3000.0)
    write_native_static_cache(cache, native_static_export_fields(fields, grid))
    receipt = tmp_path / "geometry-receipt.json"
    write_native_geometry_receipt(receipt, grid, cfg, cache)
    verify_native_static_receipt(receipt, cache, grid, cfg)
    loaded = load_native_static_cache(cache, grid, cfg.ny, cfg.nx)
    for name in wanted:
        assert loaded[name].tobytes() == fields[name].tobytes()


def test_fetch_tries_pinned_mirror_after_wrong_origin_bytes(synthetic, tmp_path, monkeypatch):
    from dataclasses import replace
    from io import BytesIO
    from gpuwm import fetch_guard
    row = replace(es.static_source_row("test-src"),
                  mirrors=("https://example.invalid/mirror.nc",))
    monkeypatch.setattr(es, "static_source_row", lambda source_id: row)
    monkeypatch.setenv(fetch_guard.LOCK_ROOT_ENV, str(tmp_path / "locks"))
    data = synthetic.path.read_bytes()
    wrong = bytearray(data)
    wrong[-1] ^= 1
    attempted = []

    class Response(BytesIO):
        status = 200
        headers = {}

    def open_url(request):
        attempted.append(request.full_url)
        return Response(bytes(wrong) if request.full_url == row.url else data)

    root = tmp_path / "downloaded"
    final = es.fetch_static_source("test-src", root, progress=lambda message: None,
                                    urlopen_fn=open_url)
    assert attempted == [row.url, row.mirrors[0]]
    assert final.read_bytes() == data
    es.verify_local_file(final, row)


def _declare_sampling(synthetic):
    row = es.static_source_row("test-src")
    nearest = {"LANDMASK", "LU_INDEX", "SCT_DOM", "SCB_DOM", "SLOPECAT"}
    with es.TABLE_PATH.open("a", encoding="utf-8") as stream:
        stream.write("\n[source.field_sampling]\n")
        for name in row.fields:
            stream.write(f'{name} = "{("nearest" if name in nearest else "bilinear")}"\n')
    es._load_table.cache_clear()


def test_fractional_policy_keeps_strict_crop_and_samples_in_rust(synthetic):
    _declare_sampling(synthetic)
    row = es.static_source_row("test-src")
    # Quarter-cell origins keep categorical sampling away from a tie.
    center_lat, center_lon = es.source_grid(row).ij_to_latlon(20.25, 15.25)
    grid = _lambert(22, 12, ref_lat=float(center_lat), ref_lon=float(center_lon))
    with pytest.raises(es.StaticSourceError, match="off the source's mass points"):
        es.crop_window(row, grid)
    i0, j0 = es.sampling_window(row, grid)
    assert i0 == pytest.approx(9.25, abs=1e-10)
    assert j0 == pytest.approx(9.25, abs=1e-10)
    fields = es.overlay_static_source(_base_fields(11, 21), grid, synthetic.geog,
        es.StaticSourceSetting(id="test-src"), requested=("SLOPECAT",),
        landuse_attrs=LEGEND, report=(report := {}))
    for name in ("LANDMASK", "LU_INDEX", "SCT_DOM", "SLOPECAT"):
        np.testing.assert_array_equal(fields[name], synthetic.values[name][9:20, 9:30])
    expected = 100.0 * (9.25 + np.arange(21))[None, :] + (9.25 + np.arange(11))[:, None]
    np.testing.assert_allclose(fields["HGT_M"], expected, rtol=0, atol=1e-7)
    evidence = report["static_source"]
    assert evidence["sampling"]["mode"] == "fractional-same-spacing"
    assert evidence["sampling"]["methods"]["HGT_M"] == "bilinear"
    assert evidence["sampling"]["methods"]["SCT_DOM"] == "nearest"
    assert es.sampling_window(row, _lambert(21, 11, dx=1000.)) is None
    with pytest.raises(es.StaticSourceError, match="leaves the source"):
        es.sampling_window(row, _lambert(62, 12))


def test_native_sample_integer_origin_has_exact_crop_bytes(synthetic):
    from gpuwm.netcdf_bridge import Dataset
    variable = Dataset(synthetic.path).variables["HGT_M"]
    exact = variable.read_window(i0=10, ni=20, j0=10, nj=10)
    for method in ("nearest", "bilinear"):
        sampled = variable.read_sample_window(i0=10., ni=20, j0=10., nj=10, method=method)
        assert sampled.tobytes() == exact.tobytes()
    fractional = variable.read_sample_window(i0=9.5, ni=21, j0=9.5, nj=11, method="bilinear")
    expected = 100.0 * (9.5 + np.arange(21))[None, :] + (9.5 + np.arange(11))[:, None]
    np.testing.assert_array_equal(fractional[0], expected)


def test_fractional_source_registration_mutation_is_refused(synthetic, monkeypatch):
    _declare_sampling(synthetic)
    from gpuwm.netcdf_bridge import Variable
    original = Variable.read_window
    def shifted_coordinates(self, **window):
        if self.name == "XLONG_M":
            return original(self, **window) + 0.03
        return original(self, **window)
    monkeypatch.setattr(Variable, "read_window", shifted_coordinates)
    with pytest.raises(es.StaticSourceError, match="latitudes and longitudes"):
        es.overlay_static_source(_base_fields(11, 21), _lambert(22, 12), synthetic.geog,
            es.StaticSourceSetting(id="test-src"), landuse_attrs=LEGEND)
