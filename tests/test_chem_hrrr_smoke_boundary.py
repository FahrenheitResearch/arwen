"""HRRR-Smoke's own 3-D smoke as WOOF's start state and edge values (CPU).

The hrrr-native-smoke row is a data-store boundary source: its wrfnat
MASSDEN/PRES/TMP records are fetched by byte range from the public bucket
(``noaa_index`` acquisition), decoded on HRRR's Lambert grid (GRIB2 template
3.30), converted from kg m-3 to ug kg-1 with each level's own density, placed
in the vertical by the levels' own pressure, and filled into every forcing
frame by gpuwm.chem_source_init, so the start state and the lateral boundary
carry it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.chem_table import catalog

ROW = catalog().sources["hrrr-native-smoke"]


def _index(names):
    lines, offset = [], 0
    for n, name in enumerate(names, start=1):
        lines.append(f"{n}:{offset}:d=2025010812:{name}:anl:")
        offset += 100
    return "\n".join(lines) + "\n"


def test_index_selection_takes_each_wanted_record_once():
    from gpuwm.data_store_fetch import index_selection
    text = _index(["TMP:1 hybrid level", "MASSDEN:1 hybrid level", "UGRD:1 hybrid level",
                   "MASSDEN:2 hybrid level"])
    chosen = index_selection(text, {"MASSDEN:1 hybrid level", "MASSDEN:2 hybrid level"})
    assert chosen == [("MASSDEN:1 hybrid level", 100, 199), ("MASSDEN:2 hybrid level", 300, None)]
    with pytest.raises(ValueError, match="missing"):
        index_selection(text, {"MASSDEN:3 hybrid level"})
    with pytest.raises(ValueError, match="repeated"):
        index_selection(text + "5:400:d=2025010812:TMP:1 hybrid level:anl:\n", {"TMP:1 hybrid level"})


def test_the_plan_asks_for_smoke_with_its_own_pressure_and_temperature():
    from gpuwm.data_store_fetch import resolve_acquisition
    plan = resolve_acquisition(ROW, area="31.5,-122,36,-115.8",
                               start="2025-01-08T12:00:00+00:00",
                               end="2025-01-09T00:00:00+00:00")
    groups = {r.group: dict(r.request) for r in plan.requests}
    assert set(groups) == {"hybrid_level", "surface"}
    assert groups["hybrid_level"]["variable"] == ("MASSDEN", "PRES", "TMP")
    assert len(groups["hybrid_level"]["hybrid_level"]) == 50
    assert groups["surface"]["variable"] == ("PRES",)
    assert groups["hybrid_level"]["time"] == ("12:00",)
    assert groups["hybrid_level"]["leadtime_hour"][0] == "0"
    assert groups["hybrid_level"]["leadtime_hour"][-1] == "12"


def test_index_range_download_concatenates_every_lead_in_inventory_order(tmp_path):
    from gpuwm.data_store_fetch import index_range_download, resolve_acquisition
    plan = resolve_acquisition(ROW, area="31.5,-122,36,-115.8",
                               start="2025-01-08T12:00:00+00:00",
                               end="2025-01-08T13:00:00+00:00")
    request = next(r for r in plan.requests if r.group == "surface")
    names = ["UGRD:surface", "PRES:surface", "TMP:surface"]
    calls = []

    def opener(url, headers=None):
        calls.append((url, dict(headers or {})))
        if url.endswith(".idx"):
            return _index(names).encode()
        first = int(headers["Range"].split("=")[1].split("-")[0])
        return b"GRIB" + url[-8:-6].encode() + str(first).encode()

    part = tmp_path / "out.grib2"
    index_range_download(request, part, progress=lambda *a: None, opener=opener)
    urls = [u for u, _h in calls if not u.endswith(".idx")]
    assert urls == [
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20250108/conus/hrrr.t12z.wrfnatf00.grib2",
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20250108/conus/hrrr.t12z.wrfnatf01.grib2"]
    assert [h["Range"] for u, h in calls if "Range" in h] == ["bytes=100-199", "bytes=100-199"]
    assert part.read_bytes() == b"GRIB00100GRIB01100"


def test_a_lambert_stack_reads_as_index_axes_with_its_projection(tmp_path):
    import hashlib
    import json
    from gpuwm.grib2_stack import read_stack
    data = np.arange(2 * 3 * 4, dtype="<f4").tobytes()
    (tmp_path / "x.f32le").write_bytes(data)
    grid = {"template": 30, "nx": 4, "ny": 3, "lat1": 21.138123, "lat2": 0, "lon1": 237.280472,
            "lon2": 0, "dlat": 3000.0, "dlon": 3000.0, "scan_mode": 64, "latin1": 38.5,
            "latin2": 38.5, "lov": 262.5}
    row = {"key": "x", "file": "x.f32le", "sha256": hashlib.sha256(data).hexdigest(),
           "levels": [1, 2], "pdt": 0, "valid_time": "2025-01-08T12:00:00+00:00", "grid": grid}
    (tmp_path / "stack.json").write_text(json.dumps(
        {"schema": "gpuwm-grib2-stack-v1", "outputs": [row], "coordinate_values": {"x": []}}))
    (field,) = read_stack(tmp_path)
    assert field.array.shape == (2, 3, 4)
    assert field.projection["lov"] == 262.5 and field.projection["nx"] == 4
    assert list(field.latitude) == [0, 1, 2] and list(field.longitude) == [0, 1, 2, 3]
    grid["scan_mode"] = 0
    (tmp_path / "stack.json").write_text(json.dumps(
        {"schema": "gpuwm-grib2-stack-v1", "outputs": [row], "coordinate_values": {"x": []}}))
    with pytest.raises(ValueError, match="wrong order"):
        read_stack(tmp_path)


def _hrrr_projection(nx=1799, ny=1059):
    return {"template": 30, "nx": nx, "ny": ny, "lat1": 21.138123, "lon1": 237.280472,
            "dlat": 3000.0, "dlon": 3000.0, "latin1": 38.5, "latin2": 38.5, "lov": 262.5,
            "scan_mode": 64}


def test_targets_land_on_hrrrs_own_grid_points():
    from gpuwm.chem_boundary_remap import TargetGrid, _horizontal_plan
    from gpuwm.ingest.hrrr import hrrr_source_grid
    grid = hrrr_source_grid()
    # Two HRRR mass points (0-based i, j) as geographic targets.
    lat, lon = grid.ij_to_latlon(np.array([[401.0, 1201.0]]), np.array([[301.0, 801.0]]))
    seen = {}
    backend = SimpleNamespace(indexed_plan=lambda shape, y, x: seen.update(shape=shape, y=y, x=x) or "plan")
    field = SimpleNamespace(projection=_hrrr_projection())
    target = TargetGrid(lat, lon, np.ones((2, 1, 2), np.float32))
    assert _horizontal_plan(backend, ROW, [field], None, None, target) == "plan"
    assert seen["shape"] == (1059, 1799)
    assert np.allclose(seen["x"], [[400.0, 1200.0]], atol=1e-6)
    assert np.allclose(seen["y"], [[300.0, 800.0]], atol=1e-6)
    far = TargetGrid(np.array([[60.0]]), np.array([[-150.0]]), np.ones((2, 1, 1), np.float32))
    with pytest.raises(ValueError, match="outside"):
        _horizontal_plan(backend, ROW, [field], None, None, far)


def test_mass_density_becomes_mixing_ratio_on_the_models_levels():
    """A synthetic HRRR column end to end on the real Rust operators: smoke
    density rho*q with rho = P/(R T) on four hybrid levels numbered from the
    ground, onto target pressures inside the source column."""
    from gpuwm.chem_boundary_remap import SourceFrames, TargetGrid, boundary_fields_for_grid
    from gpuwm.chem_table import load_sets
    try:
        from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
        backend = CpuPreprocessBackend()
        backend.weighted_combination
    except Exception as error:   # pragma: no cover - a box without the bridge
        pytest.skip(f"CPU preprocessing bridge not loadable: {error}")
    nx, ny, nz = 6, 5, 4
    proj = _hrrr_projection(nx, ny)
    p_levels = np.array([95000.0, 85000.0, 70000.0, 50000.0], np.float32)   # level 1 = ground
    pres = np.broadcast_to(p_levels[:, None, None], (nz, ny, nx)).astype(np.float32)
    tmp = np.full((nz, ny, nx), 280.0, np.float32)
    q = np.float32(20.0)                                    # ug/kg
    rho = pres / (np.float32(287.04) * tmp)
    massden = (q * 1e-9 * rho).astype(np.float32)
    psfc = np.full((ny, nx), 100000.0, np.float32)

    def fld(values, levels):
        return SimpleNamespace(values=values, levels=levels, latitude=np.arange(ny, dtype=float),
                               longitude=np.arange(nx, dtype=float), coordinate_values=(),
                               projection=proj)
    when = datetime(2025, 1, 8, 12, tzinfo=timezone.utc)
    levels = tuple(float(k) for k in range(1, nz + 1))
    frame = {"MASSDEN": fld(massden, levels), "PRES": fld(pres, levels), "TMP": fld(tmp, levels),
             "PSFC": fld(psfc[None], (0.0,))}
    frames = SourceFrames("hrrr-native-smoke", {when: frame})
    from gpuwm.ingest.hrrr import hrrr_source_grid
    lat, lon = hrrr_source_grid().ij_to_latlon(np.array([[2.5, 3.5]]), np.array([[2.0, 2.5]]))
    target = TargetGrid(lat, lon, np.broadcast_to(
        np.array([90000.0, 80000.0, 60000.0], np.float32)[:, None, None], (3, 1, 2)).copy())
    table = load_sets(["smoke"], ["rave-3km", "hrrr-native-smoke"])
    out, receipt = boundary_fields_for_grid(table, ROW, frames, target, when, backend=backend,
                                            available=("rave-3km", "hrrr-native-smoke"))
    assert set(out) == {"smoke"}
    assert np.allclose(out["smoke"], 20.0, rtol=1e-4)
    assert "kg_m3_to_ug_kg_dry" in receipt["operators"]["convert_and_time"]
    assert "Lambert" in receipt["operators"]["horizontal"]


def test_a_run_started_from_hrrr_takes_hrrr_smoke_without_naming_it():
    from gpuwm.config import apply_route_chem_sources
    raw = {"shared": {"chem_sets": "smoke", "chem_sources": "rave-3km"}}
    apply_route_chem_sources(raw, {"source": "hrrr-prs"})
    assert raw["shared"]["chem_sources"] == "rave-3km,hrrr-native-smoke"
    # Stated already: unchanged.  Another start (GFS) or chem off: unchanged.
    raw = {"shared": {"chem_sets": ["smoke"], "chem_sources": ["hrrr-native-smoke", "rave-3km"]}}
    apply_route_chem_sources(raw, {"source": "hrrr"})
    assert raw["shared"]["chem_sources"] == ["hrrr-native-smoke", "rave-3km"]
    for raw, hints in (({"shared": {"chem_sets": "smoke", "chem_sources": "rave-3km"}}, {"source": "gfs"}),
                       ({"shared": {"chem_sets": "smoke", "chem_sources": "rave-3km"}}, None),
                       ({"shared": {"chem_sources": "rave-3km"}}, {"source": "hrrr"}),
                       ({"shared": {"chem_sets": "gocart_primary"}}, {"source": "hrrr"})):
        before = dict(raw["shared"])
        apply_route_chem_sources(raw, hints)
        assert raw["shared"] == before


def test_the_hrrr_smoke_row_is_a_data_store_boundary_source_with_its_own_levels():
    from gpuwm.chem_source_init import chem_boundary_fields
    from gpuwm.chem_table import load_sets
    from gpuwm.config import RunConfig, validate_chem_config
    assert ROW.acquisition["kind"] == "noaa_index"
    assert ROW.vertical["kind"] == "pressure_field"
    assert set(ROW.default_for_fetch_sources) == {"hrrr", "hrrr-prs"}
    cfg = RunConfig(nx=12, ny=10, nz=6, dx=3000.0, dy=3000.0, ztop=12000.0, dt=18.0,
                    run_seconds=60.0, bl_pbl_physics=1, sf_sfclay_physics=1,
                    chem_sets="smoke", chem_sources="rave-3km,hrrr-native-smoke")
    validate_chem_config(cfg)          # the old "no forcing-frame fill" refusal is retired
    table = load_sets(["smoke"], ["rave-3km", "hrrr-native-smoke"])
    assert chem_boundary_fields(table, cfg) == ("chem_smoke",)
