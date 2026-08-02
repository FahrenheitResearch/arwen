"""Generic polygon-bound domain authoring, entirely CPU-side."""

from __future__ import annotations

import json
import tomllib

import numpy as np
import pytest

from gpuwm.cli import main as cli_main
from gpuwm.domain_wizard import (_buffer_cells, load_polygon_footprint,
                                 verify_polygon_containment)
from gpuwm.experiment import load_experiment
from gpuwm.static.projection import grids_from_projection_config


def _ring(west=-97.2, south=35.1, east=-96.8, north=35.4):
    return [[west, south], [east, south], [east, north], [west, north],
            [west, south]]


def _write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.mark.parametrize(("document", "ring_count"), [
    ({"type": "Polygon", "coordinates": [_ring()]}, 1),
    ({"type": "MultiPolygon",
      "coordinates": [[_ring()], [_ring(-96.6, 35.0, -96.4, 35.2)]]}, 2),
    ({"type": "Feature", "properties": {},
      "geometry": {"type": "Polygon", "coordinates": [_ring()]}}, 1),
    ({"type": "FeatureCollection",
      "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
      "features": [
        {"type": "Feature", "properties": {}, "geometry": {
            "type": "MultiPolygon", "coordinates": [
                [_ring()], [_ring(-96.6, 35.0, -96.4, 35.2)]]}}]}, 2),
])
def test_supported_geojson_containers_supply_safe_bounds(
        tmp_path, document, ring_count):
    footprint = load_polygon_footprint(
        _write(tmp_path / "target.geojson", document))
    assert len(footprint.rings) == ring_count
    assert footprint.south <= footprint.center_lat <= footprint.north
    assert footprint.west <= footprint.center_lon <= footprint.east
    assert 0.0 <= footprint.longitude_span <= 180.0


def test_antimeridian_uses_the_minimum_circular_span(tmp_path):
    document = {"type": "Polygon", "coordinates": [[
        [179.8, 10.0], [-179.7, 10.0], [-179.7, 10.3],
        [179.8, 10.3], [179.8, 10.0],
    ]]}
    footprint = load_polygon_footprint(
        _write(tmp_path / "crossing.geojson", document))
    assert abs(abs(footprint.center_lon) - 180.0) < 0.1
    assert 0.5 * (footprint.west + footprint.east) == pytest.approx(
        footprint.center_lon)
    assert footprint.longitude_span == pytest.approx(0.5)


def test_cli_emits_every_level_containing_polygon_and_its_buffer(
        tmp_path, capsys):
    polygon = _write(tmp_path / "crossing.geojson", {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": {
            "type": "MultiPolygon", "coordinates": [[[
                [179.8, 10.0], [-179.7, 10.0], [-179.7, 10.3],
                [179.8, 10.3], [179.8, 10.0],
            ]]]}}],
    })
    out = tmp_path / "domain.toml"
    rc = cli_main([
        "domain", "--polygon", str(polygon), "--buffer-km", "80,40,20",
        "--card", "32gb", "--ladder", "12-3-1", "--source", "gfs",
        "--cycle", "2000-01-01T00", "--hours", "3", "--out", str(out),
    ])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert out.is_file()
    assert "polygon center" in printed
    assert "buffers 80,40,20 km" in printed

    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    south, west, north, east = (
        float(value) for value in raw["fetch"]["area"].split(","))
    assert south < north
    assert west > 0.0 > east
    assert "Polygon buffers by domain level" in out.read_text(
        encoding="utf-8")

    footprint = load_polygon_footprint(polygon)
    exp = load_experiment(out)
    buffers = (80.0, 40.0, 20.0)
    verify_polygon_containment(exp, footprint, buffers)
    sample_lats = np.asarray(
        [position[1] for ring in footprint.rings for position in ring])
    sample_lons = np.asarray(
        [position[0] for ring in footprint.rings for position in ring])
    for grid, buffer_km in zip(grids_from_projection_config(exp), buffers):
        i, j = grid.latlon_to_ij(sample_lats, sample_lons)
        required = _buffer_cells(grid, footprint, buffer_km)
        actual = min(float(np.min(i)) - 0.5,
                     float(grid.e_we) - 0.5 - float(np.max(i)),
                     float(np.min(j)) - 0.5,
                     float(grid.e_sn) - 0.5 - float(np.max(j)))
        assert actual >= required


def test_footprint_wider_than_half_the_globe_is_refused(tmp_path, capsys):
    features = []
    for west in (-171.0, -11.0, 109.0):
        features.append({"type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon",
            "coordinates": [_ring(west, 5.0, west + 2.0, 6.0)]}})
    polygon = _write(tmp_path / "too-wide.geojson", {
        "type": "FeatureCollection", "features": features})
    out = tmp_path / "domain.toml"
    rc = cli_main([
        "domain", "--polygon", str(polygon), "--card", "32gb",
        "--ladder", "12", "--source", "gfs",
        "--cycle", "2000-01-01T00", "--out", str(out),
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert "wider than 180 degrees" in captured.err
    assert "Traceback" not in captured.err
    assert not out.exists()


def test_unfittable_buffered_polygon_refuses_without_writing(
        tmp_path, capsys):
    polygon = _write(tmp_path / "large.geojson", {
        "type": "Polygon", "coordinates": [
            _ring(-101.0, 32.0, -93.0, 39.0)]})
    out = tmp_path / "domain.toml"
    rc = cli_main([
        "domain", "--polygon", str(polygon), "--buffer-km", "50",
        "--vram-gib", "5", "--root-dx", "0.1", "--source", "gfs",
        "--cycle", "2000-01-01T00", "--out", str(out),
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert "polygon plus the requested per-level buffers requires" \
        in captured.err
    assert "Traceback" not in captured.err
    assert not out.exists()


def test_polygon_input_is_local_and_buffer_count_matches_levels(
        tmp_path, capsys):
    with pytest.raises(ValueError, match="local GeoJSON file path"):
        load_polygon_footprint("https://example.invalid/target.geojson")

    polygon = _write(tmp_path / "target.geojson", {
        "type": "Polygon", "coordinates": [_ring()]})
    out = tmp_path / "domain.toml"
    rc = cli_main([
        "domain", "--polygon", str(polygon), "--buffer-km", "20,10",
        "--card", "32gb", "--ladder", "12-3-1", "--source", "gfs",
        "--cycle", "2000-01-01T00", "--out", str(out),
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert "supplies 2 distances" in captured.err
    assert "selected ladder has 3 domain levels" in captured.err
    assert not out.exists()


def test_point_route_rejects_polygon_only_buffer_without_writing(
        tmp_path, capsys):
    out = tmp_path / "domain.toml"
    rc = cli_main([
        "domain", "--point=35.2,-97.1", "--buffer-km", "20",
        "--card", "32gb", "--ladder", "12", "--source", "gfs",
        "--cycle", "2000-01-01T00", "--out", str(out),
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--buffer-km requires --polygon" in captured.err
    assert not out.exists()


def test_hrrr_polygon_outside_native_coverage_refuses_before_writing(
        tmp_path, capsys):
    polygon = _write(tmp_path / "atlantic.geojson", {
        "type": "Polygon", "coordinates": [
            _ring(-40.2, 24.8, -39.8, 25.2)]})
    out = tmp_path / "atlantic.toml"

    rc = cli_main([
        "domain", "--polygon", str(polygon), "--buffer-km", "20",
        "--card", "32gb", "--ladder", "12", "--source", "hrrr",
        "--cycle", "2026-07-29T18", "--hours", "1", "--out", str(out),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "polygon" in captured.err
    assert "outside HRRR coverage" in captured.err
    assert "--polygon" in captured.err
    assert "Traceback" not in captured.err
    assert not out.exists()
    assert not list(tmp_path.glob("atlantic.*.input"))
    assert not list(tmp_path.glob("atlantic.*.json"))
    assert not (tmp_path / "atlantic.namelist.wps").exists()


def test_valid_hrrr_polygon_emits_round_tripping_route_bundle(
        tmp_path, capsys):
    from gpuwm.hrrr_route_inputs import route_input_paths, verify_round_trip

    polygon = _write(tmp_path / "oklahoma.geojson", {
        "type": "Polygon", "coordinates": [_ring()]})
    out = tmp_path / "oklahoma.toml"

    rc = cli_main([
        "domain", "--polygon", str(polygon), "--buffer-km", "20",
        "--card", "32gb", "--ladder", "12", "--source", "hrrr",
        "--cycle", "2026-07-29T18", "--hours", "1", "--out", str(out),
    ])

    captured = capsys.readouterr()
    assert rc == 0, captured.err
    paths = route_input_paths(out)
    assert out.is_file()
    assert all(path.is_file() for path in paths.values())
    exp = load_experiment(out)
    verify_round_trip(exp, paths["wps_namelist"], paths["namelist_input"])
