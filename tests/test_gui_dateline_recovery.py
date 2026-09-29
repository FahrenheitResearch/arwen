"""The map viewer across the 180th meridian, run in Node on the shipped modules.

A forecast whose pictures cross the 180th meridian used to be stretched
round the world.  Framing it as the few degrees it covers is not enough on
its own: the field, its outline and the frame must sit on one copy of the
world, or the field is drawn a whole world width away from the view and is
not seen.  A picture with no grid on record, centred east of 180 (179 W),
was placed on longitudes -189 to -169 while the view framed 171 to 191.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "gpuwm" / "gui" / "static" / "js"

SCRIPT = r"""
import { grids, gridOutline, gridForPicture, placement, wrapLon, toWorld, boundsBox, screenRings, worldShifts }
  from './geo.mjs';
import { placePicture, drawPlaced } from './field.mjs';
// Canvas calls do not change where a picture sits: the real triangulation runs with drawing that does nothing.
globalThis.document = {createElement: () => ({width: 0, height: 0, getContext: () => new Proxy({}, {get: () => () => {}})})};
globalThis.Image = class { naturalWidth = 200; naturalHeight = 200; set src(v) { queueMicrotask(() => this.onload()); } };
const out = {};

// a geographic picture from 170 to 190 degrees with no grid on record
const geo = {projection: {kind: 'geographic', central_meridian_deg: 0}, plot_rect_px: {x: 0, y: 0, width: 200, height: 200},
  extent: {x_min: 170, x_max: 190, y_min: -10, y_max: 10}};
const placed = await placePicture('local', geo, null);
out.span = (placed.bent.box[2] - placed.bent.box[0]) * 360;
out.ratio = placed.bent.canvas.width / placed.bent.canvas.height;

// the same picture cut along a grid whose longitudes the model writes as 170..180 and -180..-170
const grid = {nx: 11, ny: 11, toLatLon: (i, j) => [-10 + 2 * j, wrapLon(170 + 2 * i)]};
const clipped = placement(geo, grid);
out.pixelMin = Math.min(...clipped.verts.map((p) => p.u));
out.pixelMax = Math.max(...clipped.verts.map((p) => p.u));

// a 960 km Lambert grid centred at 17S 179E
const root = grids({map_proj: 'lambert', ref_lat: -17, ref_lon: 179, truelat1: -17, truelat2: -17, stand_lon: 179},
  [{grid_id: 1, parent_id: 0, nx: 81, ny: 81, dx_km: 12}]).get(1);
const ring = gridOutline(root, 8); const lons = ring.map((p) => p[0]); const lats = ring.map((p) => p[1]);
const b = [Math.min(...lons), wrapLon(Math.max(...lons)), Math.min(...lats), Math.max(...lats)];
out.lambertSpan = Math.max(...lons) - Math.min(...lons);
out.snapStable = gridForPicture(root, {geographic_bounds: b}).key === root.key;

// whole-world extents, written four ways, stay whole
out.globals = [];
for (const [x_min, x_max] of [[-180, 180], [0, 360], [90, 450], [180, -180]]) {
  const p = await placePicture(`world-${x_min}`, {...geo, extent: {x_min, x_max, y_min: -70, y_max: 70}}, null);
  out.globals.push((p.bent.box[2] - p.bent.box[0]) * 360);
}

// Pictures with no grid on record, centred either side of 180, with the renderer's own bounds (west > east
// across the meridian): the field's world box against the box the viewer frames (boundsBox), in world units.
const worldX = (lon) => toWorld(lon, 0)[0];
out.centred = {};
for (const c of [-179, -175, -170, 170, 175, 179, 180]) {
  const record = {image_width_px: 1200, image_height_px: 900, plot_rect_px: {x: 20, y: 60, width: 1000, height: 800},
    projection: {kind: 'geographic', central_meridian_deg: c}, extent: {x_min: -8, x_max: 8, y_min: 10, y_max: 30},
    geographic_bounds: [wrapLon(c - 8), wrapLon(c + 8), 10, 30]};
  const p = placement(record, null);
  const xs = p.verts.map((v) => worldX(v.lon));
  const frame = boundsBox([record]);
  out.centred[c] = {field: [Math.min(...xs), Math.max(...xs)], frame: [worldX(frame[0]), worldX(frame[2])]};
}
// a Lambert picture of an imported run (no plan projection, so no grid) centred at 17S 179W
const lam = {image_width_px: 1200, image_height_px: 1200, plot_rect_px: {x: 20, y: 20, width: 1100, height: 1100},
  projection: {kind: 'lambert_conformal', standard_parallel_1_deg: -10, standard_parallel_2_deg: -25, central_meridian_deg: -179,
    reference_latitude_deg: -17}, extent: {x_min: -480000, x_max: 480000, y_min: -480000, y_max: 480000},
  geographic_bounds: [176.4, -174.4, -21.3, -12.6]};
{
  const p = placement(lam, null);
  const xs = p.verts.map((v) => worldX(v.lon));
  const frame = boundsBox([lam]);
  out.lambert = {field: [Math.min(...xs), Math.max(...xs)], frame: [worldX(frame[0]), worldX(frame[2])]};
}
// two pictures either side of the meridian are framed together, not 360 degrees apart
const pair = boundsBox([{geographic_bounds: [170, -170, -20, -10]}, {geographic_bounds: [-178, -172, -18, -12]}]);
out.pairSpan = pair[2] - pair[0];

// The map, as the page makes it, framed on 171..191 (world x about 0.975..1.031): a field placed on the other
// copy of the world (-189..-169) is still drawn on screen, where the view looks.
const scale = 256 * 2 ** 5;
const view = (cx) => ({cx, cy: 0.5, width: 900, height: 600,
  worldToScreen(wx, wy) { return [(wx - this.cx) * scale + this.width / 2, (wy - this.cy) * scale + this.height / 2]; },
  screen(lon, lat) { const [wx, wy] = toWorld(lon, lat); return this.worldToScreen(wx, wy); }});
const drawn = (map, box) => {
  const calls = [];
  const ctx = {drawImage: (...a) => calls.push(a.slice(1))};
  drawPlaced(ctx, map, {bent: {canvas: {width: 400, height: 300}, box}}, 0.8);
  return calls.filter(([x, y, w]) => x + w >= 0 && x <= map.width);
};
const eastView = view(worldX(181));
out.drawnFromWest = drawn(eastView, [worldX(-189), 0.5, worldX(-169), 0.52]).length;
out.drawnFromEast = drawn(view(worldX(-179)), [worldX(171), 0.5, worldX(191), 0.52]).length;
out.drawnHere = drawn(eastView, [worldX(171), 0.5, worldX(191), 0.52]).length;
// the area asked for, written 170 to -170, is drawn as the 20 degrees it covers, where the view looks
const region = [[170, -20], [-170, -20], [-170, -10], [170, -10], [170, -20]];
const rings = screenRings(region, eastView);
out.regionRings = rings.length;
out.regionWidth = rings.length ? Math.max(...rings[0].map((p) => p[0])) - Math.min(...rings[0].map((p) => p[0])) : null;
out.regionTwenty = (worldX(190) - worldX(170)) * scale;
out.regionOnScreen = rings.length ? rings[0].every(([x]) => x >= 0 && x <= eastView.width) : false;

// A 1920 pixel window zoomed right out (the map's least zoom, 1.2) and panned as far east as it goes (centre
// 1.2 world widths) shows the world from -0.43 to 2.83: four copies of the land, three of a field or an
// outline near 0 degrees, and none beyond what it shows.
const wideScale = 256 * 2 ** 1.2;
const wide = {cx: 1.2, cy: 0.5, width: 1920, height: 900,
  worldToScreen(wx, wy) { return [(wx - this.cx) * wideScale + this.width / 2, (wy - this.cy) * wideScale + this.height / 2]; },
  screen(lon, lat) { const [wx, wy] = toWorld(lon, lat); return this.worldToScreen(wx, wy); }};
const half = wide.width / 2 / wideScale;
out.wideLand = worldShifts(0, 1, wide.cx - half, wide.cx + half);
const wideCalls = [];
drawPlaced({drawImage: (...a) => wideCalls.push(a.slice(1))}, wide,
  {bent: {canvas: {width: 400, height: 300}, box: [worldX(-10), 0.45, worldX(10), 0.55]}}, 1);
out.wideFieldLefts = wideCalls.map(([x]) => x);
out.wideFieldOnScreen = wideCalls.every(([x, y, w]) => x + w >= 0 && x <= wide.width);
out.wideRings = screenRings([[-10, -5], [10, -5], [10, 5], [-10, 5], [-10, -5]], wide).length;
out.oneWorldPx = wideScale;
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def seam(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    folder = tmp_path_factory.mktemp("date-line-placement")
    (folder / "geo.mjs").write_text((SOURCE / "geo.js").read_text(encoding="utf-8"), encoding="utf-8")
    (folder / "field.mjs").write_text(
        (SOURCE / "field.js").read_text(encoding="utf-8").replace("./geo.js", "./geo.mjs"), encoding="utf-8")
    (folder / "probe.mjs").write_text(SCRIPT, encoding="utf-8")
    result = subprocess.run([node, "probe.mjs"], cwd=folder, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_a_local_date_line_field_has_a_local_footprint(seam):
    assert 19 < seam["span"] < 21
    assert 0.95 < seam["ratio"] < 1.05


def test_geographic_pixel_placement_keeps_both_sides_in_the_picture(seam):
    assert 0 <= seam["pixelMin"] < 5
    assert 195 < seam["pixelMax"] <= 199


def test_a_date_line_grid_frames_locally_and_does_not_snap_across_the_world(seam):
    assert 9 < seam["lambertSpan"] < 10
    assert seam["snapStable"]


def test_equivalent_global_extents_remain_global(seam):
    assert all(350 < span <= 360 for span in seam["globals"])


@pytest.mark.parametrize("centre", ["-179", "-175", "-170", "170", "175", "179", "180"])
def test_a_picture_without_a_grid_is_placed_in_the_box_the_view_frames(seam, centre):
    field = seam["centred"][centre]["field"]
    frame = seam["centred"][centre]["frame"]
    # one world copy: the field inside the frame, to well under a degree (a degree is 1/360 of the world)
    assert field[0] >= frame[0] - 0.5 / 360
    assert field[1] <= frame[1] + 0.5 / 360
    assert field[1] - field[0] > 15 / 360


def test_a_lambert_picture_without_a_grid_is_placed_in_the_box_the_view_frames(seam):
    field, frame = seam["lambert"]["field"], seam["lambert"]["frame"]
    # the picture's corners reach a little past the domain's bounds, never a world away
    assert abs((field[0] + field[1]) / 2 - (frame[0] + frame[1]) / 2) < 3 / 360
    assert field[1] - field[0] < 15 / 360


def test_pictures_either_side_of_the_meridian_are_framed_together(seam):
    assert seam["pairSpan"] == pytest.approx(20)


def test_a_field_is_drawn_on_the_world_copy_the_view_looks_at(seam):
    assert seam["drawnFromWest"] == 1
    assert seam["drawnFromEast"] == 1
    assert seam["drawnHere"] == 1


def test_the_area_asked_for_is_outlined_as_the_degrees_it_covers(seam):
    assert seam["regionRings"] == 1
    assert seam["regionOnScreen"]
    assert seam["regionWidth"] == pytest.approx(seam["regionTwenty"], rel=1e-6)


def test_a_wide_window_zoomed_right_out_draws_every_copy_it_shows(seam):
    # the land on the four copies the window shows, the field and its outline on the three that hold 0 degrees
    assert seam["wideLand"] == [-1, 0, 1, 2]
    lefts = seam["wideFieldLefts"]
    assert len(lefts) == 3
    assert seam["wideFieldOnScreen"]
    # one world width apart, as the land is
    assert [b - a for a, b in zip(lefts, lefts[1:])] == pytest.approx([seam["oneWorldPx"]] * 2)
    assert seam["wideRings"] == 3
