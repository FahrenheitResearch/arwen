"""The map's arithmetic (gpuwm/gui/static/js/geo.js), run in Node.

A nest that follows a storm moves while the run goes.  The page must cut
each renderer picture along the grid it was drawn on: the plan's first
place for a moved nest is about 100 km off, and cutting there brings the
renderer's white margin and colour bar strip into the map as if it were
data.  The numbers are those of a real 3 km nest that moved four times
(two parent cells west each time) inside a 12 km Mercator grid.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

GEO = Path(__file__).resolve().parents[1] / "gpuwm" / "gui" / "static" / "js" / "geo.js"

SCRIPT = r"""
import { grids, placesAt, gridForPicture, borrowGeoref, placement } from "./geo.mjs";
const projection = {map_proj: "mercator", ref_lat: 17.101911979269417, ref_lon: -105.05708127563236,
  truelat1: 17.1, truelat2: 17.1, stand_lon: -105.05708127563236};
const domains = [
  {grid_id: 1, parent_id: 0, nx: 200, ny: 160, dx_km: 12, i_parent_start: 1, j_parent_start: 1, parent_grid_ratio: 1},
  {grid_id: 2, parent_id: 1, nx: 160, ny: 160, dx_km: 3, i_parent_start: 81, j_parent_start: 61, parent_grid_ratio: 4}];
const moves = [
  {valid: "2026-09-24T18:45:00Z", grid_id: 2, i_parent_start: 79, j_parent_start: 61},
  {valid: "2026-09-24T19:45:00Z", grid_id: 2, i_parent_start: 77, j_parent_start: 60},
  {valid: "2026-09-24T21:30:00Z", grid_id: 2, i_parent_start: 75, j_parent_start: 60},
  {valid: "2026-09-24T23:15:00Z", grid_id: 2, i_parent_start: 73, j_parent_start: 61}];
// the renderer's record of the 23:30 frame
const georef = {image_width_px: 1200, image_height_px: 900, plot_rect_px: {x: 27, y: 64, width: 1087, height: 818},
  projection: {kind: "mercator", latitude_of_true_scale_deg: 17.100000381469727, central_meridian_deg: -105.0570831298828},
  extent: {x_min: -428431.55789630796, x_max: 236432.05477202, y_min: 1594796.2596194167, y_max: 2095126.0067792071},
  geographic_bounds: [-108.25494079589843, -103.666064453125, 14.894820404052734, 19.28430633544922]};
const west = (g) => g.toLatLon(0, 0)[1];
const at = (valid) => grids(projection, domains, placesAt(moves, domains, valid)).get(2);
const plan = grids(projection, domains).get(2);
const right = (p) => Math.max(...p.verts.map((v) => v.u));
const out = {
  west_at: Object.fromEntries(["2026-09-24T18:45:00Z", "2026-09-24T19:00:00Z", "2026-09-24T23:30:00Z"]
    .map((v) => [v, west(at(v))])),
  plan_west: west(plan),
  on_record_same_key: gridForPicture(at("2026-09-24T23:30:00Z"), georef).key === at("2026-09-24T23:30:00Z").key,
  snapped_west: west(gridForPicture(plan, georef)),
  right_on_record: right(placement(georef, at("2026-09-24T23:30:00Z"))),
  right_snapped: right(placement(georef, gridForPicture(plan, georef))),
  right_plan: right(placement(georef, plan)),
};
const early = at("2026-09-24T18:00:00Z");
const lent = borrowGeoref(georef, at("2026-09-24T23:30:00Z"), early);
out.borrowed_right = right(placement(lent, early));
out.borrowed_west = lent.geographic_bounds[0];
// a downscale: the parent grid (from another run) only places the child, and is not drawn
const child = grids(projection, [{...domains[0], context: true},
  {grid_id: 2, parent_id: 1, nx: 288, ny: 288, dx_km: 4, i_parent_start: 54, j_parent_start: 34, parent_grid_ratio: 3}]);
out.downscale_ids = [...child.keys()];
out.downscale_west = child.get(2).toLatLon(0, 0)[1];
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def geo(tmp_path_factory):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    folder = tmp_path_factory.mktemp("geo")
    shutil.copyfile(GEO, folder / "geo.mjs")
    (folder / "t.mjs").write_text(SCRIPT, encoding="utf-8")
    done = subprocess.run([node, "t.mjs"], cwd=folder, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_frame_is_cut_along_the_place_the_nest_stood_when_it_was_written(geo):
    # the wrfout corners of this run: -107.3015 until the first move, which a frame at the move's own
    # time was written before; -107.5274 from the next frame; -108.2049 after all four moves
    assert geo["west_at"]["2026-09-24T18:45:00Z"] == pytest.approx(-107.3015, abs=2e-3)
    assert geo["west_at"]["2026-09-24T19:00:00Z"] == pytest.approx(-107.5274, abs=2e-3)
    assert geo["west_at"]["2026-09-24T23:30:00Z"] == pytest.approx(-108.2049, abs=2e-3)
    assert geo["plan_west"] == pytest.approx(-107.3015, abs=2e-3)


def test_a_moved_nest_never_brings_the_colour_bar_strip_into_the_map(geo):
    # The picture's map ends near pixel 957; right of it is the renderer's margin, then its colour bar.
    # Cut along the plan's first place, the picture reaches the plot rectangle's right edge (1114).
    assert geo["right_plan"] > 1100
    assert geo["right_on_record"] < 970
    # With no moves on record, the grid is moved to the renderer's own bounds and cut there instead.
    assert geo["snapped_west"] == pytest.approx(-108.2049, abs=2e-2)
    assert geo["right_snapped"] < 975
    assert geo["on_record_same_key"]


def test_a_frame_left_out_of_the_record_borrows_a_neighbours_moved_with_the_nest(geo):
    assert geo["borrowed_right"] < 970
    assert geo["borrowed_west"] == pytest.approx(-108.2549 + (-107.3015 + 108.2049), abs=2e-2)


def test_a_downscale_is_placed_by_its_parent_and_only_the_child_is_drawn(geo):
    # the plan's outline: the child covers parent cells 54..149, whose first mass point is at -110.308
    assert geo["downscale_ids"] == [2]
    assert geo["downscale_west"] == pytest.approx(-110.308 - 12 / 3 / 111.0, abs=0.05)


# A domain across the 180th meridian.  The page used to subtract longitudes as they came (170 and -170), so a
# 20 degree Pacific footprint was bent onto a 359 degree box and framed as the whole world.
DATELINE = r"""
import { placement, toWorld, grids, gridOutline, gridForPicture, lonLatBox } from "./geo.mjs";
const spanDeg = (pts) => { const xs = pts.map(([lon, lat]) => toWorld(lon, lat)[0]); return (Math.max(...xs) - Math.min(...xs)) * 360; };
const out = {};
// a geographic picture from 170E to 170W with no grid on record: the field's bent footprint
const geographic = {image_width_px: 1200, image_height_px: 900, plot_rect_px: {x: 20, y: 60, width: 1000, height: 800},
  projection: {kind: "geographic", central_meridian_deg: 180}, extent: {x_min: -10, x_max: 10, y_min: 10, y_max: 30}};
const p = placement(geographic, null);
out.geographic_field = spanDeg(p.verts.map((v) => [v.lon, v.lat]));
// a 960 km Lambert grid centred at 17S 179E, and a nest on its east side across the meridian
const projection = {map_proj: "lambert", ref_lat: -17, ref_lon: 179, truelat1: -10, truelat2: -25, stand_lon: 179};
const doms = [{grid_id: 1, parent_id: 0, nx: 321, ny: 321, dx_km: 3, i_parent_start: 1, j_parent_start: 1, parent_grid_ratio: 1},
  {grid_id: 2, parent_id: 1, nx: 181, ny: 181, dx_km: 1, i_parent_start: 200, j_parent_start: 130, parent_grid_ratio: 3}];
const g = grids(projection, doms);
const outer = g.get(1);
const nest = g.get(2);
out.lambert_outline = spanDeg(gridOutline(outer, 8));
const everything = [...gridOutline(outer, 8), ...gridOutline(nest, 8)];
const box = lonLatBox(everything);
out.lambert_frame = box[2] - box[0];
out.nest_inside = gridOutline(nest, 8).every(([lon]) => lon >= box[0] && lon <= box[2]);
out.nest_east = Math.max(...gridOutline(nest, 8).map(([lon]) => lon));
// the renderer's Lambert picture of the outer grid, placed along the grid
const lam = {image_width_px: 1200, image_height_px: 1200, plot_rect_px: {x: 20, y: 20, width: 1100, height: 1100},
  projection: {kind: "lambert_conformal", standard_parallel_1_deg: -10, standard_parallel_2_deg: -25, central_meridian_deg: 179, reference_latitude_deg: -17},
  extent: {x_min: -480000, x_max: 480000, y_min: -480000 + outer.origin[1] + 480000, y_max: outer.origin[1] + 960000},
  geographic_bounds: [174.4, -176.4, -21.3, -12.6]};
out.lambert_field = spanDeg(placement(lam, outer).verts.map((v) => [v.lon, v.lat]));
// bounds written across the meridian agree with the grid, which is not moved
out.same_grid = gridForPicture(outer, lam, 40).key === outer.key;
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def dateline(tmp_path_factory):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    folder = tmp_path_factory.mktemp("dateline")
    shutil.copyfile(GEO, folder / "geo.mjs")
    (folder / "t.mjs").write_text(DATELINE, encoding="utf-8")
    done = subprocess.run([node, "t.mjs"], cwd=folder, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_picture_across_the_date_line_is_bent_onto_the_twenty_degrees_it_covers(dateline):
    assert dateline["geographic_field"] == pytest.approx(20, abs=0.5)
    assert dateline["lambert_field"] < 12


def test_a_grid_across_the_date_line_is_outlined_and_framed_where_it_is(dateline):
    # 960 km at 17S is about 9 degrees of longitude
    assert dateline["lambert_outline"] < 12
    assert dateline["lambert_frame"] < 12
    # the nest east of the meridian is read beside its parent (above 180), not at the far side of the world
    assert dateline["nest_inside"]
    assert dateline["nest_east"] > 180
    assert dateline["same_grid"]
