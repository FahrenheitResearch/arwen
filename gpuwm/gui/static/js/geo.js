// Map arithmetic: the page's Web Mercator world, the renderer's projections, and where a run's grids sit.
//
// Nothing here draws. The map (map.js) asks where a longitude and latitude land on screen; the field layer
// (field.js) asks where a renderer picture's pixels sit on the Earth. Both answers come from numbers the run
// wrote down: the projection and grids of its resolved plan, and the renderer's own record of each picture
// (render-georef.json: the plot rectangle, the projected extent it spans, the projection with every parameter).

const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;
// The Earth radius the renderer and the model grids both use.
const R_EARTH = 6370000;
export const MAX_LAT = 85.05112878;

// ---- Web Mercator: world coordinates 0..1, x east, y south

export function toWorld(lon, lat) {
  const phi = Math.max(-MAX_LAT, Math.min(MAX_LAT, lat)) * D2R;
  return [(lon + 180) / 360, (1 - Math.log(Math.tan(Math.PI / 4 + phi / 2)) / Math.PI) / 2];
}

export function fromWorld(x, y) {
  const lon = x * 360 - 180;
  const lat = Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * R2D;
  return [lon, lat];
}

export function wrapLon(lon) {
  let out = lon % 360;
  if (out > 180) out -= 360;
  else if (out <= -180) out += 360;
  return out;
}

// The copies of the world a map draws, as whole world widths k: every k that moves the span x0..x1 (world
// units, the world itself is 0..1) onto part of the view v0..v1. A view across the 180th meridian (a Pacific
// domain framed from 170 to 190 degrees) sees two copies, and a wide window zoomed right out sees three or
// four, so the land, the fields and the outlines are drawn on each one the view shows and nowhere else.
export function worldShifts(x0, x1, v0, v1) {
  if (![x0, x1, v0, v1].every(Number.isFinite)) return [0];
  const out = [];
  // at most 16: the map's widest view is a few worlds, and a runaway span must not become a runaway loop
  for (let k = Math.floor(v0 - x1) + 1; k < v1 - x0 && out.length < 16; k++) out.push(k);
  return out;
}

// A longitude moved by whole turns to within half a turn of `ref`. A domain across the 180th meridian keeps
// its longitudes in one run (170 to 190, not 170 and -170), so its box is 20 degrees wide and not 340.
export function unwrapLon(lon, ref) {
  return ref + wrapLon(lon - ref);
}

// ---- projections, in the renderer's own formulas (rustwx-render projection.rs)

function refLat(value) {
  const c = Math.max(-85, Math.min(85, value));
  return Math.abs(c) < 1 ? (c < 0 ? -10 : 10) : c;
}

const stableLat = (lat) => Math.max(-89.999, Math.min(89.999, lat));

function lambert(t1, t2, lon0, lat0) {
  const p1 = refLat(t1) * D2R;
  const p2 = refLat(t2) * D2R;
  const p0 = refLat(lat0) * D2R;
  let n = Math.abs(t1 - t2) < 1e-10 ? Math.sin(p1)
    : (Math.log(Math.cos(p1)) - Math.log(Math.cos(p2)))
      / (Math.log(Math.tan(Math.PI / 4 + p2 / 2)) - Math.log(Math.tan(Math.PI / 4 + p1 / 2)));
  if (Math.abs(n) < 1e-8) n = Math.sin(Math.abs(p0) >= 1e-8 ? p0 : Math.abs(p1) >= 1e-8 ? p1 : 10 * D2R);
  const f = (Math.cos(p1) * Math.pow(Math.tan(Math.PI / 4 + p1 / 2), n)) / n;
  const rho0 = (R_EARTH * f) / Math.pow(Math.tan(Math.PI / 4 + p0 / 2), n);
  return {
    project(lat, lon) {
      const rho = (R_EARTH * f) / Math.pow(Math.tan(Math.PI / 4 + (stableLat(lat) * D2R) / 2), n);
      const theta = n * wrapLon(lon - lon0) * D2R;
      return [rho * Math.sin(theta), rho0 - rho * Math.cos(theta)];
    },
    unproject(x, y) {
      const sign = Math.sign(n) || 1;
      const rho = Math.hypot(x, rho0 - y) * sign;
      if (!Number.isFinite(rho) || rho === 0) return null;
      const theta = Math.atan2(x * sign, (rho0 - y) * sign);
      const ratio = (R_EARTH * f) / rho;
      if (!(ratio > 0)) return null;
      const phi = 2 * Math.atan(Math.pow(ratio, 1 / n)) - Math.PI / 2;
      return [phi * R2D, wrapLon(lon0 + (theta / n) * R2D)];
    },
  };
}

function mercator(latTs, lon0) {
  const scale = Math.max(1e-6, Math.cos(refLat(latTs) * D2R));
  return {
    project(lat, lon) {
      const phi = stableLat(lat) * D2R;
      return [R_EARTH * scale * wrapLon(lon - lon0) * D2R, R_EARTH * scale * Math.log(Math.tan(Math.PI / 4 + phi / 2))];
    },
    unproject(x, y) {
      const lon = lon0 + (x / (R_EARTH * scale)) * R2D;
      const lat = (2 * Math.atan(Math.exp(y / (R_EARTH * scale))) - Math.PI / 2) * R2D;
      return [stableLat(lat), wrapLon(lon)];
    },
  };
}

function polar(trueLat, lon0, south) {
  const k = (1 + Math.sin(refLat(trueLat) * D2R)) / 2;
  return {
    project(lat, lon) {
      const phi = stableLat(lat) * D2R;
      const theta = wrapLon(lon - lon0) * D2R;
      if (south) {
        const rho = 2 * R_EARTH * k * Math.tan(Math.PI / 4 + phi / 2);
        return [rho * Math.sin(theta), rho * Math.cos(theta)];
      }
      const rho = 2 * R_EARTH * k * Math.tan(Math.PI / 4 - phi / 2);
      return [rho * Math.sin(theta), -rho * Math.cos(theta)];
    },
    unproject(x, y) {
      const rho = Math.hypot(x, y);
      if (south) {
        const phi = 2 * Math.atan(rho / (2 * R_EARTH * k)) - Math.PI / 2;
        return [phi * R2D, wrapLon(lon0 + Math.atan2(x, y) * R2D)];
      }
      const phi = Math.PI / 2 - 2 * Math.atan(rho / (2 * R_EARTH * k));
      return [phi * R2D, wrapLon(lon0 + Math.atan2(x, -y) * R2D)];
    },
  };
}

function geographic(lon0) {
  return {
    project: (lat, lon) => [wrapLon(lon - lon0), stableLat(lat)],
    unproject: (x, y) => [stableLat(y), wrapLon(x + lon0)],
  };
}

// The renderer's resolved projection (a georeference's "projection"), or null for one the page cannot invert.
export function rendererProjection(spec) {
  if (!spec) return null;
  switch (spec.kind) {
    case "lambert_conformal":
      return lambert(spec.standard_parallel_1_deg, spec.standard_parallel_2_deg, spec.central_meridian_deg,
        spec.reference_latitude_deg);
    case "mercator":
      return mercator(spec.latitude_of_true_scale_deg, spec.central_meridian_deg);
    case "polar_stereographic":
      return polar(spec.true_latitude_deg, spec.central_meridian_deg, !!spec.south_pole_on_projection_plane);
    case "geographic":
      return geographic(spec.central_meridian_deg || 0);
    default:
      return null;
  }
}

// A model grid's projection (the resolved plan's "projection": map_proj, truelat1/2, stand_lon, ref_lat/lon).
export function gridProjection(p) {
  if (!p) return null;
  const kind = String(p.map_proj || "lambert").toLowerCase();
  const t1 = Number(p.truelat1 ?? 30);
  const t2 = Number(p.truelat2 ?? t1);
  const lon0 = Number(p.stand_lon ?? p.ref_lon ?? 0);
  // each projection carries its parameters, so a renderer record can be checked against it
  const tag = (proj, name) => Object.assign(proj, { spec: { kind: name, t1, t2, lon0 } });
  if (kind === "lambert" || kind === "1") return tag(lambert(t1, t2, lon0, Number(p.ref_lat ?? t1)), "lambert");
  if (kind === "mercator" || kind === "3") return tag(mercator(t1, lon0), "mercator");
  if (kind === "polar" || kind === "2") return tag(polar(t1, lon0, Number(p.ref_lat ?? 1) < 0), "polar");
  return null;
}

// ---- where a run's grids sit
//
// grids(projection, domains, places) -> Map(grid_id -> grid) with grid.toLatLon(i, j) for 0-based mass-point
// indices, grid.nx, grid.ny, grid.dx_km. The outer grid is centred on (ref_lat, ref_lon); a nest's first mass
// point is placed from its parent's i_parent_start, j_parent_start and ratio, as the model places it. `places`
// (optional, Map grid_id -> [i_parent_start, j_parent_start]) overrides the plan's start for a nest that has
// moved since (placesAt below). grid.key names the footprint, so a picture bent onto one place is never
// reused for another.

export function grids(projection, domains, places = null) {
  const out = new Map();
  const proj = gridProjection(projection);
  if (!proj || !domains || !domains.length) return out;
  const [cx, cy] = proj.project(Number(projection.ref_lat), Number(projection.ref_lon));
  // every grid of the run reads its longitudes near the outer grid's centre, so a nest across the 180th
  // meridian from its parent's centre still sits beside it on the map
  const lonRef = wrapLon(Number(projection.ref_lon) || 0);
  const byId = new Map(domains.map((d) => [d.grid_id, d]));
  // plane(i, j) of each grid: its mass points in the outer grid's projection plane, metres
  const planes = new Map();
  function plane(d) {
    if (planes.has(d.grid_id)) return planes.get(d.grid_id);
    const dx = Number(d.dx_km) * 1000;
    let fn;
    const parent = byId.get(d.parent_id);
    if (!parent || d.parent_id === d.grid_id || !d.parent_id) {
      const ic = (Number(d.nx) - 1) / 2;
      const jc = (Number(d.ny) - 1) / 2;
      fn = (i, j) => [cx + (i - ic) * dx, cy + (j - jc) * dx];
    } else {
      const up = plane(parent);
      const r = Number(d.parent_grid_ratio) || 1;
      const at = places && places.get(d.grid_id);
      const i0 = Number(at ? at[0] : d.i_parent_start) - 1.5 + 0.5 / r;
      const j0 = Number(at ? at[1] : d.j_parent_start) - 1.5 + 0.5 / r;
      const [ax, ay] = up(i0, j0);
      fn = (i, j) => [ax + i * dx, ay + j * dx];
    }
    planes.set(d.grid_id, fn);
    return fn;
  }
  for (const d of domains) {
    // a context grid only places the grids inside it (a downscale's parent, from another run): not this run's
    if (d.context || !(d.nx > 1 && d.ny > 1 && d.dx_km > 0)) continue;
    out.set(d.grid_id, makeGrid(proj, d, plane(d), lonRef));
  }
  return out;
}

function makeGrid(proj, d, fn, lonRef = null) {
  const dxm = Number(d.dx_km) * 1000;
  const [ox, oy] = fn(0, 0);
  return {
    id: d.grid_id, parent: d.parent_id, nx: Number(d.nx), ny: Number(d.ny), dx_km: Number(d.dx_km), proj, lonRef,
    // the first mass point in the plane, to the metre: two grids with one key sit in one place
    key: `${d.grid_id}:${d.nx}x${d.ny}@${Math.round(ox)},${Math.round(oy)}`,
    origin: [ox, oy],
    toLatLon: (i, j) => { const [x, y] = fn(i, j); return proj.unproject(x, y); },
    // (lat, lon) -> fractional mass-point indices, for the hover readout
    toIndex: (lat, lon) => {
      const [x, y] = proj.project(lat, lon);
      return [(x - ox) / dxm, (y - oy) / dxm];
    },
    // the same grid moved by (mx, my) metres in its plane
    moved: (mx, my) => makeGrid(proj, d, (i, j) => { const [x, y] = fn(i, j); return [x + mx, y + my]; }, lonRef),
  };
}

// Where each moving nest stood when the frame valid at `valid` (ISO, UTC) was written: Map(grid_id -> [i, j]),
// from the run's nest moves (GET /api/runs/RUN/map "moves"). A frame written at a move's own time was written
// before the move, so only moves strictly earlier count. A parent's slide under a mover that stays put on the
// Earth takes the mover's start back by the slide times the parent's ratio.
export function placesAt(moves, domains, valid) {
  const places = new Map();
  if (!moves || !moves.length || !valid) return places;
  const byId = new Map((domains || []).map((d) => [d.grid_id, d]));
  const start = (id) => {
    if (places.has(id)) return places.get(id);
    const d = byId.get(id);
    return d ? [Number(d.i_parent_start), Number(d.j_parent_start)] : null;
  };
  for (const m of moves) {
    if (!(m.valid < valid)) continue;
    if (m.shift_by !== undefined) {
      const was = start(m.grid_id);
      const parent = byId.get(m.shift_by);
      const r = Number(parent && parent.parent_grid_ratio) || 1;
      if (was && m.shift) places.set(m.grid_id, [was[0] - m.shift[0] * r, was[1] - m.shift[1] * r]);
    } else {
      places.set(m.grid_id, [Number(m.i_parent_start), Number(m.j_parent_start)]);
    }
  }
  return places;
}

// The picture and the grid agree when the grid's middle is the middle of the picture's geographic bounds to
// within `cells` grid cells. Returns the grid when they agree, and otherwise the grid moved in its plane, in
// whole cells, to the picture's bounds: a nest whose moves are not on record is drawn where the renderer drew
// it, never cut from a place it no longer was.
export function gridForPicture(grid, georef, cells = 2) {
  const b = georef && georef.geographic_bounds;
  if (!grid || !b || b.length !== 4 || !b.every(Number.isFinite)) return grid;
  let w = Infinity; let e = -Infinity; let s = Infinity; let n = -Infinity;
  for (const [lon, lat] of gridOutline(grid, 8)) { w = Math.min(w, lon); e = Math.max(e, lon); s = Math.min(s, lat); n = Math.max(n, lat); }
  const [gx, gy] = grid.proj.project((s + n) / 2, (w + e) / 2);
  // bounds written across the 180th meridian run east from west through it (170, -170 is 20 degrees wide)
  const [west, east] = boundsRun(b);
  const [px, py] = grid.proj.project((b[2] + b[3]) / 2, (west + east) / 2);
  const dxm = grid.dx_km * 1000;
  if (Math.hypot(px - gx, py - gy) / dxm <= cells) return grid;
  return grid.moved(Math.round((px - gx) / dxm) * dxm, Math.round((py - gy) / dxm) * dxm);
}

// A picture the renderer's record leaves out borrows the record of another frame of the same grid and look.
// When the grid moved between the two frames, the borrowed record moves with it: the renderer draws a grid in
// the grid's own projection, so a nest's move is the same shift of the picture's extent in metres. Returns the
// record to use, or null when the renderer's projection is not the grid's and the shift cannot be carried over.
export function borrowGeoref(georef, fromGrid, toGrid) {
  if (!georef) return null;
  if (!fromGrid || !toGrid || fromGrid.key === toGrid.key) return georef;
  if (!sameProjection(georef.projection, fromGrid.proj.spec) || !georef.extent) return null;
  const mx = toGrid.origin[0] - fromGrid.origin[0];
  const my = toGrid.origin[1] - fromGrid.origin[1];
  const ext = georef.extent;
  const moved = { ...georef, extent: { x_min: ext.x_min + mx, x_max: ext.x_max + mx, y_min: ext.y_min + my, y_max: ext.y_max + my } };
  const b = georef.geographic_bounds;
  if (b && b.length === 4) {
    const [fromLat, fromLon] = fromGrid.toLatLon((fromGrid.nx - 1) / 2, (fromGrid.ny - 1) / 2);
    const [toLat, toLon] = toGrid.toLatLon((toGrid.nx - 1) / 2, (toGrid.ny - 1) / 2);
    const dLon = wrapLon(toLon - fromLon);
    moved.geographic_bounds = [b[0] + dLon, b[1] + dLon, b[2] + toLat - fromLat, b[3] + toLat - fromLat];
  }
  return moved;
}

const near = (a, b) => Math.abs(Number(a) - Number(b)) < 1e-3;

function sameProjection(r, g) {
  if (!r || !g) return false;
  if (g.kind === "lambert" && r.kind === "lambert_conformal") {
    const pair = (near(r.standard_parallel_1_deg, g.t1) && near(r.standard_parallel_2_deg, g.t2))
      || (near(r.standard_parallel_1_deg, g.t2) && near(r.standard_parallel_2_deg, g.t1));
    return pair && near(r.central_meridian_deg, g.lon0);
  }
  if (g.kind === "mercator" && r.kind === "mercator") {
    return near(r.latitude_of_true_scale_deg, g.t1) && near(r.central_meridian_deg, g.lon0);
  }
  if (g.kind === "polar" && r.kind === "polar_stereographic") {
    return near(r.true_latitude_deg, g.t1) && near(r.central_meridian_deg, g.lon0);
  }
  return false;
}

// The longitude a grid's points are read near: the run's outer grid centre when the grid carries one, else
// the grid's own middle.
function gridLonRef(grid) {
  if (grid && Number.isFinite(grid.lonRef)) return grid.lonRef;
  return grid.toLatLon((grid.nx - 1) / 2, (grid.ny - 1) / 2)[1];
}

// The box that frames a set of lon/lat points, the longitudes read in one run near the first point's (or
// `ref`), so a domain across the 180th meridian is framed as the 20 degrees it covers.
export function lonLatBox(points, ref = null) {
  let west = Infinity; let south = Infinity; let east = -Infinity; let north = -Infinity;
  for (const [lon0, lat] of points) {
    if (!Number.isFinite(lon0) || !Number.isFinite(lat)) continue;
    if (ref === null) ref = lon0;
    const lon = unwrapLon(lon0, ref);
    west = Math.min(west, lon); east = Math.max(east, lon); south = Math.min(south, lat); north = Math.max(north, lat);
  }
  return Number.isFinite(west) ? [west, south, east, north] : null;
}

// The west and east edges of a renderer record's geographic_bounds as one run of longitudes, east never west
// of west, or null. The renderer writes a box across the 180th meridian with its west edge east of its east
// edge (170, -170 is the 20 degrees from 170 to 190) and a whole world as a span of 359 degrees or more,
// either way round (rustwx-products longitude_bounds_span_deg).
export function boundsRun(b) {
  if (!b || b.length !== 4 || !Number.isFinite(b[0]) || !Number.isFinite(b[1])) return null;
  const span = Math.abs(b[1] - b[0]);
  if (span >= 359) {
    const west = Math.min(b[0], b[1]);
    return [west, west + Math.min(span, 360)];
  }
  const west = wrapLon(b[0]);
  let east = wrapLon(b[1]);
  if (east < west) east += 360;
  return [west, east];
}

// The box that frames the renderer records' geographic bounds: each record's run of longitudes (boundsRun),
// moved by whole turns beside the first record's, so two pictures on either side of the 180th meridian are
// framed as the few degrees they cover together.
export function boundsBox(georefs) {
  let west = Infinity; let south = Infinity; let east = -Infinity; let north = -Infinity;
  let ref = null;
  for (const g of georefs || []) {
    const b = g && g.geographic_bounds;
    const run = boundsRun(b);
    if (!run || !Number.isFinite(b[2]) || !Number.isFinite(b[3])) continue;
    const mid = (run[0] + run[1]) / 2;
    if (ref === null) ref = mid;
    const shift = unwrapLon(mid, ref) - mid;
    west = Math.min(west, run[0] + shift); east = Math.max(east, run[1] + shift);
    south = Math.min(south, b[2], b[3]); north = Math.max(north, b[2], b[3]);
  }
  return Number.isFinite(west) ? [west, south, east, north] : null;
}

// A line of lon/lat points with each longitude read beside the one before it, so an outline that crosses the
// 180th meridian stays one line over the few degrees it covers instead of jumping across the whole world.
export function unwrapRing(points) {
  let prev = null;
  return (points || []).map(([lon, lat]) => {
    const out = prev === null || !Number.isFinite(lon) ? lon : unwrapLon(lon, prev);
    if (Number.isFinite(out)) prev = out;
    return [out, lat];
  });
}

// A line of lon/lat points in the map's screen pixels, once for each world copy the view shows (worldShifts),
// with its longitudes put in one run first (unwrapRing): an outline across the 180th meridian is drawn as the
// few degrees it covers, on the same copies as the basemap and the fields. `map` gives screen(lon, lat) and width.
export function screenRings(points, map) {
  const run = unwrapRing(points);
  const here = run.map(([lon, lat]) => map.screen(lon, lat));
  let x0 = Infinity; let x1 = -Infinity;
  for (const [x] of here) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); }
  // one world width in screen pixels
  const world = map.screen(180, 0)[0] - map.screen(-180, 0)[0];
  if (!(world > 0) || !Number.isFinite(x0)) return [];
  return worldShifts(x0 / world, x1 / world, 0, map.width / world)
    .map((k) => (k ? run.map(([lon, lat]) => map.screen(lon + 360 * k, lat)) : here));
}

// The outline of a grid as lon/lat points, walking its edge (mass points) with `per` samples per side.
export function gridOutline(grid, per = 24) {
  const pts = [];
  const { nx, ny } = grid;
  const ref = gridLonRef(grid);
  const edge = (fx, fy) => { const [lat, lon] = grid.toLatLon(fx, fy); pts.push([unwrapLon(lon, ref), lat]); };
  for (let k = 0; k < per; k++) edge(((nx - 1) * k) / per, 0);
  for (let k = 0; k < per; k++) edge(nx - 1, ((ny - 1) * k) / per);
  for (let k = 0; k < per; k++) edge((nx - 1) * (1 - k / per), ny - 1);
  for (let k = 0; k < per; k++) edge(0, (ny - 1) * (1 - k / per));
  pts.push(pts[0]);
  return pts;
}

// "d02-3km" -> 2; "d01" -> 1; anything else -> null
export function domainNumber(token) {
  const m = /^d(\d+)/i.exec(String(token || ""));
  return m ? Number(m[1]) : null;
}

// ---- a renderer picture on the Earth
//
// placement(georef, grid) -> {mesh, legend} or null. mesh is a list of vertices on a regular (n+1) x (n+1)
// lattice over the grid's mass points: each {u, v} (the picture's pixel) and {lon, lat}. The lattice is inset
// a few picture pixels so the frame line the renderer draws round its map stays out.

export function placement(georef, grid, n = 16) {
  const proj = rendererProjection(georef && georef.projection);
  if (!proj || !georef.plot_rect_px || !georef.extent) return null;
  const rect = georef.plot_rect_px;
  const ext = georef.extent;
  const w = Math.max(1, rect.width - 1);
  const hgt = Math.max(1, rect.height - 1);
  // A geographic picture's extent is in degrees, and may run on past 180 (170 to 190): a grid longitude is read
  // in that run, so a point at -175 lands at 185, inside the picture, and not 355 degrees to its west.
  const extMid = (ext.x_min + ext.x_max) / 2;
  const degrees = georef.projection.kind === "geographic";
  const toPixel = (lat, lon) => {
    let [x, y] = proj.project(lat, lon);
    if (degrees) x = unwrapLon(x, extMid);
    const rx = (x - ext.x_min) / (ext.x_max - ext.x_min);
    const ry = 1 - (y - ext.y_min) / (ext.y_max - ext.y_min);
    return [rect.x + rx * w, rect.y + ry * hgt];
  };
  const fromPixel = (px, py) => {
    const x = ext.x_min + ((px - rect.x) / w) * (ext.x_max - ext.x_min);
    const y = ext.y_min + (1 - (py - rect.y) / hgt) * (ext.y_max - ext.y_min);
    return proj.unproject(x, y);
  };
  let sample;
  if (grid) {
    // picture pixels per grid cell, to turn a pixel inset into grid cells
    const [a0, a1] = grid.toLatLon(0, 0);
    const [b0, b1] = grid.toLatLon(grid.nx - 1, 0);
    const [pa, pb] = [toPixel(a0, a1), toPixel(b0, b1)];
    const perCell = Math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / (grid.nx - 1);
    const inset = Math.min(0.45, 2.5 / Math.max(perCell, 1e-6));
    const i0 = inset;
    const i1 = grid.nx - 1 - inset;
    const j0 = inset;
    const j1 = grid.ny - 1 - inset;
    sample = (s, t) => {
      const [lat, lon] = grid.toLatLon(i0 + (i1 - i0) * s, j0 + (j1 - j0) * t);
      const [u, v] = toPixel(lat, lon);
      return { u, v, lat, lon };
    };
  } else {
    // no grid on record: the whole plot rectangle, which is the map and the margin the renderer fits it in
    sample = (s, t) => {
      const u = rect.x + 1 + (rect.width - 3) * s;
      const v = rect.y + 1 + (rect.height - 3) * (1 - t);
      const ll = fromPixel(u, v);
      return ll ? { u, v, lat: ll[0], lon: ll[1] } : null;
    };
  }
  const verts = [];
  for (let jj = 0; jj <= n; jj++) {
    for (let ii = 0; ii <= n; ii++) {
      const p = sample(ii / n, jj / n);
      if (!p || !Number.isFinite(p.u) || !Number.isFinite(p.lat)) return null;
      verts.push(p);
    }
  }
  // One run of longitudes across the picture, so a picture across the 180th meridian is bent onto the 20
  // degrees it covers and not stretched round the world. The run is the one the map frames the picture in: near
  // the grid's reference when there is a grid (its outline), else the middle of the renderer's geographic bounds
  // read east from west (boundsBox), else the picture's own middle.
  const run = grid ? null : boundsRun(georef.geographic_bounds);
  const ref = grid ? gridLonRef(grid) : run ? (run[0] + run[1]) / 2 : verts[Math.floor(verts.length / 2)].lon;
  for (const p of verts) p.lon = unwrapLon(p.lon, ref);
  // The picture's frame in pixels: where the colour bar strip beside it starts.
  let right = -Infinity;
  let top = Infinity;
  let bottom = -Infinity;
  for (const p of verts) { right = Math.max(right, p.u); top = Math.min(top, p.v); bottom = Math.max(bottom, p.v); }
  return { n, verts, frame: { right, top, bottom }, image: { width: georef.image_width_px, height: georef.image_height_px } };
}
