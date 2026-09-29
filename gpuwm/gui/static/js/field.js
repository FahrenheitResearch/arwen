// A forecast picture on the map. Every weather field on this page is a PNG the Rust renderer wrote; this
// module only places it: the picture is cut along the model grid's edge and bent from the renderer's
// projection onto the page's Web Mercator map, triangle by triangle, using the renderer's own record of where
// the picture sits (render-georef.json). No pixel's colour is read or changed.
//
//   const placed = await placePicture(url, georef, grid)   -> {canvas, world box, legend} or null
//   drawPlaced(ctx, map, placed, alpha)
//
// A bent picture is kept in a small cache, so scrubbing back and forth costs nothing after the first pass.

import { placement, toWorld, worldShifts } from "./geo.js";

// A bent picture is about as many pixels as the map part of the renderer's picture; 64 of them is a
// couple of hundred megabytes, enough to play a day of hourly frames on two grids without a reload.
const MAX_SIDE = 1600;
const CACHE_LIMIT = 64;
const cache = new Map();

export function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("A picture did not load."));
    img.src = url;
  });
}

// Affine map taking source triangle (s0, s1, s2) onto destination triangle (d0, d1, d2): [a, b, c, d, e, f]
// for ctx.setTransform, so that dest = [a c e; b d f] * [sx sy 1].
function affine(s0, s1, s2, d0, d1, d2) {
  const [x0, y0] = s0; const [x1, y1] = s1; const [x2, y2] = s2;
  const det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
  if (Math.abs(det) < 1e-9) return null;
  const solve = (u0, u1, u2) => {
    const p = ((u1 - u0) * (y2 - y0) - (u2 - u0) * (y1 - y0)) / det;
    const q = ((u2 - u0) * (x1 - x0) - (u1 - u0) * (x2 - x0)) / det;
    return [p, q, u0 - p * x0 - q * y0];
  };
  const [a, c, e] = solve(d0[0], d1[0], d2[0]);
  const [b, d, f] = solve(d0[1], d1[1], d2[1]);
  return [a, b, c, d, e, f];
}

// Grow a triangle a little about its centre, so neighbouring triangles overlap and no seam shows.
function grow(t, by) {
  const cx = (t[0][0] + t[1][0] + t[2][0]) / 3;
  const cy = (t[0][1] + t[1][1] + t[2][1]) / 3;
  return t.map(([x, y]) => {
    const dx = x - cx; const dy = y - cy; const len = Math.hypot(dx, dy) || 1;
    return [x + (dx / len) * by, y + (dy / len) * by];
  });
}

function bend(img, place) {
  const { n, verts } = place;
  // the picture's footprint in world units and the canvas that holds it bent
  let wx0 = Infinity; let wy0 = Infinity; let wx1 = -Infinity; let wy1 = -Infinity;
  const world = verts.map((p) => {
    const [x, y] = toWorld(p.lon, p.lat);
    wx0 = Math.min(wx0, x); wy0 = Math.min(wy0, y); wx1 = Math.max(wx1, x); wy1 = Math.max(wy1, y);
    return [x, y];
  });
  let srcW = 0;
  for (let k = 0; k < n; k++) srcW += Math.hypot(verts[k + 1].u - verts[k].u, verts[k + 1].v - verts[k].v);
  const aspect = (wy1 - wy0) / Math.max(1e-12, wx1 - wx0);
  let W = Math.round(Math.min(MAX_SIDE, Math.max(64, srcW * 1.15)));
  let H = Math.round(W * aspect);
  if (H > MAX_SIDE) { W = Math.round((W * MAX_SIDE) / H); H = MAX_SIDE; }
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, W);
  canvas.height = Math.max(1, H);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  const dst = world.map(([x, y]) => [((x - wx0) / (wx1 - wx0)) * W, ((y - wy0) / (wy1 - wy0)) * H]);
  const src = verts.map((p) => [p.u, p.v]);
  const at = (i, j) => j * (n + 1) + i;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const quads = [[at(i, j), at(i + 1, j), at(i + 1, j + 1)], [at(i, j), at(i + 1, j + 1), at(i, j + 1)]];
      for (const tri of quads) {
        const s = tri.map((k) => src[k]);
        const d = tri.map((k) => dst[k]);
        const m = affine(s[0], s[1], s[2], d[0], d[1], d[2]);
        if (!m) continue;
        const g = grow(d, 0.7);
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(g[0][0], g[0][1]); ctx.lineTo(g[1][0], g[1][1]); ctx.lineTo(g[2][0], g[2][1]);
        ctx.closePath();
        ctx.clip();
        ctx.setTransform(m[0], m[1], m[2], m[3], m[4], m[5]);
        // only the source triangle's box of the picture, a pixel wider on each side
        const sx = Math.max(0, Math.floor(Math.min(s[0][0], s[1][0], s[2][0])) - 1);
        const sy = Math.max(0, Math.floor(Math.min(s[0][1], s[1][1], s[2][1])) - 1);
        const ex = Math.min(img.naturalWidth, Math.ceil(Math.max(s[0][0], s[1][0], s[2][0])) + 1);
        const ey = Math.min(img.naturalHeight, Math.ceil(Math.max(s[0][1], s[1][1], s[2][1])) + 1);
        if (ex > sx && ey > sy) ctx.drawImage(img, sx, sy, ex - sx, ey - sy, sx, sy, ex - sx, ey - sy);
        ctx.restore();
      }
    }
  }
  // the outline of the grid in world units, for the edge line the map draws round it
  const edge = [];
  for (let i = 0; i <= n; i++) edge.push(world[at(i, 0)]);
  for (let j = 1; j <= n; j++) edge.push(world[at(n, j)]);
  for (let i = n - 1; i >= 0; i--) edge.push(world[at(i, n)]);
  for (let j = n - 1; j >= 1; j--) edge.push(world[at(0, j)]);
  return { canvas, box: [wx0, wy0, wx1, wy1], edge };
}

// The colour bar and its units: the strip of the picture right of the map frame, cut out as the renderer drew
// it. Returns a canvas, or null when the strip is too narrow to hold one.
function legendStrip(img, place) {
  // The strip holds the bar, its numbers and its units; a tenth of the picture's width is room for all
  // three, and anything wider is empty margin.
  const x0 = Math.ceil(place.frame.right) + 1;
  const w = Math.min(img.naturalWidth - x0, Math.round(img.naturalWidth * 0.1));
  if (w < 30) return null;
  const y0 = Math.max(0, Math.floor(place.frame.top) - 64);
  const y1 = Math.min(img.naturalHeight, Math.ceil(place.frame.bottom) + 30);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = y1 - y0;
  canvas.getContext("2d").drawImage(img, x0, y0, w, y1 - y0, 0, 0, w, y1 - y0);
  return canvas;
}

export async function placePicture(url, georef, grid) {
  const key = `${url}|${grid ? grid.key : "-"}|${georef && georef.extent ? Math.round(georef.extent.x_min) : "-"}`;
  if (cache.has(key)) {
    const hit = cache.get(key);
    cache.delete(key);
    cache.set(key, hit);
    return hit;
  }
  const img = await loadImage(url);
  const place = georef ? placement(georef, grid) : null;
  if (!place) {
    return { url, bent: null, legend: null };
  }
  const bent = bend(img, place);
  const out = { url, bent, legend: legendStrip(img, place) };
  cache.set(key, out);
  while (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value);
  return out;
}

// A picture is drawn on every copy of the world the view shows (worldShifts), as the basemap draws the land: one
// placed on longitudes west of the 180th meridian (-189 to -169) is then seen by a view framed east of it (171 to
// 191), one world width over, and a wide window zoomed right out shows it on each copy it shows the land.
export function drawPlaced(ctx, map, placed, alpha = 1) {
  if (!placed || !placed.bent) return;
  const { canvas, box } = placed.bent;
  // the view's left and right edges in world units
  const [origin] = map.worldToScreen(0, 0);
  const world = map.worldToScreen(1, 0)[0] - origin;
  if (!(world > 0)) return;
  ctx.globalAlpha = alpha;
  for (const k of worldShifts(box[0], box[2], -origin / world, (map.width - origin) / world)) {
    const [x0, y0] = map.worldToScreen(box[0] + k, box[1]);
    const [x1, y1] = map.worldToScreen(box[2] + k, box[3]);
    if (x1 < 0 || y1 < 0 || x0 > map.width || y0 > map.height) continue;
    ctx.imageSmoothingEnabled = (x1 - x0) < canvas.width * 1.6;
    ctx.drawImage(canvas, x0, y0, x1 - x0, y1 - y0);
  }
  ctx.globalAlpha = 1;
}

export function outlinePath(map, edge) {
  const path = new Path2D();
  edge.forEach(([wx, wy], k) => {
    const [x, y] = map.worldToScreen(wx, wy);
    if (k) path.lineTo(x, y); else path.moveTo(x, y);
  });
  path.closePath();
  return path;
}
