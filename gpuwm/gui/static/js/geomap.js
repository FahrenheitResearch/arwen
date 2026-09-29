// The forecast map: one canvas in Web Mercator, panned by dragging and zoomed by the wheel, with a basemap
// drawn from the page's own vector file (static/map/basemap.bin: Natural Earth and US counties). Nothing is
// fetched from outside this computer.
//
// Layers are objects with draw(ctx, map); the map calls them in order after the land and water and before
// the lines, or after the lines when a layer says {above: true}.
//
//   const map = new MapView(el);
//   map.fit(west, south, east, north)      frame a box
//   map.screen(lon, lat) -> [x, y]         CSS pixels
//   map.lonlat(x, y)     -> [lon, lat]
//   map.on("hover", fn(lon, lat, ev))      also "click", "leave", "view"
//   map.drawBox(fn(box))                   the next drag draws a box instead of panning

import { toWorld, fromWorld, MAX_LAT, worldShifts } from "./geo.js";

const TILE = 256;
const MIN_Z = 1.2;
const MAX_Z = 13;
const DETAIL_Z = 3.6;
const COUNTY_Z = 6.2;

// The map's colours are the page's map tokens (--aw-map-*, shared with every other map), read when the map
// is made; the values here only stand in when a token is missing, and are the same light ones.
const STYLE = {
  water: "#edf1f5",
  land: "#fafafa",
  lake: "#edf1f5",
  coast: "#a3a3a3",
  borders: "#b4b4b4",
  states: "#d9d9d9",
  counties: "#e8e8e8",
  graticule: "#efefef",
  box: "#2563eb",
};

function readStyle() {
  const css = getComputedStyle(document.documentElement);
  const pick = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
  STYLE.water = pick("--aw-map-water", STYLE.water);
  STYLE.lake = STYLE.water;
  STYLE.land = pick("--aw-map-land", STYLE.land);
  STYLE.coast = pick("--aw-map-coast", STYLE.coast);
  STYLE.borders = pick("--aw-map-border", STYLE.borders);
  STYLE.states = pick("--aw-map-state", STYLE.states);
  STYLE.counties = pick("--aw-map-county", STYLE.counties);
  STYLE.graticule = pick("--aw-map-grid", STYLE.graticule);
  STYLE.box = pick("--aw-map-ask", STYLE.box);
}

// ---- the basemap file

export async function loadBasemap(url) {
  const response = await fetch(url, { credentials: "same-origin" });
  if (!response.ok) throw new Error(`The map outlines did not load (${response.status}).`);
  const buf = await response.arrayBuffer();
  const view = new DataView(buf);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== "AWBM") throw new Error("The map outline file is not one this page reads.");
  const headerLen = view.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, headerLen)));
  const base = 12 + headerLen;
  const layers = [];
  for (const layer of header.layers) {
    const at = base + layer.offset;
    const counts = new Uint32Array(buf, at, layer.parts);
    const starts = new Int32Array(buf, at + 4 * layer.parts, 2 * layer.parts);
    const deltaAt = at + 12 * layer.parts;
    const deltas = new Int16Array(buf, deltaAt, 2 * (layer.points - layer.parts));
    const q = layer.quantum;
    // world coordinates (0..1) of every part, a box per part, and one Path2D for the zoomed-out view
    const xy = new Float64Array(2 * layer.points);
    const box = new Float64Array(4 * layer.parts);
    const first = new Uint32Array(layer.parts + 1);
    const path = new Path2D();
    let d = 0;
    let at2 = 0;
    for (let p = 0; p < layer.parts; p++) {
      first[p] = at2 / 2;
      let x = starts[2 * p];
      let y = starts[2 * p + 1];
      let x0 = Infinity; let y0 = Infinity; let x1 = -Infinity; let y1 = -Infinity;
      for (let k = 0; k < counts[p]; k++) {
        if (k) { x += deltas[d++]; y += deltas[d++]; }
        const [wx, wy] = toWorld(x * q, y * q);
        xy[at2++] = wx; xy[at2++] = wy;
        if (wx < x0) x0 = wx; if (wx > x1) x1 = wx; if (wy < y0) y0 = wy; if (wy > y1) y1 = wy;
        if (k) path.lineTo(wx, wy); else path.moveTo(wx, wy);
      }
      if (layer.kind === "poly") path.closePath();
      box.set([x0, y0, x1, y1], 4 * p);
    }
    first[layer.parts] = at2 / 2;
    layers.push({ name: layer.name, kind: layer.kind, level: layer.level, path, xy, box, first, parts: layer.parts });
  }
  return layers;
}

// ---- drawing a layer when zoomed in
//
// The canvas refuses a path whose points lie tens of thousands of pixels away, and a coastline zoomed in
// has most of its points there. So past WORLD_PATH_PX the visible part of each ring is cut out against the
// window (a little wider) in world units, and only that is drawn, in screen units.

const WORLD_PATH_PX = 12000;

function clipRing(xy, a, b, r) {
  // Sutherland and Hodgman against the four sides of r = [x0, y0, x1, y1]
  let pts = [];
  for (let k = a; k < b; k++) pts.push(xy[2 * k], xy[2 * k + 1]);
  const sides = [[0, r[0], 1], [0, r[2], -1], [1, r[1], 1], [1, r[3], -1]];
  for (const [axis, edge, sign] of sides) {
    const out = [];
    const n = pts.length / 2;
    if (!n) break;
    for (let k = 0; k < n; k++) {
      const cx = pts[2 * k]; const cy = pts[2 * k + 1];
      const pk = (k + n - 1) % n;
      const px = pts[2 * pk]; const py = pts[2 * pk + 1];
      const cIn = sign * ((axis ? cy : cx) - edge) >= 0;
      const pIn = sign * ((axis ? py : px) - edge) >= 0;
      if (cIn !== pIn) {
        const pa = axis ? py : px; const ca = axis ? cy : cx;
        const t = (edge - pa) / (ca - pa);
        out.push(px + (cx - px) * t, py + (cy - py) * t);
      }
      if (cIn) out.push(cx, cy);
    }
    pts = out;
  }
  return pts;
}

function clipSegment(ax, ay, bx, by, r) {
  // Liang and Barsky: the part of a segment inside r, or null
  let t0 = 0; let t1 = 1;
  const dx = bx - ax; const dy = by - ay;
  const edges = [[-dx, ax - r[0]], [dx, r[2] - ax], [-dy, ay - r[1]], [dy, r[3] - ay]];
  for (const [p, q] of edges) {
    if (p === 0) { if (q < 0) return null; continue; }
    const t = q / p;
    if (p < 0) { if (t > t1) return null; if (t > t0) t0 = t; } else { if (t < t0) return null; if (t < t1) t1 = t; }
  }
  return [ax + dx * t0, ay + dy * t0, ax + dx * t1, ay + dy * t1, t0 > 0, t1 < 1];
}

function screenPath(layer, view, fill) {
  const { r, s, tx, ty } = view;
  const path = new Path2D();
  const { xy, box, first } = layer;
  for (let p = 0; p < layer.parts; p++) {
    if (box[4 * p] > r[2] || box[4 * p + 2] < r[0] || box[4 * p + 1] > r[3] || box[4 * p + 3] < r[1]) continue;
    const a = first[p]; const b = first[p + 1];
    if (fill) {
      const pts = clipRing(xy, a, b, r);
      if (pts.length < 6) continue;
      path.moveTo(pts[0] * s + tx, pts[1] * s + ty);
      for (let k = 2; k < pts.length; k += 2) path.lineTo(pts[k] * s + tx, pts[k + 1] * s + ty);
      path.closePath();
      continue;
    }
    let open = false;
    for (let k = a; k < b - 1; k++) {
      const seg = clipSegment(xy[2 * k], xy[2 * k + 1], xy[2 * k + 2], xy[2 * k + 3], r);
      if (!seg) { open = false; continue; }
      if (!open || seg[4]) path.moveTo(seg[0] * s + tx, seg[1] * s + ty);
      path.lineTo(seg[2] * s + tx, seg[3] * s + ty);
      open = !seg[5];
    }
  }
  return path;
}

// ---- the view

export class MapView {
  constructor(el) {
    readStyle();
    this.el = el;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "mapcanvas";
    el.append(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.cx = 0.25;
    this.cy = 0.37;
    this.z = 3;
    this.layers = [];
    this.basemap = null;
    this.handlers = { hover: [], click: [], leave: [], view: [] };
    this.boxHandler = null;
    this.pending = false;
    this.inset = { left: 0, right: 0, top: 0, bottom: 0 };
    this._bind();
    this.resize();
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(el);
  }

  // Stop watching the element's size: the page is leaving this map.
  close() { this.observer.disconnect(); }

  on(kind, fn) { this.handlers[kind].push(fn); return () => { this.handlers[kind] = this.handlers[kind].filter((f) => f !== fn); }; }
  emit(kind, ...args) { for (const fn of this.handlers[kind]) fn(...args); }

  get scale() { return TILE * Math.pow(2, this.z); }

  resize() {
    const r = this.el.getBoundingClientRect();
    this.width = Math.max(1, r.width);
    this.height = Math.max(1, r.height);
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.width * this.dpr);
    this.canvas.height = Math.round(this.height * this.dpr);
    this.redraw();
  }

  // world (0..1) -> CSS pixels
  worldToScreen(wx, wy) {
    const s = this.scale;
    return [(wx - this.cx) * s + this.width / 2, (wy - this.cy) * s + this.height / 2];
  }

  screen(lon, lat) { const [wx, wy] = toWorld(lon, lat); return this.worldToScreen(wx, wy); }

  lonlat(x, y) {
    const s = this.scale;
    return fromWorld(this.cx + (x - this.width / 2) / s, this.cy + (y - this.height / 2) / s);
  }

  // Frame a lon/lat box inside the part of the window the panels leave free.
  fit(west, south, east, north, pad = 40) {
    const [x0, y1] = toWorld(west, Math.max(south, -MAX_LAT));
    const [x1, y0] = toWorld(east, Math.min(north, MAX_LAT));
    const inset = this.inset;
    const w = Math.max(80, this.width - inset.left - inset.right - 2 * pad);
    const hgt = Math.max(80, this.height - inset.top - inset.bottom - 2 * pad);
    const z = Math.log2(Math.min(w / Math.max(1e-9, x1 - x0), hgt / Math.max(1e-9, y1 - y0)) / TILE);
    this.z = Math.max(MIN_Z, Math.min(MAX_Z, z));
    const s = this.scale;
    // centre the box in the free area, not the window
    this.cx = (x0 + x1) / 2 - (inset.left - inset.right) / 2 / s;
    this.cy = (y0 + y1) / 2 - (inset.top - inset.bottom) / 2 / s;
    this.clamp();
    this.redraw();
    this.emit("view");
  }

  clamp() {
    this.z = Math.max(MIN_Z, Math.min(MAX_Z, this.z));
    this.cx = Math.max(-0.2, Math.min(1.2, this.cx));
    this.cy = Math.max(0.0, Math.min(1.0, this.cy));
  }

  zoomAt(x, y, dz) {
    const [lon, lat] = this.lonlat(x, y);
    this.z = Math.max(MIN_Z, Math.min(MAX_Z, this.z + dz));
    const [wx, wy] = toWorld(lon, lat);
    const s = this.scale;
    this.cx = wx - (x - this.width / 2) / s;
    this.cy = wy - (y - this.height / 2) / s;
    this.clamp();
    this.redraw();
    this.emit("view");
  }

  add(layer) { this.layers.push(layer); this.redraw(); return () => this.remove(layer); }
  remove(layer) { this.layers = this.layers.filter((l) => l !== layer); this.redraw(); }

  // The next drag draws a box; fn({west, south, east, north}) gets it. drawBox(null) cancels.
  drawBox(fn) {
    this.boxHandler = fn;
    this.el.classList.toggle("drawing", !!fn);
  }

  redraw() {
    if (this.pending) return;
    this.pending = true;
    // A frame callback when the page is shown; a timer too, for a tab the browser is not painting.
    const run = () => { if (!this.pending) return; this.pending = false; this._paint(); };
    requestAnimationFrame(run);
    setTimeout(run, 120);
  }

  _paint() {
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = STYLE.water;
    ctx.fillRect(0, 0, this.width, this.height);
    const below = this.layers.filter((l) => !l.above);
    const above = this.layers.filter((l) => l.above);
    const s = this.scale;
    const tx = this.width / 2 - this.cx * s;
    const ty = this.height / 2 - this.cy * s;
    const level = this.z >= DETAIL_Z ? 1 : 0;
    const base = this.basemap || [];
    const pick = (name, lv) => base.find((l) => l.name === name && l.level === lv);
    const near = s > WORLD_PATH_PX;
    const m = 0.02 * Math.max(this.width, this.height) / s;
    const r = [this.cx - this.width / 2 / s - m, this.cy - this.height / 2 / s - m, this.cx + this.width / 2 / s + m, this.cy + this.height / 2 / s + m];
    // The world repeats east and west: a view across the 180th meridian (a Pacific domain framed from 170 to
    // 190 degrees) draws the land beyond it from the neighbouring copy, one world width over, and a wide window
    // zoomed right out draws every copy it shows.
    const copies = worldShifts(0, 1, r[0], r[2]);
    const views = copies.map((k) => ({ k, s, tx: tx + k * s, ty, r: [r[0] - k, r[1], r[2] - k, r[3]] }));
    const cachePaths = new Map();
    const pathOf = (layer, fill, view) => {
      if (!near) return layer.path;
      const key = `${layer.name}${layer.level}${fill}${view.k}`;
      if (!cachePaths.has(key)) cachePaths.set(key, screenPath(layer, view, fill));
      return cachePaths.get(key);
    };
    const space = (view) => (near
      ? ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0)
      : ctx.setTransform(this.dpr * s, 0, 0, this.dpr * s, this.dpr * view.tx, this.dpr * ty));
    const px = near ? 1 : 1 / s;
    // land and water
    const land = pick("land", level) || pick("land", 0);
    const lakes = level ? pick("lakes", 1) : null;
    for (const view of views) {
      space(view);
      if (land) { ctx.fillStyle = STYLE.land; ctx.fill(pathOf(land, true, view), "evenodd"); }
      if (lakes) { ctx.fillStyle = STYLE.lake; ctx.fill(pathOf(lakes, true, view), "evenodd"); }
    }
    this._graticule(ctx);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    for (const layer of below) { ctx.save(); layer.draw(ctx, this); ctx.restore(); }
    // lines
    for (const view of views) {
      space(view);
      const stroke = (layer, color, width) => {
        if (!layer) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = width * px;
        ctx.stroke(pathOf(layer, false, view));
      };
      ctx.lineJoin = "round";
      if (this.z >= COUNTY_Z) stroke(pick("counties", 2), STYLE.counties, 0.8);
      stroke(pick("states", level), STYLE.states, this.z >= COUNTY_Z ? 1.3 : 1);
      stroke(pick("borders", level), STYLE.borders, 1.3);
      if (level) { stroke(pick("land", 1), STYLE.coast, 1.1); stroke(pick("lakes", 1), STYLE.coast, 0.9); }
      else stroke(pick("coast", 0), STYLE.coast, 1.1);
    }
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    for (const layer of above) { ctx.save(); layer.draw(ctx, this); ctx.restore(); }
    if (this.boxDraft) {
      const b = this.boxDraft;
      ctx.setLineDash([7, 5]);
      ctx.strokeStyle = STYLE.box;
      ctx.lineWidth = 1.6;
      ctx.fillStyle = "rgba(37, 99, 235, 0.06)";
      const x = Math.min(b.x0, b.x1);
      const y = Math.min(b.y0, b.y1);
      ctx.fillRect(x, y, Math.abs(b.x1 - b.x0), Math.abs(b.y1 - b.y0));
      ctx.strokeRect(x, y, Math.abs(b.x1 - b.x0), Math.abs(b.y1 - b.y0));
      ctx.setLineDash([]);
    }
  }

  _graticule(ctx) {
    // faint latitude and longitude lines, 10 degrees apart, 5 or 1 when zoomed in
    const step = this.z >= 7 ? 1 : this.z >= 4.5 ? 5 : 10;
    const [west, north] = this.lonlat(0, 0);
    const [east, south] = this.lonlat(this.width, this.height);
    ctx.save();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.strokeStyle = STYLE.graticule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let lon = Math.ceil(west / step) * step; lon <= east; lon += step) {
      const [x] = this.screen(lon, 0);
      ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, this.height);
    }
    for (let lat = Math.ceil(Math.max(south, -80) / step) * step; lat <= Math.min(north, 80); lat += step) {
      const [, y] = this.screen(0, lat);
      ctx.moveTo(0, Math.round(y) + 0.5); ctx.lineTo(this.width, Math.round(y) + 0.5);
    }
    ctx.stroke();
    ctx.restore();
  }

  _bind() {
    const c = this.canvas;
    let drag = null;
    const local = (ev) => { const r = c.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
    c.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0) return;
      const [x, y] = local(ev);
      c.setPointerCapture(ev.pointerId);
      drag = { x, y, cx: this.cx, cy: this.cy, moved: false, box: !!this.boxHandler };
      if (drag.box) this.boxDraft = { x0: x, y0: y, x1: x, y1: y };
    });
    c.addEventListener("pointermove", (ev) => {
      const [x, y] = local(ev);
      if (drag) {
        if (Math.abs(x - drag.x) + Math.abs(y - drag.y) > 3) drag.moved = true;
        if (drag.box) {
          this.boxDraft = { x0: drag.x, y0: drag.y, x1: x, y1: y };
          this.redraw();
        } else if (drag.moved) {
          const s = this.scale;
          this.cx = drag.cx - (x - drag.x) / s;
          this.cy = drag.cy - (y - drag.y) / s;
          this.clamp();
          this.redraw();
          this.emit("view");
        }
      }
      const [lon, lat] = this.lonlat(x, y);
      this.emit("hover", lon, lat, x, y);
    });
    const end = (ev) => {
      if (!drag) return;
      const [x, y] = local(ev);
      const was = drag;
      drag = null;
      if (was.box) {
        const draft = this.boxDraft;
        this.boxDraft = null;
        if (was.moved && Math.abs(draft.x1 - draft.x0) > 12 && Math.abs(draft.y1 - draft.y0) > 12) {
          const [west, north] = this.lonlat(Math.min(draft.x0, draft.x1), Math.min(draft.y0, draft.y1));
          const [east, south] = this.lonlat(Math.max(draft.x0, draft.x1), Math.max(draft.y0, draft.y1));
          const fn = this.boxHandler;
          if (fn) fn({ west, south, east, north });
        }
        this.redraw();
        return;
      }
      if (!was.moved) { const [lon, lat] = this.lonlat(x, y); this.emit("click", lon, lat, x, y); }
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", () => { drag = null; this.boxDraft = null; this.redraw(); });
    c.addEventListener("pointerleave", () => this.emit("leave"));
    c.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const [x, y] = local(ev);
      const dz = -Math.sign(ev.deltaY) * Math.min(1, Math.abs(ev.deltaY) / (ev.deltaMode ? 3 : 120)) * 0.5;
      this.zoomAt(x, y, dz);
    }, { passive: false });
    c.addEventListener("dblclick", (ev) => { const [x, y] = local(ev); this.zoomAt(x, y, ev.shiftKey ? -1 : 1); });
  }
}

export { STYLE };
