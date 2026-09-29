// The Create map: an offline basemap (Natural Earth coastlines, lakes, country borders and state lines, bundled in
// /static/map/basemap.json, public domain), the box the person asked for (dashed) and the grid the engine fitted
// (solid). Plain canvas, no library, no network.
//
//   const m = boxMap(canvas, {onchange(box), view});
//   m.setBox({lat, lon, width_km, height_km});  m.setFit(fit);  m.zoom(f);  m.frameBox();
//
// Click places the box's centre, dragging the box moves it, dragging a corner sizes it, dragging elsewhere pans,
// the wheel zooms. The map is drawn in the projection the engine's domain wizard picks for the box (Lambert
// conformal between 25 and 60 degrees, Mercator nearer the equator, polar stereographic nearer a pole), and once a
// fit is shown, in the fit's own projection: the fitted grid is then a true rectangle on the map, the shape the
// forecast will have.

const KM_PER_DEG = 111.32;
const MIN_KM = 50;
const MAX_KM = 8000;
const HANDLE_PX = 9;

let basemapPromise = null;

// The basemap, decoded once per page: each line a Float32Array of lon, lat pairs with its bounding box.
export function basemap() {
  if (!basemapPromise) {
    basemapPromise = fetch("/static/map/basemap.json", { credentials: "same-origin" })
      .then((r) => (r.ok ? r.json() : null))
      .then((doc) => {
        if (!doc) return null;
        const layers = {};
        for (const [key, lines] of Object.entries(doc.layers)) {
          layers[key] = lines.map((code) => {
            const pts = new Float32Array(code.length);
            let x = 0;
            let y = 0;
            let w = Infinity; let e = -Infinity; let s = Infinity; let n = -Infinity;
            for (let i = 0; i < code.length; i += 2) {
              x += code[i]; y += code[i + 1];
              const lon = x / doc.scale; const lat = y / doc.scale;
              pts[i] = lon; pts[i + 1] = lat;
              if (lon < w) w = lon; if (lon > e) e = lon; if (lat < s) s = lat; if (lat > n) n = lat;
            }
            return { pts, w, e, s, n };
          });
        }
        return { notice: doc.notice, layers };
      })
      .catch(() => null);
  }
  return basemapPromise;
}

function css(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

const EARTH_KM = 6370;
const RAD = Math.PI / 180;
const MERCATOR_MAX_LAT = 25;
const LAMBERT_MAX_LAT = 60;
const wrap180 = (v) => ((((v + 180) % 360) + 360) % 360) - 180;

// The projection the domain wizard picks for a point (domain_wizard._projection_entries).
export function projectionFor(lat, lon) {
  const a = Math.abs(lat);
  if (a >= MERCATOR_MAX_LAT && a <= LAMBERT_MAX_LAT) {
    const sign = lat < 0 ? -1 : 1;
    return { map_proj: "lambert", ref_lat: lat, ref_lon: lon, stand_lon: lon,
      truelat1: sign * Math.max(15, a - 10), truelat2: sign * Math.min(70, a + 10) };
  }
  return { map_proj: a < MERCATOR_MAX_LAT ? "mercator" : "polar", ref_lat: lat, ref_lon: lon, stand_lon: lon,
    truelat1: lat, truelat2: lat };
}

// fwd(lon, lat) -> [x, y] km on the projection plane, 0,0 at the reference point, true scale at the true latitudes
// (the engine measures its grid spacing on this plane); inv(x, y) -> [lon, lat].
export function makeProjection(p) {
  const stand = p.stand_lon ?? p.ref_lon;
  const tl1 = p.truelat1 ?? p.ref_lat;
  const tl2 = p.truelat2 ?? tl1;
  let raw;
  let unraw;
  if (p.map_proj === "mercator") {
    const c = EARTH_KM * Math.cos(tl1 * RAD);
    raw = (lon, lat) => {
      const la = Math.max(-85, Math.min(85, lat));
      return [c * wrap180(lon - stand) * RAD, c * Math.log(Math.tan((45 + la / 2) * RAD))];
    };
    unraw = (x, y) => [wrap180(stand + x / c / RAD), (2 * Math.atan(Math.exp(y / c))) / RAD - 90];
  } else {
    // Lambert conformal and polar stereographic share one form, rho = C tan(45 - lat/2)^n, polar being n = 1.
    const hemi = tl1 < 0 ? -1 : 1;
    const t1 = hemi * tl1 * RAD;
    const t2 = hemi * tl2 * RAD;
    const tq = (t) => Math.tan(Math.PI / 4 - t / 2);
    let n = 1;
    if (p.map_proj !== "polar") {
      n = Math.abs(t1 - t2) < 1e-9 ? Math.sin(t1) : Math.log(Math.cos(t1) / Math.cos(t2)) / Math.log(tq(t1) / tq(t2));
    }
    const C = (EARTH_KM * Math.cos(t1)) / (n * tq(t1) ** n);
    raw = (lon, lat) => {
      const la = Math.max(-89.5, Math.min(89.9, hemi * lat)) * RAD;
      const rho = C * tq(la) ** n;
      const th = n * wrap180(lon - stand) * RAD;
      return [rho * Math.sin(th), -hemi * rho * Math.cos(th)];
    };
    unraw = (x, y) => {
      const Y = -hemi * y;
      const rho = Math.hypot(x, Y);
      const th = Math.atan2(x, Y);
      const la = 90 - (2 * Math.atan((rho / C) ** (1 / n))) / RAD;
      return [wrap180(stand + th / n / RAD), hemi * la];
    };
  }
  const [x0, y0] = raw(p.ref_lon, p.ref_lat);
  return {
    params: p,
    fwd: (lon, lat) => { const [x, y] = raw(lon, lat); return [x - x0, y - y0]; },
    inv: (x, y) => unraw(x + x0, y + y0),
  };
}

export function boxHalves(box) {
  const dlat = box.height_km / 2 / KM_PER_DEG;
  const dlon = box.width_km / 2 / (KM_PER_DEG * Math.max(0.05, Math.cos((box.lat * Math.PI) / 180)));
  return { dlat, dlon };
}

// The asked-for box as the engine reads it: a latitude-longitude rectangle (api.region_polygon).
function boxEdges(b) {
  const { dlat, dlon } = boxHalves(b);
  return { w: b.lon - dlon, e: b.lon + dlon, s: b.lat - dlat, n: b.lat + dlat };
}

export function boxMap(canvas, opts = {}) {
  const state = {
    // the view: centre and pixels per degree of latitude (CSS pixels)
    lon0: opts.view ? opts.view.lon : -96, lat0: opts.view ? opts.view.lat : 38.5, k: opts.view ? opts.view.k : 13,
    box: null, fit: null, map: null, proj: null, c: [0, 0],
  };
  let W = 0; let H = 0; let dpr = 1;

  const scale = () => state.k / KM_PER_DEG; // pixels per km on the plane
  const recentre = () => { state.c = state.proj.fwd(state.lon0, state.lat0); };
  // The projection follows the fit when one is shown, else the box, else the view's centre.
  function reproject() {
    const f = state.fit;
    const fp = f && f.projection;
    const p = fp && fp.map_proj && typeof fp.ref_lat === "number" && typeof fp.ref_lon === "number" ? fp
      : state.box ? projectionFor(state.box.lat, state.box.lon) : projectionFor(state.lat0, state.lon0);
    state.proj = makeProjection(p);
    recentre();
  }
  const plane = (X, Y) => [W / 2 + (X - state.c[0]) * scale(), H / 2 - (Y - state.c[1]) * scale()];
  const px = (lon, lat) => { const [X, Y] = state.proj.fwd(lon, lat); return plane(X, Y); };
  const geo = (x, y) => state.proj.inv(state.c[0] + (x - W / 2) / scale(), state.c[1] - (y - H / 2) / scale());
  function centreOn(X, Y) {
    const [lon, lat] = state.proj.inv(X, Y);
    state.lon0 = lon;
    state.lat0 = Math.max(-85, Math.min(85, lat));
    recentre();
  }

  function size() {
    const r = canvas.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    W = Math.max(1, r.width); H = Math.max(1, r.height);
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  }

  // A polyline through projected points; a jump wider than half the map is the projection's seam, not a line.
  function path(ctx, points) {
    let last = null;
    for (const [x, y] of points) {
      if (!last || Math.abs(x - last[0]) > W / 2 || Math.abs(y - last[1]) > H / 2) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      last = [x, y];
    }
  }

  function lines(ctx, list, color, width, view) {
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.beginPath();
    for (const line of list) {
      if (line.e < view.w || line.w > view.e || line.n < view.s || line.s > view.n) continue;
      const p = line.pts;
      const points = [];
      for (let i = 0; i < p.length; i += 2) points.push(px(p[i], p[i + 1]));
      path(ctx, points);
    }
    ctx.stroke();
  }

  // The latitude-longitude extent on screen, from points around its edge; a pole in view, or the date line,
  // widens it to every longitude.
  function viewBounds() {
    let w = Infinity; let e = -Infinity; let s = Infinity; let n = -Infinity;
    const take = (x, y) => {
      const [lon, lat] = geo(x, y);
      if (!Number.isFinite(lon) || !Number.isFinite(lat)) return;
      if (lon < w) w = lon; if (lon > e) e = lon; if (lat < s) s = lat; if (lat > n) n = lat;
    };
    const N = 12;
    for (let i = 0; i <= N; i++) {
      take((W * i) / N, 0); take((W * i) / N, H); take(0, (H * i) / N); take(W, (H * i) / N); take((W * i) / N, H / 2);
    }
    if (state.proj.params.map_proj !== "mercator") {
      for (const pole of [90, -90]) {
        const [x, y] = px(0, pole);
        if (x >= 0 && x <= W && y >= 0 && y <= H) { if (pole > 0) n = 90; else s = -90; w = -180; e = 180; }
      }
    }
    const [cl] = geo(W / 2, H / 2);
    if (e - w > 300 && Math.abs(cl) > 120) { w = -180; e = 180; }
    return { w: w - 1, e: e + 1, s: Math.max(-90, s - 1), n: Math.min(90, n + 1) };
  }

  function draw() {
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = css("--aw-map-sea", "#fafafa");
    ctx.fillRect(0, 0, W, H);
    const view = viewBounds();

    // graticule, curved as the projection bends it
    const latSpan = view.n - view.s;
    const step = latSpan > 90 ? 30 : latSpan > 36 ? 10 : latSpan > 15 ? 5 : latSpan > 6 ? 2 : latSpan > 2.5 ? 1 : 0.5;
    const meridians = [];
    const parallels = [];
    for (let lon = Math.ceil(view.w / step) * step; lon <= view.e; lon += step) {
      const pts = [];
      for (let i = 0; i <= 48; i++) pts.push(px(lon, Math.max(-89, Math.min(89, view.s + (latSpan * i) / 48))));
      meridians.push([lon, pts]);
    }
    for (let lat = Math.ceil(view.s / step) * step; lat <= view.n; lat += step) {
      const pts = [];
      for (let i = 0; i <= 96; i++) pts.push(px(view.w + ((view.e - view.w) * i) / 96, lat));
      parallels.push([lat, pts]);
    }
    ctx.strokeStyle = css("--aw-map-grid", "#efefef"); ctx.lineWidth = 1;
    ctx.beginPath();
    for (const [, pts] of meridians) path(ctx, pts);
    for (const [, pts] of parallels) path(ctx, pts);
    ctx.stroke();

    const map = state.map;
    if (map) {
      if (state.k > 5) lines(ctx, map.layers.state || [], css("--aw-map-state", "#d9d9d9"), 0.8, view);
      lines(ctx, map.layers.lake || [], css("--aw-map-lake", "#c4c4c4"), 0.9, view);
      lines(ctx, map.layers.border || [], css("--aw-map-border", "#b4b4b4"), 1.1, view);
      lines(ctx, map.layers.coast || [], css("--aw-map-coast", "#a3a3a3"), 1.2, view);
    }

    // graticule labels over the lines: meridians along the bottom edge, parallels along the left
    ctx.fillStyle = css("--aw-map-label", "#9a9a9a");
    ctx.font = "10.5px ui-monospace, Consolas, monospace";
    const deg = (v, pos, neg) => `${+Math.abs(v).toFixed(1)}°${v >= 0 ? pos : neg}`;
    const inside = ([x, y]) => x >= 0 && x <= W && y >= 0 && y <= H;
    for (const [lon, pts] of meridians) {
      const low = pts.filter(inside).sort((a, b) => b[1] - a[1])[0];
      if (low && low[1] > H - 40 && low[0] < W - 40) ctx.fillText(deg(wrap180(lon), "E", "W"), low[0] + 3, H - 5);
    }
    for (const [lat, pts] of parallels) {
      const left = pts.filter(inside).sort((a, b) => a[0] - b[0])[0];
      if (left && left[0] < 40 && left[1] > 14 && left[1] < H - 20) ctx.fillText(deg(lat, "N", "S"), 4, left[1] - 3);
    }

    // the fitted grid: solid, its nests inside it, true rectangles on the fit's own projection plane
    const fit = state.fit;
    const fitColor = css("--aw-map-fit", "#2563eb");
    if (fit && fit.domains && fit.domains.length && fit.centre && fit.centre.lat !== null) {
      const root = fit.domains[0];
      const rect = (west, south, wkm, hkm) => {
        const [x0, y0] = plane(west, south + hkm);
        const [x1, y1] = plane(west + wkm, south);
        return [x0, y0, x1 - x0, y1 - y0];
      };
      ctx.lineWidth = 2; ctx.strokeStyle = fitColor; ctx.fillStyle = "rgba(37, 99, 235, 0.06)";
      const r = rect(-root.width_km / 2, -root.height_km / 2, root.width_km, root.height_km);
      ctx.fillRect(...r); ctx.strokeRect(...r);
      const corners = { [root.grid_id]: { west: -root.width_km / 2, south: -root.height_km / 2, dx: root.dx_km } };
      for (const d of fit.domains.slice(1)) {
        const parent = corners[d.parent_id];
        if (!parent || !d.i_parent_start) continue;
        const west = parent.west + (d.i_parent_start - 1) * parent.dx;
        const south = parent.south + (d.j_parent_start - 1) * parent.dx;
        corners[d.grid_id] = { west, south, dx: d.dx_km };
        ctx.strokeRect(...rect(west, south, d.width_km, d.height_km));
      }
    }

    // the asked-for box: dashed, its parallels curving with the map, with corner handles
    const b = state.box;
    if (b) {
      const ask = css("--aw-map-draft", "#171717");
      const g = boxEdges(b);
      const N = 24;
      const ring = [];
      for (let i = 0; i <= N; i++) ring.push(px(g.w + ((g.e - g.w) * i) / N, g.s));
      for (let i = 1; i <= N; i++) ring.push(px(g.e, g.s + ((g.n - g.s) * i) / N));
      for (let i = 1; i <= N; i++) ring.push(px(g.e - ((g.e - g.w) * i) / N, g.n));
      for (let i = 1; i <= N; i++) ring.push(px(g.w, g.n - ((g.n - g.s) * i) / N));
      ctx.beginPath();
      ring.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      if (!fit) { ctx.fillStyle = "rgba(23, 23, 23, 0.04)"; ctx.fill(); }
      ctx.setLineDash([7, 5]); ctx.strokeStyle = ask; ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = ask;
      for (const [hx, hy] of cornerPoints(b)) ctx.fillRect(hx - 3.5, hy - 3.5, 7, 7);
      const [cx, cy] = px(b.lon, b.lat);
      ctx.beginPath(); ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy); ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6); ctx.stroke();
    }
  }

  function cornerPoints(b) {
    const g = boxEdges(b);
    return [px(g.w, g.n), px(g.e, g.n), px(g.w, g.s), px(g.e, g.s)];
  }

  function hit(x, y) {
    const b = state.box;
    if (!b) return null;
    for (const [hx, hy] of cornerPoints(b)) {
      if (Math.abs(x - hx) <= HANDLE_PX && Math.abs(y - hy) <= HANDLE_PX) return "corner";
    }
    const [lon, lat] = geo(x, y);
    const g = boxEdges(b);
    if (lat > g.s && lat < g.n && Math.abs(wrap180(lon - b.lon)) < (g.e - g.w) / 2) return "box";
    return null;
  }

  const clampKm = (v) => Math.round(Math.max(MIN_KM, Math.min(MAX_KM, v)) / 10) * 10;
  const changed = () => { state.fit = null; draw(); if (opts.onchange) opts.onchange(state.box); };

  let drag = null;
  canvas.addEventListener("pointerdown", (ev) => {
    const r = canvas.getBoundingClientRect();
    const x = ev.clientX - r.left; const y = ev.clientY - r.top;
    const what = hit(x, y);
    drag = { what: what || "pan", x, y, moved: false, c: [...state.c],
      box: state.box ? { ...state.box } : null, geo: geo(x, y) };
    canvas.setPointerCapture(ev.pointerId);
    if (what) canvas.classList.add("moving");
  });
  canvas.addEventListener("pointermove", (ev) => {
    const r = canvas.getBoundingClientRect();
    const x = ev.clientX - r.left; const y = ev.clientY - r.top;
    if (!drag) {
      const what = hit(x, y);
      canvas.classList.toggle("over-box", what === "box");
      canvas.classList.toggle("over-corner", what === "corner");
      return;
    }
    if (Math.abs(x - drag.x) + Math.abs(y - drag.y) > 4) drag.moved = true;
    if (!drag.moved) return;
    if (drag.what === "pan") {
      centreOn(drag.c[0] - (x - drag.x) / scale(), drag.c[1] + (y - drag.y) / scale());
      draw();
    } else if (drag.what === "box") {
      const [lon, lat] = geo(x, y);
      state.box.lon = +wrap180(drag.box.lon + wrap180(lon - drag.geo[0])).toFixed(2);
      state.box.lat = +Math.max(-80, Math.min(80, drag.box.lat + (lat - drag.geo[1]))).toFixed(2);
      changed();
    } else if (drag.what === "corner") {
      const [lon, lat] = geo(x, y);
      const b = state.box;
      b.width_km = clampKm(2 * Math.abs(wrap180(lon - b.lon)) * KM_PER_DEG * Math.cos((b.lat * Math.PI) / 180));
      b.height_km = clampKm(2 * Math.abs(lat - b.lat) * KM_PER_DEG);
      changed();
    }
  });
  const end = (ev) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    canvas.classList.remove("moving");
    if (!d.moved && d.what !== "corner") {
      const [lon, lat] = d.geo;
      const size = state.box || { width_km: opts.defaultKm || 600, height_km: opts.defaultKm || 600 };
      state.box = { lat: +lat.toFixed(2), lon: +lon.toFixed(2), width_km: size.width_km, height_km: size.height_km };
      state.fit = null;
      reproject();
      changed();
    } else if (d.moved) {
      // The projection follows the box (or the view) once a move ends, never during it.
      reproject();
      draw();
    }
    try { canvas.releasePointerCapture(ev.pointerId); } catch (_) { /* already released */ }
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", () => { drag = null; canvas.classList.remove("moving"); });
  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const r = canvas.getBoundingClientRect();
    zoomAt(ev.deltaY < 0 ? 1.25 : 0.8, ev.clientX - r.left, ev.clientY - r.top);
  }, { passive: false });

  function zoomAt(f, x, y) {
    const [X, Y] = state.proj.fwd(...geo(x, y));
    state.k = Math.max(1.5, Math.min(400, state.k * f));
    // keep the point under the pointer where it was
    centreOn(X - (x - W / 2) / scale(), Y + (y - H / 2) / scale());
    draw();
  }

  reproject();
  const observer = new ResizeObserver(() => { size(); draw(); });
  observer.observe(canvas);
  size();
  draw();
  basemap().then((m) => { state.map = m; draw(); if (opts.onmap) opts.onmap(m); });

  return {
    setBox(box) { state.box = box ? { ...box } : null; state.fit = null; reproject(); draw(); },
    setFit(fit) {
      if (fit === state.fit) return;
      state.fit = fit;
      reproject();
      draw();
    },
    box: () => (state.box ? { ...state.box } : null),
    zoom(f) { zoomAt(f, W / 2, H / 2); },
    frameBox() {
      let x0; let x1; let y0; let y1;
      const f = state.fit;
      if (f && f.domains && f.domains.length) {
        const root = f.domains[0];
        [x0, x1, y0, y1] = [-root.width_km / 2, root.width_km / 2, -root.height_km / 2, root.height_km / 2];
      } else if (state.box) {
        const g = boxEdges(state.box);
        const pts = [];
        for (let i = 0; i <= 8; i++) {
          const lon = g.w + ((g.e - g.w) * i) / 8;
          const lat = g.s + ((g.n - g.s) * i) / 8;
          pts.push(state.proj.fwd(lon, g.s), state.proj.fwd(lon, g.n), state.proj.fwd(g.w, lat), state.proj.fwd(g.e, lat));
        }
        x0 = Math.min(...pts.map((q) => q[0])); x1 = Math.max(...pts.map((q) => q[0]));
        y0 = Math.min(...pts.map((q) => q[1])); y1 = Math.max(...pts.map((q) => q[1]));
      } else return;
      const s = 0.9 * Math.min(H / Math.max(1, y1 - y0), W / Math.max(1, x1 - x0));
      state.k = Math.max(1.5, Math.min(400, s * KM_PER_DEG * 0.5));
      centreOn((x0 + x1) / 2, (y0 + y1) / 2);
      draw();
    },
    view: () => ({ lat: state.lat0, lon: state.lon0, k: state.k }),
    close() { observer.disconnect(); },
  };
}
