// The wiki's maps: the offline basemap with outlines and annotations drawn on it, never a weather field. A
// cyclone's best track (one ink line with a dot a day, its peak in the accent), a tornado's path from its start
// point to its end point, an event's run recipe box (dashed) and the box a run asked for (solid). Every colour is
// one of the page's --aw-map-* tokens. Drawn in the projection
// the engine would pick for the middle of what is shown, fitted to it, redrawn on resize.
//
//   const m = wikiMap(canvas, {tracks: [{points, label}], paths: [{from, to}], dots: [{lon, lat, href, label}],
//                              boxes: [{w, e, s, n, dashed}], names: [{lon, lat, text}], inset: true, onpick(dot)});
//
// names label the place on the map; inset draws a small map of the region around it in a corner, with the
// main map's extent outlined, so a close-in map still says where on Earth it is.
//   m.close()

import { basemap, makeProjection, projectionFor } from "./map.js";

const wrap180 = (v) => ((((v + 180) % 360) + 360) % 360) - 180;

function css(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

// every lon, lat the map should hold, unwrapped around the first one so a track across 180 stays in one piece
function extent(opts) {
  const pts = [];
  for (const t of opts.tracks || []) for (const p of t.points) pts.push([p[0], p[1]]);
  for (const p of opts.paths || []) { pts.push(p.from); if (p.to) pts.push(p.to); }
  for (const d of opts.dots || []) pts.push([d.lon, d.lat]);
  for (const b of opts.boxes || []) { pts.push([b.w, b.s], [b.e, b.n]); }
  if (!pts.length) return null;
  const lon0 = pts[0][0];
  const lons = pts.map((p) => lon0 + wrap180(p[0] - lon0));
  const lats = pts.map((p) => p[1]);
  return { w: Math.min(...lons), e: Math.max(...lons), s: Math.min(...lats), n: Math.max(...lats) };
}

export function wikiMap(canvas, opts = {}) {
  let W = 0; let H = 0; let dpr = 1;
  let map = null;
  let proj = null;
  let scale = 1;
  let cx = 0; let cy = 0;
  const ext = extent(opts) || { w: -130, e: 40, s: -40, n: 60 };
  const minSpan = opts.minSpanDeg || 6;
  const lat0 = Math.max(-80, Math.min(80, (ext.s + ext.n) / 2));
  const lon0 = wrap180((ext.w + ext.e) / 2);
  const wide = ext.e - ext.w > 100 || ext.n - ext.s > 70;
  // A view wider than a continent reads best on Mercator; closer in, the engine's own choice for the point.
  proj = makeProjection(wide ? { map_proj: "mercator", ref_lat: 0, ref_lon: lon0, stand_lon: lon0, truelat1: 0, truelat2: 0 }
    : projectionFor(lat0, lon0));

  function fit() {
    const pts = [];
    const n = 12;
    const w = ext.w - Math.max(0, (minSpan - (ext.e - ext.w)) / 2);
    const e = ext.e + Math.max(0, (minSpan - (ext.e - ext.w)) / 2);
    const s = Math.max(-84, ext.s - Math.max(0, (minSpan - (ext.n - ext.s)) / 2));
    const nn = Math.min(84, ext.n + Math.max(0, (minSpan - (ext.n - ext.s)) / 2));
    for (let i = 0; i <= n; i++) {
      const lon = w + ((e - w) * i) / n;
      const lat = s + ((nn - s) * i) / n;
      pts.push(proj.fwd(lon, s), proj.fwd(lon, nn), proj.fwd(w, lat), proj.fwd(e, lat));
    }
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const x0 = Math.min(...xs); const x1 = Math.max(...xs); const y0 = Math.min(...ys); const y1 = Math.max(...ys);
    const pad = 0.12;
    scale = Math.min(W / ((x1 - x0) * (1 + 2 * pad) || 1), H / ((y1 - y0) * (1 + 2 * pad) || 1));
    cx = (x0 + x1) / 2; cy = (y0 + y1) / 2;
  }
  const px = (lon, lat) => { const [x, y] = proj.fwd(lon, lat); return [W / 2 + (x - cx) * scale, H / 2 - (y - cy) * scale]; };
  const geo = (x, y) => proj.inv(cx + (x - W / 2) / scale, cy - (y - H / 2) / scale);

  function path(ctx, points) {
    let last = null;
    for (const [x, y] of points) {
      if (!last || Math.abs(x - last[0]) > W / 2) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      last = [x, y];
    }
  }

  function viewBounds() {
    let w = Infinity; let e = -Infinity; let s = Infinity; let n = -Infinity;
    for (let i = 0; i <= 10; i++) {
      for (const [x, y] of [[(W * i) / 10, 0], [(W * i) / 10, H], [0, (H * i) / 10], [W, (H * i) / 10]]) {
        const [lon, lat] = geo(x, y);
        if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
        w = Math.min(w, lon); e = Math.max(e, lon); s = Math.min(s, lat); n = Math.max(n, lat);
      }
    }
    if (e - w > 300 || wide) { w = -180; e = 180; }
    return { w: w - 1, e: e + 1, s: s - 1, n: n + 1 };
  }

  function layer(ctx, list, colour, width, view) {
    ctx.strokeStyle = colour; ctx.lineWidth = width;
    ctx.beginPath();
    for (const line of list || []) {
      if (line.e < view.w || line.w > view.e || line.n < view.s || line.s > view.n) continue;
      const pts = [];
      for (let i = 0; i < line.pts.length; i += 2) pts.push(px(line.pts[i], line.pts[i + 1]));
      path(ctx, pts);
    }
    ctx.stroke();
  }

  const hits = [];
  function draw() {
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = css("--aw-map-sea", "#fafafa");
    ctx.fillRect(0, 0, W, H);
    const view = viewBounds();
    // graticule
    const span = view.n - view.s;
    const step = span > 90 ? 30 : span > 36 ? 10 : span > 15 ? 5 : 2;
    ctx.strokeStyle = css("--aw-map-grid", "#efefef"); ctx.lineWidth = 1;
    ctx.beginPath();
    for (let lon = Math.ceil(view.w / step) * step; lon <= view.e; lon += step) {
      const pts = [];
      for (let i = 0; i <= 40; i++) pts.push(px(lon, Math.max(-85, Math.min(85, view.s + (span * i) / 40))));
      path(ctx, pts);
    }
    for (let lat = Math.ceil(view.s / step) * step; lat <= view.n; lat += step) {
      const pts = [];
      for (let i = 0; i <= 80; i++) pts.push(px(view.w + ((view.e - view.w) * i) / 80, lat));
      path(ctx, pts);
    }
    ctx.stroke();
    if (map) {
      if (!wide) layer(ctx, map.layers.state, css("--aw-map-state", "#d9d9d9"), 0.7, view);
      layer(ctx, map.layers.lake, css("--aw-map-lake", "#c4c4c4"), 0.8, view);
      layer(ctx, map.layers.border, css("--aw-map-border", "#b4b4b4"), 0.9, view);
      layer(ctx, map.layers.coast, css("--aw-map-coast", "#a3a3a3"), 1.1, view);
    }
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    // boxes: the recipe dashed, a run solid
    for (const b of opts.boxes || []) {
      const ring = [];
      const N = 20;
      const e = b.w <= b.e ? b.e : b.e + 360;
      for (let i = 0; i <= N; i++) ring.push(px(b.w + ((e - b.w) * i) / N, b.s));
      for (let i = 1; i <= N; i++) ring.push(px(e, b.s + ((b.n - b.s) * i) / N));
      for (let i = 1; i <= N; i++) ring.push(px(e - ((e - b.w) * i) / N, b.n));
      for (let i = 1; i <= N; i++) ring.push(px(b.w, b.n - ((b.n - b.s) * i) / N));
      ctx.beginPath();
      ring.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.setLineDash(b.dashed ? [4, 3] : []);
      ctx.fillStyle = "rgba(37, 99, 235, 0.05)";
      ctx.fill();
      ctx.strokeStyle = b.dashed ? css("--aw-map-ask", "#2563eb") : css("--aw-map-fit", "#2563eb");
      ctx.lineWidth = b.dashed ? 1 : 1.6;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // tracks: one line, a dot at each 00 UTC point, the strongest point in the accent
    const ink = css("--aw-map-track", "#171717");
    for (const t of opts.tracks || []) {
      const pts = t.points;
      const thin = !!t.thin;
      ctx.strokeStyle = ink; ctx.globalAlpha = thin ? 0.45 : 1;
      ctx.lineWidth = thin ? 1 : 1.6;
      ctx.beginPath();
      path(ctx, pts.map((p) => px(p[0], p[1])));
      ctx.stroke();
      ctx.globalAlpha = 1;
      if (!thin) {
        ctx.fillStyle = ink;
        for (const p of pts) {
          if (!/ 00:00$/.test(String(p[3] || ""))) continue;
          const [x, y] = px(p[0], p[1]);
          ctx.beginPath(); ctx.arc(x, y, 2.2, 0, 2 * Math.PI); ctx.fill();
        }
        let peak = null;
        for (const p of pts) if (Number.isFinite(p[2]) && (!peak || p[2] > peak[2])) peak = p;
        if (peak) {
          const [x, y] = px(peak[0], peak[1]);
          ctx.fillStyle = css("--aw-map-peak", "#2563eb"); ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
        }
      }
    }
    // tornado paths
    for (const p of opts.paths || []) {
      const a = px(p.from[0], p.from[1]);
      const pc = css("--aw-map-path", "#171717");
      ctx.strokeStyle = pc; ctx.fillStyle = pc; ctx.lineWidth = 2.2;
      if (p.to) {
        const b = px(p.to[0], p.to[1]);
        ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
        ctx.beginPath(); ctx.arc(b[0], b[1], 3, 0, 2 * Math.PI); ctx.fill();
      }
      ctx.beginPath(); ctx.arc(a[0], a[1], 5, 0, 2 * Math.PI);
      ctx.fillStyle = css("--aw-map-peak", "#2563eb"); ctx.fill();
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2; ctx.stroke();
    }
    // place names, with a dark halo so they read over any line
    ctx.font = "600 12px 'Segoe UI', system-ui, sans-serif";
    for (const l of opts.names || []) {
      const [x, y] = px(l.lon, l.lat);
      const tw = ctx.measureText(l.text).width;
      let tx = Math.min(Math.max(6, x + 9), W - tw - 6);
      let ty = l.under ? y + 18 : y - 9;
      if (ty < 14) ty = y + 18;
      ctx.lineWidth = 3.5; ctx.strokeStyle = css("--aw-map-halo", "rgba(255,255,255,.92)"); ctx.strokeText(l.text, tx, ty);
      ctx.fillStyle = css("--aw-map-name", "#171717"); ctx.fillText(l.text, tx, ty);
    }
    if (opts.inset && map) drawInset(ctx);
    // dots: other events, each a link
    hits.length = 0;
    ctx.font = "11px 'Segoe UI', system-ui, sans-serif";
    for (const d of opts.dots || []) {
      const [x, y] = px(d.lon, d.lat);
      ctx.fillStyle = d.colour || css("--aw-map-dot", "#171717");
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, d.big ? 5.5 : 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      if (d.label && opts.labels) {
        ctx.fillStyle = css("--aw-map-name", "#171717");
        ctx.fillText(d.label, x + 8, y + 4);
      }
      hits.push({ x, y, d });
    }
  }

  // The locator: the same projection zoomed out to three times the map (at least 40 degrees) around the middle, in the lower left corner.
  function drawInset(ctx) {
    const iw = Math.round(Math.min(104, W * 0.28)); const ih = Math.round(iw * 0.72);
    const x0 = 6; const y0 = H - ih - 6;
    const mid = [lon0, lat0];
    const span = Math.max(40, Math.min(120, 3 * Math.max(ext.e - ext.w, ext.n - ext.s, minSpan)));
    const probe = [];
    for (let i = 0; i <= 8; i++) {
      probe.push(proj.fwd(mid[0] - span / 2 + (span * i) / 8, Math.max(-84, mid[1] - span / 3)));
      probe.push(proj.fwd(mid[0] - span / 2 + (span * i) / 8, Math.min(84, mid[1] + span / 3)));
    }
    const xs = probe.map((p) => p[0]); const ys = probe.map((p) => p[1]);
    const k = Math.min(iw / (Math.max(...xs) - Math.min(...xs) || 1), ih / (Math.max(...ys) - Math.min(...ys) || 1));
    const [mx, my] = proj.fwd(mid[0], mid[1]);
    const ipx = (lon, lat) => { const [x, y] = proj.fwd(lon, lat); return [x0 + iw / 2 + (x - mx) * k, y0 + ih / 2 - (y - my) * k]; };
    ctx.save();
    ctx.beginPath(); ctx.rect(x0, y0, iw, ih); ctx.clip();
    ctx.fillStyle = css("--aw-map-sea", "#fafafa"); ctx.fillRect(x0, y0, iw, ih);
    const view = { w: mid[0] - span, e: mid[0] + span, s: mid[1] - span, n: mid[1] + span };
    for (const [name, colour, width] of [["border", css("--aw-map-border", "#cdcdcd"), 0.6], ["coast", css("--aw-map-coast", "#b4b4b4"), 0.8]]) {
      ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.beginPath();
      for (const line of map.layers[name] || []) {
        if (line.e < view.w || line.w > view.e || line.n < view.s || line.s > view.n) continue;
        let last = null;
        for (let i = 0; i < line.pts.length; i += 2) {
          const q = ipx(line.pts[i], line.pts[i + 1]);
          if (!last || Math.abs(q[0] - last[0]) > iw) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
          last = q;
        }
      }
      ctx.stroke();
    }
    // the main map's extent
    const corners = [[0, 0], [W, 0], [W, H], [0, H]].map(([x, y]) => geo(x, y)).filter((g) => Number.isFinite(g[0]) && Number.isFinite(g[1]));
    ctx.strokeStyle = css("--aw-map-inset", "#2563eb"); ctx.lineWidth = 1.4;
    if (corners.length === 4) {
      ctx.beginPath();
      corners.forEach((g, i) => { const q = ipx(g[0], g[1]); if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); });
      ctx.closePath(); ctx.stroke();
    }
    ctx.restore();
    ctx.strokeStyle = css("--aw-map-border", "#cdcdcd"); ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, iw, ih);
  }

  function size() {
    const r = canvas.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    W = Math.max(1, r.width); H = Math.max(1, r.height);
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    fit();
  }

  function near(ev) {
    const r = canvas.getBoundingClientRect();
    const x = ev.clientX - r.left; const y = ev.clientY - r.top;
    let best = null; let dist = 12;
    for (const hit of hits) {
      const d = Math.hypot(hit.x - x, hit.y - y);
      if (d < dist) { dist = d; best = hit.d; }
    }
    return best;
  }
  if (opts.onpick) {
    canvas.addEventListener("click", (ev) => { const d = near(ev); if (d) opts.onpick(d); });
    canvas.addEventListener("mousemove", (ev) => {
      const d = near(ev);
      canvas.style.cursor = d ? "pointer" : "default";
      canvas.title = d ? d.label || "" : "";
    });
  }

  const observer = new ResizeObserver(() => { size(); draw(); });
  observer.observe(canvas);
  size();
  draw();
  basemap().then((m) => { map = m; draw(); });
  return { close() { observer.disconnect(); }, redraw: draw };
}
