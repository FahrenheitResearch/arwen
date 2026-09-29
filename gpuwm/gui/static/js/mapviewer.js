// One forecast on the map. Its maps sit where they belong, one grid over another at their true extents, with
// a map picker and a time bar with Play on the map itself. While the forecast runs, the event stream brings
// each new frame in as the renderer draws it, and the time bar grows.
//
//   openViewer(app, runId, wanted, route)  app: {map, w (the screens' words), stage (the element the floating
//   controls go in), looks, links (more controls for the top right), onchange()}; route: the page name the
//   address keeps ("results" or "watch"), so a picked map survives a reload.

import { h, append, bar, fill, paceLine, prepLine } from "./core.js";
import * as api from "./api.js";
import { notice, errorText } from "./router.js";
import { grids, gridOutline, domainNumber, placesAt, gridForPicture, borrowGeoref, lonLatBox, boundsBox, screenRings,
  wrapLon } from "./geo.js";
import { placePicture, drawPlaced } from "./field.js";
import { productName, groupProducts } from "./looks.js";
import { parseTime, longTime, leftWords, hourWords } from "./time.js";
import { actionButton } from "./command.js";
import { openDownscale } from "./downscale.js";

const PLAY_MS = 650;
const REFRESH_MS = 1500;
const POLL_MS = 8000;
const QUICK = 3;

function latLonWords(lat, lon) {
  // a map framed across the 180th meridian reads 185 under the pointer, which is 175 W
  lon = wrapLon(lon);
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(2)}° ${ns}, ${Math.abs(lon).toFixed(2)}° ${ew}`;
}

function domainLabel(token, grid) {
  const n = domainNumber(token);
  const name = n ? `d${String(n).padStart(2, "0")}` : token;
  const km = grid ? grid.dx_km : Number((/-(\d+(?:\.\d+)?)km$/.exec(token) || [])[1]);
  const m = /-(\d+)m$/.exec(token);
  if (km) return `${name} · ${+km.toFixed(2)} km`;
  if (m) return `${name} · ${m[1]} m`;
  return name;
}

export async function openViewer(app, runId, wanted = null, route = "results") {
  const { map, w } = app;
  const V = w.mapviewer;
  const s = {
    detail: null, info: null, grids: new Map(), index: null, product: null, times: [], byTime: new Map(),
    georefs: [], t: -1, on: new Set(), alpha: 0.85, playing: false, placed: new Map(), dead: false,
    preload: 0, status: null, domains: [], gridsAt: new Map(), showing: 0,
  };

  // ---- the pieces on the map
  const title = h("div", { class: "runtitle" });
  const zoomIn = h("button", { type: "button", title: V.zoom_in, "aria-label": V.zoom_in }, "+");
  const zoomOut = h("button", { type: "button", title: V.zoom_out, "aria-label": V.zoom_out }, "−");
  zoomIn.addEventListener("click", () => map.zoomAt(map.width / 2, map.height / 2, 0.75));
  zoomOut.addEventListener("click", () => map.zoomAt(map.width / 2, map.height / 2, -0.75));
  const zoom = h("div", { class: "zoom float" }, zoomIn, zoomOut);
  const pickBtn = h("button", { class: "picker-btn", type: "button", "aria-haspopup": "listbox" });
  const quick = h("div", { class: "quick" });
  const domainBox = h("div", { class: "domains" });
  const alpha = h("input", { type: "range", min: "0.2", max: "1", step: "0.05", value: String(s.alpha), "aria-label": V.see_through });
  const pictureBtn = h("button", { class: "qchip", type: "button", title: V.picture_help }, V.picture);
  const top = h("div", { class: "mvtop" }, h("div", { class: "grp" }, title, pickBtn, quick), h("div", { class: "spacer" }),
    h("div", { class: "grp" }, domainBox, h("label", { class: "alpha" }, h("span", {}, V.see_through), alpha), pictureBtn,
      app.links || null));
  const picker = h("div", { class: "picker float", hidden: true });
  const playBtn = h("button", { class: "play", type: "button", "aria-label": V.play, title: V.play });
  const backBtn = h("button", { class: "step", type: "button", "aria-label": V.back, title: V.back }, "‹");
  const fwdBtn = h("button", { class: "step", type: "button", "aria-label": V.forward, title: V.forward }, "›");
  const when = h("div", { class: "when" });
  const track = h("div", { class: "track" });
  const live = h("div", { class: "liveline" });
  const timebar = h("div", { class: "timebar float" }, h("div", { class: "transport" }, backBtn, playBtn, fwdBtn), when, track, live);
  const legend = h("div", { class: "legend float", hidden: true });
  const readout = h("div", { class: "readout", hidden: true });
  const card = h("div", { class: "statecard float", hidden: true });
  // One-line notes sit in a row under the top bar at the right, never over the middle of the map.
  const loose = h("div", { class: "loose", hidden: true });
  const staleNote = h("div", { class: "loose", hidden: true }, h("span", { class: "lnote" }, V.stale));
  // the renderer drew without its map files: the pictures lack coastlines, borders and state lines
  const basemapNote = h("div", { class: "loose", hidden: true }, h("span", { class: "lnote" }, V.no_basemap));
  // A machine's run is seen here through its follower. When that ends early, what shows is the last state
  // received (running, with Stop), so the note says so and Resume updates starts following again.
  const resumeBtn = h("button", { class: "qchip", type: "button" }, V.resume_updates);
  const followNote = h("div", { class: "loose", hidden: true }, h("span", { class: "lnote" }, V.follow_lost), resumeBtn);
  // What the engine warned about for this attempt (a warm bubble above 10 K, say): it ran as configured, and the
  // warning stays here after it ends. The text is the engine's, set as text.
  const warningList = h("div", { class: "runwarnings" });
  const warningHead = h("summary", {});
  const warningNote = h("div", { class: "loose warnnote", hidden: true }, h("details", {}, warningHead, warningList));
  const notes = h("div", { class: "mvnotes" }, warningNote, followNote, basemapNote, staleNote, loose);
  const lightbox = h("div", { class: "lightbox", hidden: true });
  app.stage.append(top, zoom, picker, timebar, legend, readout, card, notes);
  // the full picture sits over everything, the list included
  document.body.append(lightbox);

  // The view the page framed the run in, kept while the person has not moved the map: a bar whose height settles
  // after that frame (the top bar wrapping once the grids and the map are named, the time bar once its times are
  // drawn) frames the run again in the space the bars now leave.
  let framed = null;
  function setInset(side, value) {
    if (map.inset && map.inset[side] === value) return;
    map.inset = { ...(map.inset || {}), [side]: value };
    if (framed && framed.cx === map.cx && framed.cy === map.cy && framed.z === map.z) frameMap();
  }

  // ---- the top bar fits the width it has
  // The map shortcuts are the first to go (the picker names the map and holds them all): the one that repeats the
  // picker's own choice, then the others from the end. A bar still too wide takes a second row, and the notes,
  // the zoom and the picker move down under it.
  const topFits = () => {
    const edge = top.getBoundingClientRect().right + 1;
    return top.scrollWidth <= top.clientWidth + 1 && [...top.children].every((el) => el.getBoundingClientRect().right <= edge);
  };
  function fitTop() {
    top.classList.remove("wrap");
    const chips = [...quick.children];
    chips.forEach((c) => { c.hidden = false; });
    const order = [...chips.filter((c) => c.classList.contains("on")), ...chips.filter((c) => !c.classList.contains("on")).reverse()];
    for (const chip of order) { if (topFits()) break; chip.hidden = true; }
    if (!topFits()) top.classList.add("wrap");
    const tall = Math.ceil(top.getBoundingClientRect().height);
    app.stage.style.setProperty("--mvtop-h", `${tall}px`);
    setInset("top", tall + 18);
  }
  const topWatch = typeof ResizeObserver === "function" ? new ResizeObserver(() => fitTop()) : null;
  if (topWatch) topWatch.observe(app.stage);
  // The time bar is one row on a wide window and three on a phone: the legend, the readout and the map's framing
  // sit above whatever height it has.
  function fitBottom() {
    const tall = Math.ceil(timebar.getBoundingClientRect().height);
    app.stage.style.setProperty("--mvbottom-h", `${tall}px`);
    setInset("bottom", tall + 18);
  }
  const bottomWatch = typeof ResizeObserver === "function" ? new ResizeObserver(() => fitBottom()) : null;
  if (bottomWatch) bottomWatch.observe(timebar);

  // ---- drawing
  const fieldLayer = {
    draw(ctx) {
      const tokens = [...s.placed.keys()].sort((a, b) => (domainNumber(a) || 0) - (domainNumber(b) || 0));
      for (const token of tokens) if (s.on.has(token)) drawPlaced(ctx, map, s.placed.get(token), s.alpha);
    },
  };
  const gridLayer = {
    above: true,
    draw(ctx) {
      for (const [id, grid] of s.grids) {
        const token = s.domains.find((d) => domainNumber(d) === id);
        const shown = token ? s.on.has(token) : true;
        const ring = grid.ring || (grid.ring = gridOutline(grid));
        // on each world copy the view shows, as the fields and the land are (screenRings)
        for (const pts of screenRings(ring, map)) {
          ctx.beginPath();
          pts.forEach(([x, y], k) => { if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
          ctx.closePath();
          ctx.setLineDash(shown ? [] : [5, 5]);
          ctx.strokeStyle = shown ? ink.grid : ink.gridOff;
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.setLineDash([]);
          let x = Infinity; let y = Infinity;
          for (const [sx, sy] of pts) { if (sy < y || (sy === y && sx < x)) { x = sx; y = sy; } }
          const text = domainLabel(token || `d${String(id).padStart(2, "0")}`, grid);
          ctx.font = "600 11px 'Segoe UI', system-ui, sans-serif";
          const tw = ctx.measureText(text).width;
          ctx.fillStyle = "rgba(255, 255, 255, 0.94)";
          ctx.fillRect(x, y - 19, tw + 12, 17);
          ctx.strokeStyle = ink.grid;
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, y - 18.5, tw + 11, 16);
          ctx.fillStyle = ink.grid;
          ctx.fillText(text, x + 6, y - 6.5);
        }
      }
      if (!s.grids.size && s.info && s.info.region) {
        // no grid on record yet (ready, or just started): the area that was asked for, which may run across the
        // 180th meridian (170 to -170) and is drawn as the 20 degrees it covers
        for (const pts of screenRings(s.info.region, map)) {
          ctx.beginPath();
          pts.forEach(([x, y], k) => { if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
          ctx.setLineDash([7, 5]);
          ctx.strokeStyle = ink.ask;
          ctx.lineWidth = 1.6;
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = "rgba(37, 99, 235, 0.05)";
          ctx.fill();
        }
      }
      if (s.hover) {
        ctx.strokeStyle = ink.grid;
        ctx.lineWidth = 1;
        const [x, y] = s.hover;
        ctx.beginPath(); ctx.moveTo(x - 8, y); ctx.lineTo(x + 8, y); ctx.moveTo(x, y - 8); ctx.lineTo(x, y + 8); ctx.stroke();
      }
    },
  };
  const removers = [map.add(fieldLayer), map.add(gridLayer)];
  // The line colours come from the page's own tokens, so the map reads as part of the page.
  const token = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  const ink = { grid: token("--mv-grid", "#171717"), gridOff: "rgba(23, 23, 23, 0.4)", ask: token("--aw-map-ask", "#2563eb") };

  // ---- the run's records
  async function loadRun() {
    const [detail, info] = await Promise.all([api.get(api.runPath(runId)), api.get(`${api.runPath(runId)}/map`)]);
    s.detail = detail;
    s.info = info;
    s.status = detail.status;
    setGridInfo(info);
    title.replaceChildren(h("span", { class: "rname", title: detail.title || detail.name }, detail.title || detail.name),
      h("span", { class: `st ${detail.status.state}` }, w.states[detail.status.state] || detail.status.state));
  }

  // A nest that follows a storm moves while the run goes: each frame's grids are where the nests stood when
  // that frame was written (placesAt), so a picture is cut along the grid it was drawn on.
  function setGridInfo(info) {
    s.info = info;
    s.gridsAt = new Map();
    s.grids = gridsFor(s.times[s.t] || null);
  }
  function gridsFor(valid) {
    const info = s.info || {};
    const places = placesAt(info.moves, info.domains, valid);
    const key = [...places].map(([id, [i, j]]) => `${id}:${i},${j}`).join(";");
    if (!s.gridsAt.has(key)) s.gridsAt.set(key, grids(info.projection, info.domains, places));
    return s.gridsAt.get(key);
  }

  function frameMap() {
    // Longitudes are read in one run near the first point's, so a domain across the 180th meridian is framed
    // as the few degrees it covers (lonLatBox), not as the whole world between 170 and -170.
    const pts = [];
    for (const grid of s.grids.values()) for (const p of gridOutline(grid, 8)) pts.push(p);
    let box = lonLatBox(pts);
    if (!box && s.info && s.info.region) box = lonLatBox(s.info.region);
    // the renderer's bounds read east from west through the meridian, the run a picture without a grid is
    // placed in (geo.js placement), so the frame and the field are one world copy
    if (!box) box = boundsBox(s.georefs);
    if (!box) return;
    const [west, south, east, north] = box;
    const padLon = (east - west) * 0.2;
    const padLat = (north - south) * 0.2;
    map.fit(west - padLon, south - padLat, east + padLon, north + padLat, 24);
    framed = { cx: map.cx, cy: map.cy, z: map.z };
  }

  async function loadIndex() {
    s.index = await api.get(`${api.runPath(runId)}/pictures`);
    const prev = new Set(s.domains);
    s.domains = [...s.index.domains].sort((a, b) => (domainNumber(a) || 0) - (domainNumber(b) || 0));
    for (const d of s.domains) if (!prev.has(d)) s.on.add(d);
    drawDomains();
    drawQuick();
    if (!s.product && wanted && s.index.products.includes(wanted)) s.product = wanted;
    if (!s.product || !s.index.products.includes(s.product)) {
      s.product = s.index.favourites[0] || s.index.products[0] || null;
    }
  }

  async function loadProduct(keepTime) {
    const product = s.product;
    // the map controls show once there is a map to control
    for (const el of [pickBtn, pictureBtn, alpha.parentElement]) el.hidden = !product;
    pickBtn.replaceChildren(h("span", { class: "pk" }, V.map), h("span", { class: "pv" }, product ? productName(product, app.looks) : "…"), h("span", { class: "chev" }));
    drawQuick();
    if (!product) { s.times = []; s.byTime = new Map(); drawTrack(); showEmpty(); return; }
    const list = await api.get(`${api.runPath(runId)}/pictures/list?product=${encodeURIComponent(product)}`);
    if (s.dead || product !== s.product) return;
    s.georefs = list.georefs || [];
    const byTime = new Map();
    for (const pic of list.pictures) {
      const key = pic.valid || pic.name;
      if (!byTime.has(key)) byTime.set(key, new Map());
      // A nest that retired and re-armed draws its last and its first frame at one time: the later life is shown.
      const had = byTime.get(key).get(pic.domain);
      if (!had || (pic.episode || "") >= (had.episode || "")) byTime.get(key).set(pic.domain, pic);
    }
    const oldKey = s.times[s.t];
    const atEnd = s.t === s.times.length - 1;
    s.times = [...byTime.keys()].sort();
    s.byTime = byTime;
    let t = s.times.length - 1;
    if (keepTime && oldKey && !atEnd && s.times.includes(oldKey)) t = s.times.indexOf(oldKey);
    else if (keepTime === "same" && oldKey && s.times.includes(oldKey)) t = s.times.indexOf(oldKey);
    drawTrack();
    await showTime(t);
    warm();
  }

  // ---- showing one time
  // Every showing has its own number. A picture that lands after a newer showing began (another map, another
  // time, a refresh) belongs to a selection the user has left, and is dropped: the valid time alone is shared
  // by every map at that time, so it cannot tell a late temperature picture from the radar one now selected.
  const current = (ticket, product, key) => !s.dead && ticket === s.showing && product === s.product && s.times[s.t] === key;
  // The time bar holds every grid's times, and grids are drawn on their own cadences (a parent hourly, its nest
  // every quarter hour): at some times no grid that is turned on has a picture.
  const shownAt = (k) => {
    const pics = s.byTime.get(s.times[k]);
    return !!pics && s.domains.some((token) => s.on.has(token) && pics.has(token));
  };
  // Play and the step buttons go to the next time, forward or back, at which a grid turned on has a picture (the
  // next time of all when none has one).
  function step(dir) {
    const n = s.times.length;
    for (let i = 1; i <= n; i++) { const k = (((s.t + dir * i) % n) + n) % n; if (shownAt(k)) return k; }
    return (((s.t + dir) % n) + n) % n;
  }
  // The time nearest the one shown at which a grid turned on has a picture, or -1.
  function nearestShown() {
    for (let d = 1; d < s.times.length; d++) for (const k of [s.t - d, s.t + d]) if (k >= 0 && k < s.times.length && shownAt(k)) return k;
    return -1;
  }
  async function showTime(t) {
    const ticket = ++s.showing;
    if (!s.times.length) { s.t = -1; showEmpty(); return; }
    s.t = Math.max(0, Math.min(s.times.length - 1, t));
    const key = s.times[s.t];
    const product = s.product;
    s.grids = gridsFor(key);
    drawWhen();
    drawReadout();
    drawTrack();
    card.hidden = !cardWanted();
    const pics = s.byTime.get(key) || new Map();
    const jobs = [];
    for (const token of s.domains) {
      const pic = pics.get(token);
      if (!pic) { s.placed.delete(token); continue; }
      jobs.push(place(pic).then((placed) => {
        if (!current(ticket, product, key)) return;
        s.placed.set(token, placed);
      }));
    }
    await Promise.all(jobs);
    if (!current(ticket, product, key)) return;
    // the outline and the readout follow the footprint each picture was placed on
    const shown = new Map(s.grids);
    for (const placed of s.placed.values()) if (placed && placed.bent && placed.grid) shown.set(placed.grid.id, placed.grid);
    s.grids = shown;
    drawLegendAndLoose();
    drawReadout();
    map.redraw();
  }

  function place(pic) {
    const n = domainNumber(pic.domain);
    let grid = gridsFor(pic.valid || null).get(n) || null;
    let georef = pic.geo === null || pic.geo === undefined ? null : s.georefs[pic.geo];
    if (!georef) georef = borrowed(pic, grid);
    // a picture whose record puts it somewhere else than the grid on record is placed where the renderer drew it
    if (georef && grid) grid = gridForPicture(grid, georef);
    return placePicture(api.filePath(runId, pic.path), georef, grid)
      .then((placed) => (placed ? { ...placed, grid } : placed)).catch(() => null);
  }

  // The renderer's record can leave frames out. The same grid and look draws the same frame, so a frame
  // without a record takes the nearest frame's that has one, moved with the nest if the nest moved between.
  function borrowed(pic, grid) {
    if (!grid) return null;
    const k = s.times.indexOf(pic.valid || pic.name);
    let best = null;
    for (let d = 1; d < s.times.length && !best; d++) {
      for (const at of [k - d, k + d]) {
        const other = at >= 0 && at < s.times.length ? (s.byTime.get(s.times[at]) || new Map()).get(pic.domain) : null;
        if (other && other.geo !== null && other.geo !== undefined && s.georefs[other.geo]) { best = other; break; }
      }
    }
    if (!best) return null;
    const from = gridsFor(best.valid || null).get(grid.id);
    return borrowGeoref(s.georefs[best.geo], from, grid);
  }

  // Bend every frame of this map in the background, one at a time, so Play and the scrubber are instant.
  async function warm() {
    const ticket = ++s.preload;
    const order = s.times.map((_, k) => (s.t - k + s.times.length) % s.times.length);
    for (const k of order) {
      if (s.dead || ticket !== s.preload) return;
      const pics = s.byTime.get(s.times[k]);
      if (!pics) continue;
      for (const token of s.domains) { const pic = pics.get(token); if (pic && s.on.has(token)) await place(pic); }
    }
  }

  function drawLegendAndLoose() {
    const shown = s.domains.filter((token) => s.on.has(token) && s.placed.get(token));
    const placed = shown.find((token) => s.placed.get(token).bent && s.placed.get(token).legend);
    if (placed) { legend.replaceChildren(s.placed.get(placed).legend); legend.hidden = false; } else legend.hidden = true;
    const unplaced = shown.filter((token) => !s.placed.get(token).bent);
    if (unplaced.length) {
      const names = unplaced.map((token) => domainLabel(token, s.grids.get(domainNumber(token)))).join(", ");
      const open = h("button", { class: "qchip", type: "button" }, V.show_picture);
      open.addEventListener("click", () => pictureBtn.click());
      loose.replaceChildren(h("span", { class: "lnote" }, fill(V.not_placed, { grids: names })), open);
      loose.hidden = false;
    } else loose.hidden = true;
  }

  function showEmpty() {
    s.placed.clear();
    legend.hidden = true;
    loose.hidden = true;
    drawWhen();
    card.hidden = !cardWanted();
    map.redraw();
  }

  // ---- the top bar
  function drawQuick() {
    if (!s.index) return;
    const favs = s.index.favourites.slice(0, QUICK);
    quick.replaceChildren(...favs.map((p) => {
      const b = h("button", { class: `qchip${p === s.product ? " on" : ""}`, type: "button" }, productName(p, app.looks));
      b.addEventListener("click", () => setProduct(p));
      return b;
    }));
    fitTop();
  }

  function drawDomains() {
    domainBox.replaceChildren();
    if (s.domains.length < 2) return;
    domainBox.append(h("span", { class: "dlabel" }, V.grids));
    for (const token of s.domains) {
      const grid = s.grids.get(domainNumber(token));
      const drawn = ((s.index && s.index.by_domain) || {})[token];
      const b = h("button", { class: `dchip${s.on.has(token) ? " on" : ""}`, type: "button", "aria-pressed": s.on.has(token) ? "true" : "false",
        title: drawn ? fill(V.grid_pictures, { grid: domainLabel(token, grid), count: drawn.count.toLocaleString("en-US") }) : null }, domainLabel(token, grid));
      b.addEventListener("click", () => {
        if (s.on.has(token)) s.on.delete(token); else s.on.add(token);
        drawDomains();
        showTime(s.t);
        warm();
      });
      domainBox.append(b);
    }
    fitTop();
  }

  function openPicker() {
    if (!s.index) return;
    const search = h("input", { type: "search", placeholder: V.find, "aria-label": V.find, spellcheck: "false" });
    const body = h("div", { class: "pickbody" });
    const groups = groupProducts(s.index.products, s.index.favourites, app.looks, V.groups);
    const draw = () => {
      const q = search.value.trim().toLowerCase();
      body.replaceChildren();
      let count = 0;
      for (const g of groups) {
        const items = g.products.filter((p) => !q || p.toLowerCase().includes(q) || productName(p, app.looks).toLowerCase().includes(q));
        if (!items.length) continue;
        count += items.length;
        body.append(h("div", { class: "pgroup" }, g.name), ...items.map((p) => {
          const b = h("button", { class: `pitem${p === s.product ? " on" : ""}`, type: "button", title: p }, productName(p, app.looks));
          b.addEventListener("click", () => { closePicker(); setProduct(p); });
          return b;
        }));
      }
      if (!count) body.append(h("p", { class: "empty" }, V.nothing_found));
    };
    search.addEventListener("input", draw);
    search.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { const first = body.querySelector(".pitem"); if (first) first.click(); }
      if (ev.key === "Escape") closePicker();
    });
    draw();
    picker.replaceChildren(search, body);
    picker.style.left = `${Math.round(pickBtn.getBoundingClientRect().left - app.stage.getBoundingClientRect().left)}px`;
    picker.hidden = false;
    search.focus();
    const on = body.querySelector(".pitem.on");
    if (on) on.scrollIntoView({ block: "center" });
  }
  function closePicker() { picker.hidden = true; }
  pickBtn.addEventListener("click", () => (picker.hidden ? openPicker() : closePicker()));
  const outside = (ev) => { if (!picker.hidden && !picker.contains(ev.target) && !pickBtn.contains(ev.target)) closePicker(); };
  document.addEventListener("pointerdown", outside);

  async function setProduct(p) {
    if (p === s.product) return;
    s.product = p;
    s.showing += 1;
    // the address names the map, so a reload or a copied link opens this one (no new page load)
    history.replaceState(null, "", `#/${route}/${encodeURIComponent(runId)}/${encodeURIComponent(p)}`);
    s.placed.clear();
    try { await loadProduct("same"); } catch (err) { notice(errorText(err), "warn"); }
  }

  alpha.addEventListener("input", () => { s.alpha = Number(alpha.value); map.redraw(); });

  pictureBtn.addEventListener("click", () => {
    const pics = s.byTime.get(s.times[s.t]);
    if (!pics) return;
    const tokens = s.domains.filter((d) => pics.get(d));
    const img = h("img", { alt: "" });
    const tabs = h("div", { class: "lbtabs", hidden: tokens.length < 2 });
    const choose = (token) => {
      img.src = api.filePath(runId, pics.get(token).path);
      for (const b of tabs.children) b.classList.toggle("on", b.dataset.token === token);
    };
    for (const token of tokens) {
      const b = h("button", { class: "dchip", type: "button", dataset: { token } }, domainLabel(token, s.grids.get(domainNumber(token))));
      b.addEventListener("click", (ev) => { ev.stopPropagation(); choose(token); });
      tabs.append(b);
    }
    const close = h("button", { class: "btn small", type: "button" }, V.close);
    const d = parseTime(s.times[s.t]);
    const caption = h("div", { class: "lbcap" }, productName(s.product, app.looks), d ? ` · ${longTime(d)}` : "");
    lightbox.replaceChildren(h("div", { class: "lbframe" }, h("div", { class: "lbhead" }, caption, tabs, close), img));
    const shown = [...tokens].reverse().find((d) => s.on.has(d)) || tokens[0];
    if (shown) choose(shown);
    lightbox.hidden = false;
  });
  lightbox.addEventListener("click", (ev) => { if (ev.target === lightbox || ev.target.closest(".lbhead .btn")) lightbox.hidden = true; });

  // ---- the time bar
  function startDate() {
    return parseTime(s.status && s.status.start_time) || parseTime(s.times[0]);
  }
  function span() {
    const t0 = startDate();
    const secs = s.status && s.status.run_seconds;
    let a = t0 ? t0.getTime() : null;
    let b = a !== null && secs ? a + secs * 1000 : null;
    const first = parseTime(s.times[0]);
    const last = parseTime(s.times[s.times.length - 1]);
    if (a === null && first) a = first.getTime();
    if (b === null && last) b = last.getTime();
    if (first) a = Math.min(a, first.getTime());
    if (last) b = Math.max(b, last.getTime());
    return [a, b];
  }

  function drawWhen() {
    const key = s.times[s.t];
    const d = parseTime(key);
    const t0 = startDate();
    if (!d) {
      when.replaceChildren(key ? h("div", { class: "big" }, key) : h("div", { class: "small wide" }, V.no_pictures));
      return;
    }
    const lead = t0 ? Math.round((d - t0) / 360000) / 10 : null;
    when.replaceChildren(h("div", { class: "big" }, longTime(d)),
      h("div", { class: "small" }, lead === null ? "" : fill(V.hour, { hour: lead }), s.product ? ` · ${productName(s.product, app.looks)}` : ""));
  }

  function drawTrack() {
    track.replaceChildren();
    const [a, b] = span();
    // a lone frame with no span to place it in sits at the start of the bar, where a run's first frame goes
    const pos = (ms) => (b > a ? (ms - a) / (b - a) : 0);
    const rail = h("div", { class: "rail" });
    track.append(rail);
    if (s.status && s.status.state === "running" && s.status.percent !== null && s.status.percent !== undefined) {
      const fillEl = h("div", { class: "runfill" });
      fillEl.style.width = `${Math.max(0, Math.min(100, s.status.percent))}%`;
      rail.append(fillEl);
    }
    s.times.forEach((key, k) => {
      const d = parseTime(key);
      const x = d ? pos(d.getTime()) : (s.times.length > 1 ? k / (s.times.length - 1) : 0);
      const tick = h("button", { class: `tick${k === s.t ? " on" : ""}`, type: "button", title: d ? longTime(d) : key, "aria-label": d ? longTime(d) : key });
      tick.style.left = `${(x * 100).toFixed(3)}%`;
      tick.addEventListener("click", (ev) => { ev.stopPropagation(); stop(); showTime(k); });
      rail.append(tick);
    });
    // hour marks under the rail
    if (a !== null && b > a) {
      const hours = (b - a) / 3600000;
      const step = hours <= 12 ? 1 : hours <= 48 ? 6 : hours <= 120 ? 12 : 24;
      for (let hr = 0; hr <= hours + 1e-9; hr += step) {
        const mark = h("span", { class: "hmark" }, `${hr}`);
        mark.style.left = `${((hr * 3600000) / (b - a)) * 100}%`;
        rail.append(mark);
      }
    }
    playBtn.classList.toggle("on", s.playing);
    playBtn.textContent = s.playing ? "❚❚" : "▶";
    playBtn.title = s.playing ? V.pause : V.play;
    const none = s.times.length < 2;
    playBtn.disabled = none; backBtn.disabled = none; fwdBtn.disabled = none;
  }

  // drag along the rail to scrub
  let scrubbing = false;
  const scrubTo = (ev) => {
    const r = track.getBoundingClientRect();
    const f = (ev.clientX - r.left) / Math.max(1, r.width);
    const [a, b] = span();
    let best = 0; let bestD = Infinity;
    s.times.forEach((key, k) => {
      const d = parseTime(key);
      const x = d && b > a ? (d.getTime() - a) / (b - a) : (s.times.length > 1 ? k / (s.times.length - 1) : 0);
      if (Math.abs(x - f) < bestD) { bestD = Math.abs(x - f); best = k; }
    });
    if (best !== s.t) showTime(best);
  };
  track.addEventListener("pointerdown", (ev) => { if (!s.times.length) return; scrubbing = true; stop(); track.setPointerCapture(ev.pointerId); scrubTo(ev); });
  track.addEventListener("pointermove", (ev) => { if (scrubbing) scrubTo(ev); });
  track.addEventListener("pointerup", () => { scrubbing = false; });

  let playTimer = null;
  function stop() { s.playing = false; clearTimeout(playTimer); drawTrack(); }
  async function tickPlay() {
    if (!s.playing || s.dead) return;
    await showTime(step(1));
    // a longer rest before it starts over
    playTimer = setTimeout(tickPlay, step(1) <= s.t ? PLAY_MS * 2.2 : PLAY_MS);
  }
  playBtn.addEventListener("click", () => {
    if (s.playing) { stop(); return; }
    if (s.times.length < 2) return;
    s.playing = true;
    drawTrack();
    tickPlay();
  });
  backBtn.addEventListener("click", () => { stop(); showTime(step(-1)); });
  fwdBtn.addEventListener("click", () => { stop(); showTime(step(1)); });
  const keys = (ev) => {
    if (ev.target.closest && ev.target.closest("input, select, textarea")) return;
    if (ev.key === "ArrowRight") { ev.preventDefault(); fwdBtn.click(); }
    else if (ev.key === "ArrowLeft") { ev.preventDefault(); backBtn.click(); }
    else if (ev.key === " ") { ev.preventDefault(); playBtn.click(); }
    else if (ev.key === "Escape") { closePicker(); lightbox.hidden = true; }
  };
  window.addEventListener("keydown", keys);

  // ---- a finer forecast inside this one: offered once the run has finished or was stopped on this computer
  let downscaling = null;
  const dsBtn = h("button", { type: "button", title: V.downscale_help, hidden: true }, V.downscale);
  if (app.links) app.links.append(dsBtn);
  dsBtn.addEventListener("click", async () => {
    if (downscaling) { downscaling.close(); return; }
    dsBtn.disabled = true;
    try {
      downscaling = await openDownscale({ map, stage: app.stage, runId, words: w.downscale,
        onclose: () => { downscaling = null; dsBtn.classList.remove("on"); } });
      dsBtn.classList.add("on");
    } catch (err) {
      notice(errorText(err), "warn");
    } finally {
      dsBtn.disabled = false;
    }
  });

  // ---- run state: live line, stop, start, errors
  function drawLive() {
    const st = s.status;
    live.replaceChildren();
    if (!st) return;
    dsBtn.hidden = !["finished", "stopped"].includes(st.state) || !!(s.detail && s.detail.remote);
    const pill = title.querySelector(".st");
    if (pill) { pill.className = `st ${st.state}`; pill.textContent = w.states[st.state] || st.state; }
    if (st.state !== "running") return;
    const words = [];
    const forecasting = !st.stage || st.stage === "forecast";
    if (!forecasting) words.push(w.stages[st.stage] || st.stage);
    // While it prepares: the step, the grid, the download and how long it has gone, not "forecast hour 0" held
    // until the first model step. A chained forecast adds how many boundary times are ready.
    const prep = prepLine(st, w.preparation);
    if (!forecasting) words.push(...prep);
    if (forecasting && st.run_seconds && st.model_seconds !== null && st.model_seconds !== undefined) words.push(hourWords(st.model_seconds, st.run_seconds, V.model_hour));
    if (st.seconds_left !== null && st.seconds_left !== undefined) words.push(fill(V.left, { left: leftWords(st.seconds_left) }));
    if (forecasting) words.push(...prep);
    // every grid is drawn as its frames land, so a nested run names its grids and the one that sets the pace
    const pace = paceLine(st, V);
    if (pace) words.push(pace);
    const stopBtn = h("button", { class: "btn danger small", type: "button" }, V.stop);
    stopBtn.addEventListener("click", async () => {
      if (!window.confirm(V.stop_confirm)) return;
      stopBtn.disabled = true;
      try {
        const reply = await api.post(`${api.runPath(runId)}/stop`);
        notice(reply.message || V.stopped);
      } catch (err) {
        notice(errorText(err), "stop");
        stopBtn.disabled = false;
      }
    });
    live.append(h("div", { class: "livewords" }, st.follow_lost ? null : h("span", { class: "pulse" }), words.join(" · ")),
      bar(st.percent, true), stopBtn);
  }

  function cardWanted() {
    const st = s.status;
    if (!st) return false;
    card.replaceChildren();
    // a stale run with pictures says so in the note row; the map stays clear
    staleNote.hidden = !(st.state === "stale" && s.times.length);
    basemapNote.hidden = !st.basemap_missing;
    followNote.hidden = !st.follow_lost;
    drawWarnings(st.library_warnings || []);
    if (st.state === "ready") {
      const start = actionButton(V.start, { kind: "primary", request: () => ({ path: `${api.runPath(runId)}/start` }), onclick: async () => {
        start.button.disabled = true;
        try {
          const reply = await api.post(`${api.runPath(runId)}/start`);
          notice(reply.message);
          await refreshAll();
          follow();
        } catch (err) {
          notice(errorText(err), "stop");
          start.button.disabled = false;
        }
      } });
      card.append(h("p", {}, V.ready_intro), start);
      return true;
    }
    const end = st.end || {};
    if (st.state === "failed") {
      card.classList.add("bad");
      // core's append skips an absent line; the DOM's own append would print it as the word "null".
      append(card, [h("p", { class: "strong" }, V.failed), end.message ? h("p", {}, end.message) : null,
        end.remedy ? h("p", { class: "note" }, end.remedy) : null,
        h("a", { href: `#/explore/${encodeURIComponent(runId)}` }, V.open_files)]);
      return true;
    }
    card.classList.remove("bad");
    // Every grid turned off, or a time only a grid turned off has a picture at: the map would show outlines and
    // nothing else, and say nothing.
    if (s.times.length && s.t >= 0 && s.domains.length && !shownAt(s.t)) {
      if (!s.domains.some((token) => s.on.has(token))) {
        card.append(h("p", {}, V.no_grid_on));
        return true;
      }
      card.append(h("p", {}, V.no_selected_picture));
      const k = nearestShown();
      if (k >= 0) {
        const jump = h("button", { class: "btn small", type: "button" }, fill(V.show_time, { time: longTime(s.times[k]) || s.times[k] }));
        jump.addEventListener("click", () => { stop(); showTime(k); });
        card.append(jump);
      }
      return true;
    }
    if (st.state === "stale" && s.times.length) return false;
    if (st.state === "stale") { card.append(h("p", {}, V.stale)); return true; }
    if (!s.times.length) {
      if (s.index && s.index.count && !s.product) return false;
      card.append(h("p", {}, V.no_pictures));
      if (!s.grids.size && s.index && s.index.count) card.append(h("p", { class: "note" }, V.no_grid));
      return true;
    }
    return false;
  }

  let warningsShown = "";
  function drawWarnings(list) {
    warningNote.hidden = !list.length;
    const key = JSON.stringify(list);
    if (key === warningsShown) return;
    warningsShown = key;
    warningHead.textContent = fill(V.run_warnings, { count: list.length });
    warningList.replaceChildren(...list.map((item) => h("p", {}, item.message, item.detail ? h("span", { class: "dim" }, ` ${item.detail}`) : null)));
  }

  resumeBtn.addEventListener("click", async () => {
    resumeBtn.disabled = true;
    try {
      const reply = await api.post(`${api.runPath(runId)}/follow`);
      if (s.dead) return;
      notice(reply.message || V.resumed);
      await refreshAll();
      follow();
    } catch (err) {
      if (!s.dead) notice(errorText(err), "stop");
    } finally {
      resumeBtn.disabled = false;
    }
  });

  // ---- hover
  // The readout is worked out again whenever the frame changes (the arrow keys, Play, the scrubber) as well as
  // when the pointer moves: a nest that moved puts another cell under the same point, at another valid time.
  function drawReadout() {
    if (!s.pointer || s.dead) return;
    const [lon, lat] = s.pointer;
    const bits = [latLonWords(lat, lon)];
    const ids = [...s.grids.keys()].sort((a, b) => b - a);
    for (const id of ids) {
      const g = s.grids.get(id);
      const [i, j] = g.toIndex(lat, lon);
      if (i >= -0.5 && j >= -0.5 && i <= g.nx - 0.5 && j <= g.ny - 0.5) {
        bits.push(fill(V.column_row, { grid: `d${String(id).padStart(2, "0")}`, i: Math.round(i) + 1, j: Math.round(j) + 1 }));
        break;
      }
    }
    const d = parseTime(s.times[s.t]);
    if (d) bits.push(fill(V.valid, { time: longTime(d) }));
    readout.textContent = bits.join("  ·  ");
    readout.hidden = false;
  }
  const offHover = map.on("hover", (lon, lat, x, y) => {
    if (s.dead) return;
    s.pointer = [lon, lat];
    drawReadout();
    s.hover = [x, y];
    map.redraw();
  });
  const offLeave = map.on("leave", () => { readout.hidden = true; s.hover = null; s.pointer = null; map.redraw(); });

  // ---- live updates
  let unfollow = null;
  let refreshTimer = null;
  let pollTimer = null;
  function soon() {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => refreshPictures().catch(() => {}), REFRESH_MS);
  }
  async function refreshPictures() {
    if (s.dead) return;
    const hadProduct = s.product;
    await loadIndex();
    await loadProduct(hadProduct === s.product ? true : "same");
  }
  async function refreshAll() {
    if (s.dead) return;
    await loadRun();
    if (s.dead) return;
    const hadGrids = s.grids.size;
    await refreshPictures();
    if (s.dead) return;
    if (!hadGrids && s.grids.size) frameMap();
    drawLive();
    card.hidden = !cardWanted();
    if (app.onchange) app.onchange();
  }
  // A render worker draws on its own clock: its pictures keep arriving after the forecast has finished, until
  // its own state ends (finished or failed), so the map keeps reading until then.
  const drawing = () => !!(s.detail && s.detail.render && ["requested", "rendering"].includes(s.detail.render.state));
  let polling = false;
  async function poll() {
    if (s.dead || polling) return;
    if (drawing()) {
      // The run's record says when the drawing ends; the pass that reads its end reads the last pictures too.
      polling = true;
      try { await refreshAll(); } catch (_) { /* the next poll tries again */ } finally { polling = false; }
    } else if (s.status && s.status.state === "running") soon();
  }
  function follow() {
    if (unfollow) return;
    unfollow = api.follow(runId, {
      status: (st) => {
        const was = s.status && s.status.state;
        s.status = { ...(s.status || {}), ...st };
        drawLive();
        drawTrack();
        card.hidden = !cardWanted();
        if (was !== st.state) { soon(); if (app.onchange) app.onchange(); }
      },
      resolved_plan: async () => {
        try {
          const had = s.grids.size;
          setGridInfo(await api.get(`${api.runPath(runId)}/map`));
          if (!had && s.grids.size) frameMap();
          map.redraw();
        } catch (_) { /* the next poll tries again */ }
      },
      output_committed: soon,
      warning: (data) => {
        if (!data || data.code !== "render_basemap_missing") return;
        s.status = { ...(s.status || {}), basemap_missing: true };
        basemapNote.hidden = false;
      },
      first_products_ready: soon,
      // a nest's frame drawn while the run goes: its grid chip and its picture come in now, not at the next poll
      live_products_ready: soon,
      stage_finished: soon,
      completed: soon,
      failed: soon,
    });
    pollTimer = setInterval(poll, POLL_MS);
  }

  // ---- open
  await loadRun();
  try { await loadIndex(); } catch (err) { notice(errorText(err), "warn"); }
  frameMap();
  drawLive();
  try { await loadProduct(false); } catch (err) { notice(errorText(err), "warn"); }
  if (!s.grids.size && s.georefs.length) frameMap();
  if (["running", "ready"].includes(s.status.state) || drawing() || s.status.follow_lost) follow();

  return () => {
    s.dead = true;
    if (downscaling) downscaling.close();
    stop();
    s.preload += 1;
    clearTimeout(refreshTimer);
    clearInterval(pollTimer);
    if (unfollow) unfollow();
    removers.forEach((r) => r());
    if (topWatch) topWatch.disconnect();
    if (bottomWatch) bottomWatch.disconnect();
    offHover();
    offLeave();
    window.removeEventListener("keydown", keys);
    document.removeEventListener("pointerdown", outside);
    lightbox.remove();
  };
}
