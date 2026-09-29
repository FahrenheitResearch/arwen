// Downscale: a finer forecast inside a finished one, placed on its map. Click the map for the finer forecast's
// centre or drag a box for its extent, choose how fine and how long, then Review: the engine plans it (its
// --dry-run) and the grid it would run is drawn on the map with its price on the card. Start runs exactly what
// was reviewed and opens the new forecast. Every button is one request to /api/runs/RUN/downscale.
//
//   openDownscale({map, stage, runId, words, onclose})  -> {close()}   words: the copy's screens.downscale

import { h, append, fill, spacing } from "./core.js";
import * as api from "./api.js";
import { notice, errorText, go, runRoute } from "./router.js";
import { liveCommand } from "./command.js";

function place(lat, lon) {
  return `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(2)}° ${lon >= 0 ? "E" : "W"}`;
}

// A box's size as the page shows it (the server sizes the grid from the box itself).
function boxKm(b) {
  const r = Math.PI / 180;
  const mid = (b.south + b.north) / 2;
  const width = 6371 * Math.cos(mid * r) * Math.abs(b.east - b.west) * r;
  const height = 6371 * Math.abs(b.north - b.south) * r;
  return [Math.round(width), Math.round(height)];
}

export async function openDownscale({ map, stage, runId, words, onclose }) {
  const W = words;
  const path = `${api.runPath(runId)}/downscale`;
  const facts = await api.get(path);
  const s = { point: null, box: null, review: null, reviewed: "", busy: false, dead: false };

  const panel = h("div", { class: "dspanel float", role: "dialog", "aria-label": W.title });
  const say = h("p", { class: "note" });
  const where = h("p", { class: "dswhere" });
  const drawBtn = h("button", { class: "btn small", type: "button" }, W.draw);
  const closeBtn = h("button", { class: "btn small", type: "button" }, W.close);
  const out = h("div", { class: "dsreview" });

  const gridPick = h("select", { class: "input" });
  for (const d of facts.domains || []) {
    gridPick.append(h("option", { value: String(d.id), selected: d.id === facts.domain ? "selected" : null },
      fill(W.grid_option, { id: String(d.id).padStart(2, "0"), km: d.dx_km ? spacing(d.dx_km) : W.unknown_spacing })));
  }
  const finePick = h("select", { class: "input" });
  const hoursIn = h("input", { class: "input num", type: "number", min: "0.1", step: "0.5", inputmode: "decimal",
    placeholder: facts.window_hours ? fill(W.hours_all, { hours: facts.window_hours }) : "" });
  const cardPick = h("select", { class: "input" }, h("option", { value: "" }, W.card_measure),
    (facts.cards || []).map((c) => h("option", { value: c }, fill(W.card_option, { gb: parseInt(c, 10) }))));
  const picturesPick = h("select", { class: "input" }, h("option", { value: "all" }, W.pictures_all),
    h("option", { value: "none" }, W.pictures_none));
  const nameIn = h("input", { class: "input", spellcheck: "false", autocomplete: "off", value: facts.name || "" });

  function domainRow() { return (facts.domains || []).find((d) => String(d.id) === gridPick.value) || null; }
  // How often the finer forecast's edges are driven: the cadence of the chosen grid's saved frames, which is the
  // boundary cadence the server asks the engine for.
  const edges = h("p", { class: "note" });
  function drawEdges() {
    const row = domainRow();
    const minutes = row && row.interval_s ? row.interval_s / 60 : 0;
    edges.textContent = minutes > 0
      ? fill(W.edges, { minutes: Number.isInteger(minutes) ? String(minutes) : minutes.toFixed(1) }) : "";
  }
  function drawFine() {
    const row = domainRow();
    const was = finePick.value || String(facts.ratio);
    finePick.replaceChildren();
    for (let r = facts.ratio_range[0]; r <= Math.min(facts.ratio_range[1], 8); r++) {
      const km = row && row.dx_km ? spacing(row.dx_km / r) : null;
      finePick.append(h("option", { value: String(r), selected: String(r) === was ? "selected" : null },
        km ? fill(W.fine_option, { ratio: r, km }) : fill(W.fine_option_plain, { ratio: r })));
    }
  }
  drawFine();

  const reviewBtn = h("button", { class: "btn primary", type: "button" }, W.review);
  const startBtn = h("button", { class: "btn", type: "button", disabled: "" }, W.start);

  function payload(mode) {
    if (!s.point && !s.box) throw new Error(W.need_place);
    const body = { mode, domain: Number(gridPick.value), ratio: Number(finePick.value), products: picturesPick.value };
    if (s.box) body.box = s.box; else { body.lat = s.point[0]; body.lon = s.point[1]; }
    if (hoursIn.value.trim()) body.hours = Number(hoursIn.value);
    if (cardPick.value) body.card = cardPick.value;
    if (nameIn.value.trim()) body.name = nameIn.value.trim();
    return body;
  }
  // What a review stands for: the request without its mode, so a plan and the run it approves compare equal.
  const settings = (body) => { const b = { ...body }; delete b.mode; return JSON.stringify(b); };
  const signature = () => { try { return settings(payload("run")); } catch (_) { return ""; } };
  const command = liveCommand(() => {
    try { return { path, body: payload("run") }; } catch (err) { return { error: err.message }; }
  }, { watch: [gridPick, finePick, hoursIn, cardPick, picturesPick, nameIn] });

  function refresh() {
    where.textContent = s.box
      ? fill(W.box, { width: boxKm(s.box)[0], height: boxKm(s.box)[1], place: place((s.box.south + s.box.north) / 2, (s.box.west + s.box.east) / 2) })
      : s.point ? fill(W.centre, { place: place(s.point[0], s.point[1]) }) : W.place_note;
    const current = s.review && s.reviewed === signature();
    startBtn.disabled = !current || s.busy;
    startBtn.className = current ? "btn primary" : "btn";
    reviewBtn.className = current ? "btn" : "btn primary";
    reviewBtn.disabled = s.busy || (!s.point && !s.box);
    if (s.review && !current) say.textContent = W.changed;
    if (command.refresh) command.refresh();
    map.redraw();
  }
  for (const el of [gridPick, finePick, hoursIn, cardPick, picturesPick, nameIn]) {
    el.addEventListener("input", refresh);
    el.addEventListener("change", refresh);
  }
  gridPick.addEventListener("change", drawFine);
  gridPick.addEventListener("change", drawEdges);
  drawEdges();

  function showReview(reply) {
    const r = reply.review || {};
    const lines = [h("p", { class: "strong" }, r.words || "")];
    if (r.streaming && r.streaming.why && r.streaming.mode !== "resident") lines.push(h("p", { class: "note" }, r.streaming.why));
    if (r.regime) lines.push(h("p", { class: "note" }, r.regime));
    if ((r.warnings || []).length) lines.push(h("p", { class: "dim" }, W.warnings), h("ul", { class: "dswarn" }, r.warnings.map((t) => h("li", {}, t))));
    out.replaceChildren();
    append(out, lines);
  }

  reviewBtn.addEventListener("click", async () => {
    let body;
    try { body = payload("plan"); } catch (err) { say.textContent = err.message; return; }
    // The review answers the settings it was asked about. An edit made while it runs changes the form, not the
    // settings reviewed, so Start stays off until that edit has its own review.
    const reviewing = settings(body);
    s.busy = true; s.review = null; out.replaceChildren(); say.textContent = W.reviewing; refresh();
    try {
      const reply = await api.post(path, body);
      if (s.dead) return;
      s.review = reply.review;
      s.reviewed = reviewing;
      say.textContent = W.start_note;
      showReview(reply);
    } catch (err) {
      if (s.dead) return;
      say.textContent = errorText(err);
    } finally {
      s.busy = false;
      if (!s.dead) refresh();
    }
  });

  startBtn.addEventListener("click", async () => {
    if (!s.review || s.reviewed !== signature()) { say.textContent = W.changed; return; }
    s.busy = true; refresh();
    try {
      const reply = await api.post(path, payload("run"));
      notice(reply.message);
      close();
      go(runRoute("watch", reply.run));
    } catch (err) {
      if (s.dead) return;
      say.textContent = errorText(err);
      s.busy = false;
      refresh();
    }
  });

  // ---- placing it on the map: a click is its centre, a drag after Draw a box is its extent
  const offClick = map.on("click", (lon, lat) => {
    if (s.dead) return;
    s.point = [lat, lon]; s.box = null;
    refresh();
  });
  drawBtn.addEventListener("click", () => {
    say.textContent = W.drawing;
    map.drawBox((box) => {
      map.drawBox(null);
      if (s.dead) return;
      s.box = box; s.point = null;
      say.textContent = "";
      refresh();
    });
  });
  const ink = getComputedStyle(document.documentElement).getPropertyValue("--aw-map-ask").trim() || "#2563eb";
  const layer = {
    above: true,
    draw(ctx) {
      ctx.strokeStyle = ink;
      if (s.box) {
        const [x0, y0] = map.screen(s.box.west, s.box.north);
        const [x1, y1] = map.screen(s.box.east, s.box.south);
        ctx.setLineDash([7, 5]); ctx.lineWidth = 1.6;
        ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
        ctx.setLineDash([]);
      } else if (s.point) {
        const [x, y] = map.screen(s.point[1], s.point[0]);
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x - 9, y); ctx.lineTo(x + 9, y); ctx.moveTo(x, y - 9); ctx.lineTo(x, y + 9); ctx.stroke();
      }
      // the grid the engine planned, once reviewed: [lat, lon] corners around its edge
      const ring = s.review && s.reviewed === signature() && Array.isArray(s.review.outline) ? s.review.outline : null;
      if (ring && ring.length >= 3) {
        ctx.lineWidth = 2;
        ctx.beginPath();
        ring.forEach(([lat, lon], k) => { const [x, y] = map.screen(lon, lat); if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
        ctx.closePath();
        ctx.stroke();
        ctx.fillStyle = "rgba(37, 99, 235, 0.07)";
        ctx.fill();
      }
    },
  };
  const offLayer = map.add(layer);

  function close() {
    if (s.dead) return;
    s.dead = true;
    map.drawBox(null);
    offClick();
    offLayer();
    panel.remove();
    if (onclose) onclose();
  }
  closeBtn.addEventListener("click", close);

  const head = h("div", { class: "dshead" }, h("h3", {}, W.title), closeBtn);
  if (!facts.eligible) {
    append(panel, [head, h("p", { class: "strong" }, facts.reason || W.not_offered), facts.fix ? h("p", { class: "note" }, facts.fix) : null]);
  } else {
    append(panel, [head, h("div", { class: "dsplace" }, where, drawBtn),
      h("div", { class: "fields" },
        (facts.domains || []).length > 1 ? h("label", {}, W.grid, gridPick) : null,
        h("label", {}, W.fine, finePick), h("label", {}, W.hours, hoursIn),
        h("label", { class: "wide" }, W.card, cardPick),
        h("label", {}, W.pictures, picturesPick), h("label", {}, W.name, nameIn)),
      edges, say, out, h("div", { class: "btns" }, reviewBtn, startBtn), command]);
  }
  stage.append(panel);
  refresh();
  return { close };
}
