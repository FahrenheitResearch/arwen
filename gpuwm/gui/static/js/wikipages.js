// The storm wiki's pages: Main page, event, kind of storm, place, places, run article, search and recent changes.
// Every page is drawn from the page store's JSON (/api/wiki/...); every fact carries its footnote; every weather
// picture is a PNG the Rust renderer wrote, shown as it is. Each page is the same two columns: the article on the
// left, and on the right a small map, a picture and the sources.

import { h, append, fill, spacing } from "./core.js";
import * as api from "./api.js";
import { register, go, runRoute, notice, errorText, runContext, runCrumbs } from "./router.js";
import { actionButton } from "./command.js";
import { wikiMap } from "./wikimap.js";
import { productName } from "./looks.js";
import { parseTime, utcText } from "./time.js";
import {
  Cites, categories, day, openSource, eventHref, factIndex, factTable, generatedMark, hrefFor, kindHref, link, mapPanel,
  parseSearch, picture, placeHref, prose, rarityChip, runHref, searchBox, searchHref, section,
} from "./wikikit.js";

const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// "2 to 11 Nov 2013", "28 Aug to 2 Sep 2005"
function spanText(a, b) {
  const m1 = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(a || ""));
  const m2 = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(b || ""));
  if (!m1) return "";
  if (!m2) return day(a);
  const [y1, mo1, d1] = [m1[1], Number(m1[2]), Number(m1[3])];
  const [y2, mo2, d2] = [m2[1], Number(m2[2]), Number(m2[3])];
  if (y1 === y2 && mo1 === mo2) return d1 === d2 ? day(a) : `${d1} to ${d2} ${MON[mo2 - 1]} ${y2}`;
  if (y1 === y2) return `${d1} ${MON[mo1 - 1]} to ${d2} ${MON[mo2 - 1]} ${y2}`;
  return `${day(a)} to ${day(b)}`;
}

function mapBox(cls = "") {
  return h("canvas", { class: `wmap ${cls}`.trim() });
}

function recipeBox(r) {
  if (!r || r.lat === undefined) return null;
  const dlat = r.height_km / 2 / 111.32;
  const dlon = r.width_km / 2 / (111.32 * Math.max(0.05, Math.cos((r.lat * Math.PI) / 180)));
  return { w: r.lon - dlon, e: r.lon + dlon, s: r.lat - dlat, n: r.lat + dlat, dashed: true };
}

// A best run's grids on the map: each fitted grid's own centre and size, dashed.
function rowBoxes(row) {
  if (!row || !row.domains) return [];
  return row.domains.filter((d) => d.centre).map((d) => recipeBox({ lat: d.centre.lat, lon: d.centre.lon,
    width_km: d.width_km, height_km: d.height_km }));
}

// The event's own place, labelled on its map.
function placeLabel(ev) {
  const where = ev.where || {};
  return where.lat !== undefined && where.lon !== undefined ? [{ lon: where.lon, lat: where.lat, text: where.label || ev.title }] : [];
}

function eventGeometry(ev) {
  const g = ev.geometry || {};
  const out = { tracks: [], paths: [] };
  if (g.track && g.track.length) out.tracks.push({ points: g.track });
  if (g.path && g.path.length) {
    const [a, b] = g.path;
    out.paths.push({ from: a, to: b && (b[0] || b[1]) ? b : null });
  }
  return out;
}

function hostOf(url) {
  try { return new URL(url).host; } catch (_) { return url; }
}

function runTile(row, words) {
  const w = words.screens;
  const pic = row.thumb ? h("div", { class: "pic" }, picture(row.id, row.thumb, "")) : h("div", { class: "pic none" }, w.runs.no_picture);
  const bits = [row.start, row.hours ? `${row.hours} h` : null, (row.dx_km || []).map(spacing).join(" / ")].filter(Boolean);
  return h("a", { class: "runtile", href: runHref(row.id) }, pic,
    h("div", { class: "rt" }, h("b", {}, row.title || row.id), h("span", { class: `st ${row.state}` }, w.states[row.state] || row.state),
      h("span", { class: "dim" }, bits.join(" · "))));
}

function tileRow(row) {
  return { id: row.id, title: row.title, state: row.status.state, start: row.status.start_time, thumb: row.card && row.card.thumb,
    hours: row.card && row.card.hours, dx_km: row.card && row.card.dx_km };
}

// The seed tag tells seed pages from atlas pages, so it shows only in a list that holds both kinds.
function mixedSeed(rows) {
  const events = (rows || []).filter((r) => (r.what ? r.what === "event" : !r.kind || r.kind === "event"));
  return events.some((r) => r.seed) && events.some((r) => !r.seed);
}

function allSeed(rows, words) {
  const events = (rows || []).filter((r) => (r.what ? r.what === "event" : !r.kind || r.kind === "event"));
  return events.length && events.every((r) => r.seed) ? h("p", { class: "note" }, words.wiki.search.all_seed) : null;
}

function seedTag(row, show, words) {
  return row.seed && show ? h("span", { class: "tag", title: words.wiki.article.seed }, words.wiki.article.seed_tag) : null;
}

function eventLine(row, words, showSeed = false) {
  return h("li", {},
    h("a", { href: eventHref(row.id) }, row.title), rarityChip(row),
    h("span", { class: "dim" }, [row.type_title, row.region, day(row.start)].filter(Boolean).join(" · ")),
    seedTag(row, showSeed, words));
}

function eventRows(rows, words) {
  return h("section", { class: "panel" }, h("ul", { class: "rows" }, rows.map((row) => eventLine(row, words, mixedSeed(rows)))));
}

// The sidebar group of one kind's events, the open one lit.
async function kindContext(page, kindId, kindTitle, openId) {
  try {
    const data = await api.get(`/api/wiki/kind/${encodeURIComponent(kindId)}`);
    const events = data.events.slice().sort((a, b) => (a.rarity_pct ?? 1e9) - (b.rarity_pct ?? 1e9) || a.title.localeCompare(b.title));
    page.setContext({ title: kindTitle || data.phenomenon.plural, items: events.slice(0, 24).map((e) => ({
      href: eventHref(e.id), label: e.title, on: e.id === openId, title: e.title })) });
  } catch (_) { /* the sidebar keeps its fixed links */ }
}

// ---------------------------------------------------------------- Main page

async function mainPage(body, args, page) {
  const words = page.words;
  const w = words.wiki.main;
  const [data, runsData] = await Promise.all([api.get("/api/wiki"), api.get("/api/runs")]);
  const c = data.counts;
  page.setCrumbs([[words.wiki.name]]);
  const main = h("div", { class: "article" });
  const aside = h("aside", {});
  body.append(h("h1", {}, w.title), h("p", { class: "lede" }, fill(w.welcome, { events: c.events, places: c.places, sources: c.sources }),
    c.seed ? ` ${w.seed_line}` : ""), h("div", { class: "page2" }, main, aside));

  const closers = [];
  if (data.featured) {
    const f = data.featured;
    const ev = f.event;
    const cites = new Cites(f.sources, words);
    const canvas = mapBox("feature");
    main.append(h("section", { class: "panel" }, h("div", { class: "phd" }, h("b", {}, w.featured),
      h("span", {}, [f.phenomenon.title, ev.region].filter(Boolean).join(" · "))),
    h("div", { class: "feat" }, canvas,
      h("div", { class: "ftext" }, h("h3", {}, link(eventHref(ev.id), ev.title)),
        h("div", { class: "note" }, spanText((ev.when || {}).start, (ev.when || {}).end)),
        prose((ev.summary || {}).parts, factIndex(ev.facts), cites, "prose"),
        ev.rarity ? h("p", { class: "note" }, ev.rarity.text, cites.mark(ev.rarity.cite)) : null,
        h("div", { class: "btns" }, f.runnable ? h("a", { class: "btn primary", href: eventHref(ev.id) }, words.wiki.article.best.button) : null,
          h("a", { class: "btn", href: eventHref(ev.id) }, w.read))))));
    closers.push(wikiMap(canvas, { ...eventGeometry(ev), names: placeLabel(ev), inset: true, minSpanDeg: 8 }));
  }
  main.append(section(w.rare), eventRows(data.rare, words));

  aside.append(h("section", { class: "panel" }, h("div", { class: "phd" }, h("b", {}, w.kinds)),
    h("div", { class: "pbody kinds" }, data.phenomena.map((k) =>
      h("a", { class: "kindtile", href: kindHref(k.id) }, h("b", {}, k.plural), h("span", {}, fill(w.events, { count: k.count })))))));
  const mine = runsData.runs.slice(0, 2);
  aside.append(h("section", { class: "panel" }, h("div", { class: "phd" }, h("b", {}, w.yours), link("#/runs", `${runsData.total || runsData.runs.length}`)),
    mine.length ? h("div", { class: "pbody runtiles" }, mine.map((row) => runTile(tileRow(row), words)))
      : h("p", { class: "pbody note" }, w.yours_none)));
  aside.append(h("section", { class: "panel" }, h("div", { class: "phd" }, h("b", {}, w.changes), link("#/changes", w.all_changes)),
    h("ul", { class: "rows" }, data.changes.slice(0, 6).map((ch) => h("li", {},
      h("a", { href: ch.what === "run" ? runHref(ch.id) : eventHref(ch.id) }, ch.title),
      h("span", { class: "dim mono" }, String(ch.when || "").slice(0, 10)))))));
  aside.append(h("section", { class: "panel" }, h("div", { class: "phd" }, h("b", {}, w.places), link("#/places", w.all_places)),
    h("div", { class: "pbody chips" }, data.places.filter((p) => p.count).slice(0, 14).map((p) =>
      h("a", { class: "chip", href: placeHref(p.id) }, p.title, h("span", {}, String(p.count)))))));
  return () => closers.forEach((m) => m.close());
}

// ---------------------------------------------------------------- Event page

// Each frame's lead from the run's start: "+3 h" when every frame is on the hour, and to the minute ("+0:15")
// when frames come more often, so no two frames share a label.
function frameLeads(pictures, start) {
  const t0 = (parseTime(start) || { getTime: () => NaN }).getTime()
    || Math.min(...pictures.filter((p) => p.valid && p.hour !== null && p.hour !== undefined)
      .map((p) => Date.parse(p.valid) - p.hour * 3600000));
  const minutes = pictures.map((p) => (p.valid && Number.isFinite(t0) ? Math.round((Date.parse(p.valid) - t0) / 60000) : null));
  const hourly = minutes.every((m) => m === null || m % 60 === 0);
  return minutes.map((m, k) => {
    if (m === null) return pictures[k].hour !== null && pictures[k].hour !== undefined ? `+${pictures[k].hour} h` : null;
    const sign = m < 0 ? "−" : "+";
    const a = Math.abs(m);
    return hourly ? `${sign}${a / 60} h` : `${sign}${Math.floor(a / 60)}:${String(a % 60).padStart(2, "0")}`;
  });
}

// ---- the best run: one press runs the event's best layout for this computer's card

const gbText = (x) => `${Number(x)} GB`;

export function minutesText(min) {
  if (min === null || min === undefined) return null;
  if (min < 60) return `${min} min`;
  const hrs = Math.floor(min / 60);
  const rest = Math.round((min - hrs * 60) / 5) * 5;
  return rest ? `${hrs} h ${rest} min` : `${hrs} h`;
}

function startText(iso) {
  return String(iso || "").replace("T", " ").replace(/Z$/, "").slice(0, 16);
}

// The row this computer gets: the biggest layout its card holds whose output fits the free disk.
export function pickRow(cards, sys) {
  const usable = cards.filter((c) => c.fits && c.runnable);
  const own = sys && sys.card_gb ? sys.card_gb : null;
  if (!own) return { row: null };
  const mine = usable.filter((c) => c.card_gb <= own);
  const byMemory = mine[mine.length - 1] || null;
  const free = sys.disk_free_gib;
  if (free === null || free === undefined) return { row: byMemory, byMemory };
  const byDisk = [...mine].reverse().find((c) => !c.disk_gib || c.disk_gib <= free) || null;
  return { row: byDisk, byMemory, diskShort: byMemory && byDisk !== byMemory, free };
}

export function rowLine(row, wb) {
  const doms = row.domains || [];
  const spacings = doms.map((d) => spacing(d.dx_km)).join(", ");
  const fine = doms[doms.length - 1] || {};
  return fill(wb.line, {
    grids: doms.length === 1 ? wb.grids_one : fill(wb.grids_many, { n: doms.length, spacings }),
    finest: spacing(row.finest_km),
    box: fine.width_km ? `${Math.round(fine.width_km)} × ${Math.round(fine.height_km)} km` : "",
    hours: row.length_h, start: startText(row.start), source: row.source_name,
    time: timeText(row, wb),
  });
}

// The wait from the press to the last picture, with the forecast's share of it: the total is what a
// run measured with its pictures drawn after the forecast; the forecast part is the floor either way.
export function timeText(row, wb) {
  const time = minutesText(row.est_minutes);
  if (!time) return wb.time_none;
  const total = minutesText(row.est_total_minutes);
  const measured = row.est_kind === "measured";
  if (total && row.est_total_minutes > row.est_minutes) {
    return fill(measured ? wb.time_total_measured : wb.time_total_about, { total, time });
  }
  return fill(measured ? wb.time_measured : wb.time_about, { time: total || time });
}

function bestPanel(ev, best, words, cites, onrow) {
  const wb = words.wiki.article.best;
  const cards = best.cards;
  const holder = h("section", { class: "panel", "aria-live": "polite" });
  let sys = null;
  let chosen = null;
  let mineRow = null;
  let pick = {};

  function paint() {
    const row = chosen !== null ? cards.find((c) => c.card_gb === chosen) : pick.row;
    const own = sys && sys.card_gb;
    // the card itself, which is not always a layout size: a 10 GB card runs the 8 GB layout
    const ownGb = sys && sys.memory_gib ? Math.round(sys.memory_gib) : own;
    const seg = h("span", { class: "seg", role: "group", "aria-label": wb.cards }, cards.map((c) => {
      const on = row && c.card_gb === row.card_gb;
      const mine = own && c.card_gb === (mineRow || {}).card_gb;
      const b = h("button", { type: "button", class: `${on ? "on" : ""}${mine ? " mine" : ""}`.trim() || null,
        "aria-pressed": on ? "true" : "false",
        title: mine ? fill(wb.this_computer, { own: ownGb }) : null }, gbText(c.card_gb));
      b.addEventListener("click", () => { chosen = c.card_gb; paint(); });
      return b;
    }));
    const lines = [];
    const problems = [];
    let actions = null;
    let heading = wb.title;
    if (!sys) lines.push(h("p", { class: "note" }, wb.checking));
    if (sys && !own && chosen === null) lines.push(h("p", { class: "fitline" }, wb.no_card));
    if (sys && own && !pick.row && pick.byMemory && chosen === null) {
      lines.push(h("p", { class: "fitline no" }, fill(wb.disk_none, { free: Math.round(sys.disk_free_gib) })));
    }
    const free = sys ? sys.disk_free_gib : null;
    if (row) {
      const mine = own && chosen === null;
      const fallback = mine && mineRow && row.card_gb !== mineRow.card_gb;
      const layoutOnly = mine && !fallback && ownGb !== row.card_gb;
      heading = fill(fallback ? wb.for_fallback : layoutOnly ? wb.for_mine_layout : mine ? wb.for_mine : wb.for_card,
        { card: gbText(row.card_gb), own: ownGb });
      lines.unshift(h("code", {}, rowLine(row, wb)), cites.mark(row.cite));
      if (!row.fits) problems.push(fill(wb.no_fit, { card: gbText(row.card_gb) }));
      else if (!row.runnable) problems.push(fill(wb.not_offered, { source: row.source_name }));
      if (own && row.card_gb > own) problems.push(fill(wb.too_big, { card: gbText(row.card_gb), own: ownGb }));
      if (row.disk_gib && free !== null && free !== undefined && row.disk_gib > free) {
        problems.push(fill(wb.disk_too_small, { need: Math.round(row.disk_gib), free: Math.round(free) }));
      }
      const run = actionButton(wb.button, {
        kind: "primary", disabled: problems.length > 0,
        request: () => ({ path: "/api/wiki/simulate", body: { event: ev.id, card_gb: row.card_gb } }),
        onclick: async () => {
          run.button.disabled = true;
          notice(wb.starting);
          try {
            const reply = await api.post("/api/wiki/simulate", { event: ev.id, card_gb: row.card_gb });
            notice(wb.started);
            go(runRoute("watch", reply.run));
          } catch (err) {
            notice(errorText(err), "stop");
            run.button.disabled = false;
          }
        },
      });
      run.button.classList.add("big");
      run.button.insertAdjacentHTML("afterbegin", '<svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true">' +
        '<path d="M5 3.5v9l7-4.5z" fill="currentColor"/></svg>');
      actions = run;
      for (const text of problems) lines.push(h("p", { class: "fitline no" }, text));
    }
    // Customise opens New forecast with this run filled in, grids and all: the way to another start data, start
    // time, length, box, grid, physics or pictures, so it sits beside the button and the start is said in words.
    const customHref = row ? `#/create/recipe/${encodeURIComponent(ev.id)}/${row.card_gb}` : null;
    const custom = row ? h("a", { class: "btn", href: customHref, title: wb.customise_help }, wb.customise) : null;
    if (row && row.fits) {
      lines.push(h("p", { class: "note starts" }, fill(wb.starts_from,
        { source: row.source_name, start: startText(row.start), hours: row.length_h }), " ",
      h("a", { href: customHref, title: wb.customise_help }, wb.change)));
    }
    let diskNote = null;
    if (row && !problems.length && pick.diskShort && chosen === null) {
      diskNote = fill(wb.disk_short, { card: gbText(pick.byMemory.card_gb), need: Math.round(pick.byMemory.disk_gib), free: Math.round(pick.free) });
    } else if (row && !problems.length && row.disk_gib && free !== null && free !== undefined) {
      diskNote = fill(wb.disk, { need: Math.round(row.disk_gib), free: Math.round(free) });
    }
    const why = row && row.fits ? h("div", { class: "runwhy" },
      h("div", {}, h("b", {}, `${wb.why}. `), row.why, cites.mark(row.cite)),
      h("div", {}, row.rung === (best.ideal || {}).rung ? wb.same_as_ideal : [h("b", {}, `${wb.given_up}: `), row.what_is_given_up]),
      row.est_basis ? h("div", { class: "note" }, `${row.est_basis[0].toUpperCase()}${row.est_basis.slice(1)}.`) : null) : null;
    // the command line, once shown, takes a full-width row under the heading and the button
    const cmdRow = actions && actions.cmd ? actions.cmd.querySelector(".livecmd") : null;
    holder.replaceChildren(
      h("div", { class: "runp" }, h("div", {}, h("h3", {}, heading), lines),
        actions || custom ? h("div", { class: "runacts" }, actions, custom, custom ? h("span", { class: "note" }, wb.customise_line) : null)
          : h("span", {}),
        cmdRow ? h("div", { class: "runcmd" }, cmdRow) : null),
      h("div", { class: "pbar" }, h("span", {}, wb.cards), seg, diskNote ? h("span", { class: "note" }, diskNote) : null),
      ...(why ? [why] : []));
    onrow(row);
  }
  paint();
  api.get("/api/system").catch(() => ({})).then((reply) => {
    sys = reply || {};
    pick = pickRow(cards, sys);
    mineRow = sys.card_gb ? [...cards].reverse().find((c) => c.card_gb <= sys.card_gb) : null;
    paint();
  });
  return holder;
}

function bestTable(best, words, cites) {
  const wb = words.wiki.article.best;
  return h("section", { class: "panel" }, h("div", { class: "tablewrap" }, h("table", { class: "dtable" },
    h("thead", {}, h("tr", {}, wb.table_cols.map((c) => h("th", {}, c)))),
    h("tbody", {}, best.cards.map((c) => h("tr", {},
      h("td", { class: "mono" }, gbText(c.card_gb)),
      h("td", {}, c.fits ? (c.domains || []).map((d) => spacing(d.dx_km)).join(", ") : c.why || ""),
      h("td", {}, c.fits ? `${spacing(c.finest_km)}, ${Math.round((c.domains || []).slice(-1)[0].width_km)} × ${Math.round((c.domains || []).slice(-1)[0].height_km)} km` : ""),
      h("td", { class: "mono" }, c.fits ? String(c.length_h) : ""),
      h("td", {}, c.est_total_minutes > c.est_minutes
        ? fill(wb.table_time, { total: minutesText(c.est_total_minutes), time: minutesText(c.est_minutes) })
        : minutesText(c.est_total_minutes || c.est_minutes) || ""),
      h("td", {}, c.disk_gib ? `${Math.round(c.disk_gib)} GiB` : ""),
      h("td", {}, c.memory ? fill(wb.memory, { need: c.memory.need_gib.toFixed(1), budget: c.memory.budget_gib.toFixed(1) }) : "",
        cites.mark(c.cite))))))));
}

async function eventPage(body, args, page) {
  const words = page.words;
  const wa = words.wiki.article;
  const id = args[0];
  let pendingRow = null;
  const data = await api.get(`/api/wiki/event/${encodeURIComponent(id)}`);
  const ev = data.event;
  const facts = factIndex(ev.facts);
  const cites = new Cites(data.sources, words);
  document.title = `${ev.title} · ${words.wiki.name}`;
  page.setCrumbs([[words.wiki.name, "#/wiki"], [data.phenomenon.plural || data.phenomenon.title, kindHref(data.phenomenon.id)], [ev.title]]);
  kindContext(page, data.phenomenon.id, data.phenomenon.plural, ev.id);

  const canvas = mapBox("tall");
  const geo = eventGeometry(ev);
  const best = data.best;
  const trackCite = cites.mark((ev.geometry || {}).track_cite || (ev.geometry || {}).path_cite);
  const main = h("div", { class: "article" });
  const aside = h("aside", {});

  // head: tags, title, the lede from the page's own facts
  const lead = prose((ev.summary || {}).parts, facts, cites);
  if ((ev.summary || {}).generated) lead.append(" ", generatedMark(words));
  main.append(h("div", { class: "tags" }, h("a", { class: "tag a", href: kindHref(data.phenomenon.id) }, data.phenomenon.title),
    ev.region ? h("a", { class: "tag", href: searchHref({ region: ev.region }) }, ev.region) : null,
    ev.seed ? h("span", { class: "tag", title: wa.seed }, wa.seed_tag) : h("span", { class: "tag" }, wa.atlas_tag)),
  h("h1", {}, ev.title), lead);

  let m = null;
  const geoBase = { ...geo, names: placeLabel(ev), inset: true, minSpanDeg: ev.type === "tornado" ? 5 : 8 };
  if (best) {
    main.append(bestPanel(ev, best, words, cites, (row) => {
      pendingRow = row;
      if (!canvas.isConnected) return;
      if (m) m.close();
      m = wikiMap(canvas, { ...geoBase, boxes: rowBoxes(row) });
    }));
  }

  const rarity = ev.rarity;
  const row = data.row || {};
  main.append(rarity ? h("div", { class: "nbox" }, h("div", { class: "big" }, String(rarity.count ?? row.rarity ?? ""),
    h("small", {}, ` / ${Number(rarity.of ?? row.rarity_of ?? 0).toLocaleString("en-US")}`)),
  h("p", {}, h("span", { class: "lbl" }, wa.notable), rarity.text, cites.mark(rarity.cite)))
    : h("div", { class: "nbox" }, h("p", {}, h("span", { class: "lbl" }, wa.notable), wa.no_rarity)));

  const shownFacts = (ev.facts || []).filter((f) => f.infobox !== false);
  main.append(section([wa.facts, fill(wa.facts_note, { count: shownFacts.length })]), factTable(ev.facts, cites));

  if (data.places.length) {
    main.append(section(wa.places, h("div", { class: "chips" }, data.places.map((p) => h("a", { class: "chip", href: placeHref(p.id) }, p.title,
      h("span", {}, (words.wiki.place.kinds || {})[p.kind] || p.kind))))));
  }
  if ((ev.observed || []).length) {
    main.append(section(wa.observed, h("section", { class: "panel" }, h("ul", { class: "rows" }, ev.observed.map((o) =>
      h("li", {}, h("a", { href: o.url, target: "_blank", rel: "noopener noreferrer" }, o.label), cites.mark(o.cite),
        h("span", { class: "dim mono url", title: o.url }, hostOf(o.url))))))));
  }
  const era = (ev.era5 || {}).quality;
  main.append(section(wa.setup,
    (ev.era5 || {}).statistics ? h("div", {}, Object.entries(ev.era5.statistics).map(([k, v]) => h("p", {}, `${k}: ${v.text || v}`,
      cites.mark(v.cite)))) : h("p", { class: "note" }, wa.setup_none),
    era ? h("p", {}, h("span", { class: "tag", title: wa.era }, era.flag.replace(/-/g, " ")), " ", era.text, cites.mark(era.cite)) : null));
  if (best) main.append(section(wa.best.table), bestTable(best, words, cites));

  const runs = data.runs || [];
  main.append(section(wa.runs, runs.length ? h("div", { class: "runtiles" }, runs.map((r) => runTile(r, words)))
    : h("p", { class: "note" }, wa.runs_none), h("p", { class: "note" }, wa.runs_rule)));
  if (data.see_also.length) main.append(section(wa.see_also), eventRows(data.see_also, words));
  main.append(categories(data.categories, words));

  // the aside: the map, a run of this event, the sources
  const when = ev.when || {};
  aside.append(mapPanel(canvas, [
    h("span", {}, h("i", {}), geo.tracks.length ? wa.map_track : wa.map_path, trackCite),
    best ? h("span", {}, h("i", { class: "b" }), wa.map_area, cites.mark((best.cite || []).slice(0, 1))) : null,
  ], spanText(when.start, when.end)));
  const shown = runs.find((r) => r.thumb);
  if (shown) {
    aside.append(h("section", { class: "panel" }, h("div", { class: "phd" }, h("b", {}, wa.run_of),
      h("span", {}, (shown.dx_km || []).map(spacing).join(" / "))),
    h("a", { class: "shotlink", href: `#/${runRoute("results", shown.id)}`, title: wa.run_open }, h("img", { class: "shot", src: api.filePath(shown.id, shown.thumb), alt: "" })),
    h("div", { class: "runrow" }, h("div", { class: "mono" }, shown.title || shown.id), [shown.start, shown.hours ? `${shown.hours} h` : null].filter(Boolean).join(" · "))));
  }
  aside.append(cites.panel());

  body.append(h("div", { class: "page2" }, main, aside));
  if (m) m.close();
  m = wikiMap(canvas, { ...geoBase, boxes: rowBoxes(pendingRow) });
  // #/event/ID/source/SOURCE links straight to one reference, open in the drawer.
  if (args[1] === "source" && args[2]) openSource(args[2], cites);
  return () => { if (m) m.close(); };
}

// ---------------------------------------------------------------- Kind of storm

async function kindPage(body, args, page) {
  const words = page.words;
  const wk = words.wiki.kind;
  const data = await api.get(`/api/wiki/kind/${encodeURIComponent(args[0])}`);
  const k = data.phenomenon;
  const cites = new Cites(data.sources, words);
  document.title = `${k.title} · ${words.wiki.name}`;
  page.setCrumbs([[words.wiki.name, "#/wiki"], [k.plural || k.title]]);
  const events = data.events;
  const filters = { region: "", decade: "", season: "" };
  let sortKey = "rarity_pct";
  let sortDir = 1;
  const tbody = h("tbody", {});
  const shown = h("span", { class: "note" });
  const select = (name, label, options, fmt) => {
    const s = h("select", { class: "input" }, h("option", { value: "" }, `${label}: ${words.wiki.search.any}`),
      options.map((o) => h("option", { value: o.value }, `${fmt ? fmt(o.value) : o.value} (${o.count})`)));
    s.addEventListener("change", () => { filters[name] = s.value; paint(); });
    return s;
  };
  const f = data.facets;
  const bar = h("div", { class: "filters" },
    select("region", words.wiki.search.region, f.region),
    select("decade", words.wiki.search.decade, f.decade, (v) => `${v}s`),
    select("season", words.wiki.search.season, f.season, (v) => words.wiki.seasons[v] || v), shown);
  const keyOf = { 0: "title", 1: "region", 2: "start", 3: "season", 4: "rarity_pct" };
  const head = h("tr", {}, wk.columns.map((c, i) => {
    const th = h("th", { class: "sortable" }, c);
    th.addEventListener("click", () => { const key = keyOf[i]; sortDir = sortKey === key ? -sortDir : 1; sortKey = key; paint(); });
    return th;
  }));
  function paint() {
    const list = events.filter((e) => (!filters.region || e.region === filters.region)
      && (!filters.decade || String(e.decade) === filters.decade) && (!filters.season || e.season === filters.season));
    list.sort((a, b) => {
      const va = a[sortKey] ?? ""; const vb = b[sortKey] ?? "";
      return (va < vb ? -1 : va > vb ? 1 : 0) * sortDir || a.title.localeCompare(b.title);
    });
    shown.textContent = fill(wk.shown, { shown: list.length, count: events.length });
    tbody.replaceChildren(...list.map((e) => h("tr", {},
      h("td", {}, link(eventHref(e.id), e.title), " ", seedTag(e, mixedSeed(events), words)),
      h("td", {}, e.region), h("td", { class: "mono" }, day(e.start)), h("td", {}, words.wiki.seasons[e.season] || e.season),
      h("td", {}, rarityChip(e)))));
    for (const th of head.children) th.classList.remove("up", "down");
    const idx = Object.values(keyOf).indexOf(sortKey);
    if (head.children[idx]) head.children[idx].classList.add(sortDir > 0 ? "up" : "down");
  }
  paint();
  const canvas = mapBox("tall");
  const main = h("div", { class: "article" },
    h("div", { class: "tags" }, h("span", { class: "tag a" }, words.wiki.article.from)),
    h("h1", {}, k.plural || k.title),
    h("p", { class: "lede" }, `“${k.what.quote}”`, cites.mark(k.what.cite)),
    section(wk.how, h("p", {}, k.detect.text, cites.mark(k.detect.cite))),
    section(fill(wk.events, { kind: (k.title || "").toLowerCase() }), bar,
      h("section", { class: "panel" }, h("div", { class: "tablewrap" }, h("table", { class: "dtable" }, h("thead", {}, head), tbody)))),
    data.others.length ? section(wk.others, h("div", { class: "chips" }, data.others.map((o) => h("a", { class: "chip", href: kindHref(o.id) }, o.title)))) : null);
  const aside = h("aside", {}, mapPanel(canvas, [h("span", {}, h("i", { class: "dot" }), wk.map)], fill(words.wiki.main.events, { count: events.length })),
    cites.panel());
  body.append(h("div", { class: "page2" }, main, aside));
  kindContext(page, k.id, k.plural, null);
  const m = wikiMap(canvas, { dots: events.filter((e) => e.where).map((e) => ({ lon: e.where.lon, lat: e.where.lat, label: e.title, id: e.id })),
    onpick: (d) => go(`event/${encodeURIComponent(d.id)}`), minSpanDeg: 30, labels: false });
  return () => m.close();
}

// ---------------------------------------------------------------- Place

async function placePage(body, args, page) {
  const words = page.words;
  const wp = words.wiki.place;
  const data = await api.get(`/api/wiki/place/${encodeURIComponent(args[0])}`);
  const p = data.place;
  const cites = new Cites(data.sources, words);
  document.title = `${p.title} · ${words.wiki.name}`;
  page.setCrumbs([[words.wiki.name, "#/wiki"], [wp.title, "#/places"], data.parent ? [data.parent.title, placeHref(data.parent.id)] : null, [p.title]]);
  const counts = Object.entries(p.counts || {}).map(([, c]) => h("li", {}, c.text, cites.mark(c.cite)));
  const rarest = data.events[0];
  const factRows = [{ label: wp.kind, text: wp.kinds[p.kind] || p.kind, cite: p.cite }];
  const main = h("div", { class: "article" },
    h("div", { class: "tags" }, h("span", { class: "tag a" }, wp.kinds[p.kind] || p.kind),
      data.parent ? h("a", { class: "tag", href: placeHref(data.parent.id) }, `${wp.within} ${data.parent.title}`) : null),
    h("h1", {}, p.title),
    counts.length ? h("ul", { class: "lede plain" }, counts)
      : h("p", { class: "lede" }, data.events.length ? fill(data.events.length === 1 ? wp.lede_one : wp.lede_events, { count: data.events.length }) : wp.none),
    section(wp.events, data.events.length ? h("section", { class: "panel" }, h("ul", { class: "rows" }, data.events.map((row) => h("li", {},
      h("a", { href: eventHref(row.id) }, row.title), rarityChip(row),
      h("span", { class: "dim" }, [row.type_title, day(row.start)].join(" · ")),
      row.rarity_text ? h("div", { class: "note" }, row.rarity_text) : null))))
      : h("p", { class: "note" }, wp.none)),
    data.children.length ? section(wp.contains, h("div", { class: "chips" }, data.children.map((c) => h("a", { class: "chip", href: placeHref(c.id) }, c.title)))) : null);
  const table = factTable(factRows, cites);
  const extra = h("tbody", {},
    data.parent ? h("tr", {}, h("td", { class: "k" }, wp.within), h("td", {}, link(placeHref(data.parent.id), data.parent.title)), h("td", {})) : null,
    rarest ? h("tr", {}, h("td", { class: "k" }, wp.rarest), h("td", {}, link(eventHref(rarest.id), rarest.title)), h("td", {})) : null);
  table.querySelector("table").append(extra);
  const canvas = data.events.length ? mapBox("tall") : null;
  const aside = h("aside", {}, canvas ? mapPanel(canvas, [h("span", {}, h("i", { class: "dot" }), wp.map_line)]) : null, table, cites.panel());
  body.append(h("div", { class: "page2" }, main, aside));
  if (!canvas) return null;
  const m = wikiMap(canvas, { dots: data.events.filter((e) => e.where).map((e) => ({ lon: e.where.lon, lat: e.where.lat, label: e.title,
    id: e.id })), onpick: (d) => go(`event/${encodeURIComponent(d.id)}`), minSpanDeg: 10,
  inset: true, names: placeLabelOf(p, data.events) });
  return () => m.close();
}

// A place's name on its map: at the middle of its events, or its own point when the store gives one.
function placeLabelOf(p, events) {
  if (p.lat !== undefined && p.lon !== undefined) return [{ lon: p.lon, lat: p.lat, text: p.title }];
  const pts = events.filter((e) => e.where);
  if (!pts.length) return [];
  const lat = pts.reduce((a, e) => a + e.where.lat, 0) / pts.length;
  const lon = pts.reduce((a, e) => a + e.where.lon, 0) / pts.length;
  return [{ lon, lat, text: p.title, under: true }];
}

async function placesPage(body, args, page) {
  const words = page.words;
  const data = await api.get("/api/wiki/places");
  page.setCrumbs([[words.wiki.name, "#/wiki"], [words.wiki.place.title]]);
  const total = data.groups.reduce((a, g) => a + g.places.length, 0);
  body.append(h("div", { class: "article" }, h("h1", {}, words.wiki.place.title),
    h("p", { class: "lede" }, fill(words.wiki.place.lede, { count: total })),
    data.groups.map((g) => section([g.title, String(g.places.length)], h("div", { class: "chips" }, g.places.map((p) =>
      h("a", { class: `chip${p.count ? "" : " empty"}`, href: placeHref(p.id) }, p.title, h("span", {}, String(p.count)))))))));
}

// ---------------------------------------------------------------- Run article

// Why a stopped forecast stopped, under its title. A failed forecast's reason and what to do are its facts ("Why
// it stopped", "What to do"), so drawing them here as well printed each sentence twice; a stopped forecast has no
// such facts, and its reason (such as "Stopped before the finer forecast wrote anything.") is shown here once.
export function endNote(st) {
  const end = (st && st.end) || {};
  if (!st || st.state !== "stopped" || !end.message) return null;
  return h("div", { class: "endnote" }, h("p", {}, end.message),
    end.remedy ? h("p", { class: "note" }, end.remedy) : null);
}

async function runPage(body, args, page) {
  const words = page.words;
  const wr = words.wiki.run;
  const runId = args[0];
  const data = await api.get(`/api/wiki/run/${encodeURIComponent(runId)}`);
  const cites = new Cites(data.sources, words);
  const title = data.title || runId;
  document.title = `${title} · ${words.wiki.name}`;
  page.setCrumbs(runCrumbs(words, runId, title, null));
  page.setContext(runContext(words, runId, title, "run"));
  const looks = words.looks;
  const pics = data.pictures;
  const lead = (pics.favourites || [])[0] || (pics.products || [])[0];
  const leadGroup = pics.groups.filter((g) => g.product === lead).sort((a, b) => b.newest_mtime - a.newest_mtime)[0];
  const running = data.status.state === "running" || data.status.state === "ready";
  const mapHref = `#/${runRoute(running ? "watch" : "results", runId)}`;

  const main = h("div", { class: "article" });
  const aside = h("aside", {});
  // core's append skips the absent note; the DOM's own append would print it as the word "null".
  append(main, [
    h("div", { class: "tags" }, h("span", { class: "tag a" }, wr.yours), h("span", { class: `tag st ${data.status.state}` }, words.screens.states[data.status.state] || data.status.state),
      data.event ? h("a", { class: "tag", href: eventHref(data.event.id) }, `${wr.from_event} ${data.event.title}`) : null),
    h("h1", { class: "name" }, title),
    endNote(data.status),
    h("div", { class: "btns wsec" }, h("a", { class: "btn primary", href: mapHref }, running ? wr.watch : wr.open_viewer),
      h("a", { class: "btn", href: `#/${runRoute("explore", runId)}` }, words.screens.shell.files)),
    section([words.wiki.article.facts, fill(words.wiki.article.facts_note, { count: (data.facts || []).length })]), factTable(data.facts, cites)]);
  // What the engine warned about (a warm bubble above 10 K, say): the run went on as configured, and says so here.
  const warned = data.status.library_warnings || [];
  if (warned.length) {
    main.append(section([wr.warnings, wr.warnings_line], h("div", { class: "runwarnings" }, warned.map((item) =>
      h("p", {}, item.message, item.detail ? h("span", { class: "dim" }, ` ${item.detail}`) : null)))));
  }

  // Timeline: every frame of the lead picture, in time order.
  if (lead) {
    const strip = h("div", { class: "strip" });
    main.append(section([wr.timeline, productName(lead, looks)], strip));
    api.get(`${api.runPath(runId)}/pictures/list?${new URLSearchParams({ product: lead, domain: leadGroup.domain })}`).then((list) => {
      const leads = frameLeads(list.pictures, data.status && data.status.start_time);
      strip.replaceChildren(...list.pictures.map((p, k) => h("a", { class: "tframe", href: `${mapHref}/${encodeURIComponent(lead)}` },
        picture(runId, p.path, p.label), h("span", { class: "mono" }, leads[k] || p.label),
        p.valid ? h("small", {}, utcText(p.valid)) : null)));
    }).catch(() => {});
  } else {
    main.append(section(wr.timeline, h("p", { class: "note" }, wr.no_pictures)));
  }
  const favs = (pics.favourites || []).slice(0, 8);
  if (favs.length) {
    main.append(section([wr.pictures, wr.pictures_line], h("div", { class: "gallery" }, favs.map((product) => {
      const g = pics.groups.filter((x) => x.product === product).sort((a, b) => b.newest_mtime - a.newest_mtime)[0];
      return h("figure", {}, h("a", { href: `${mapHref}/${encodeURIComponent(product)}` }, picture(runId, g.newest, productName(product, looks))),
        h("figcaption", {}, productName(product, looks), h("span", { class: "dim mono" }, ` ${g.domain}`)));
    }))));
  }
  main.append(section(wr.events, data.events.length ? eventRows(data.events, words) : h("p", { class: "note" }, wr.events_none),
    h("p", { class: "note" }, words.wiki.article.runs_rule)));
  main.append(section(wr.verification, data.verification.length ? h("ul", { class: "plain mono" }, data.verification.map((rel) =>
    h("li", {}, h("a", { href: api.filePath(runId, rel), target: "_blank", rel: "noopener" }, rel))))
    : h("p", { class: "note" }, wr.verification_none)));

  if (leadGroup) {
    aside.append(h("section", { class: "panel" }, h("div", { class: "phd" }, h("b", {}, productName(lead, looks)), h("span", { class: "mono" }, leadGroup.domain)),
      h("a", { class: "shotlink", href: `${mapHref}/${encodeURIComponent(lead)}` }, h("img", { class: "shot", src: api.filePath(runId, leadGroup.newest), alt: "" })),
      h("div", { class: "runrow" }, wr.newest)));
  }
  const canvas = data.box ? mapBox() : null;
  if (canvas) aside.append(mapPanel(canvas, [h("span", {}, h("i", { class: "s" }), wr.map), data.events.length ? h("span", {}, h("i", { class: "dot" }), wr.events) : null]));
  aside.append(cites.panel());
  body.append(h("div", { class: "page2" }, main, aside));
  if (args[1] === "source" && args[2]) openSource(args[2], cites);
  if (!canvas) return null;
  const m = wikiMap(canvas, { boxes: [{ ...data.box, dashed: false }], dots: data.events.filter((e) => e.where)
    .map((e) => ({ lon: e.where.lon, lat: e.where.lat, id: e.id, label: e.title })), minSpanDeg: 12,
  onpick: (d) => go(`event/${encodeURIComponent(d.id)}`) });
  return () => m.close();
}

// ---------------------------------------------------------------- Search

async function searchPage(body, args, page, browse = false) {
  const words = page.words;
  const ws = words.wiki.search;
  const params = parseSearch(args, page.query);
  const q = new URLSearchParams(Object.entries(params).filter(([, v]) => v));
  const data = await api.get(`/api/wiki/search?${q}`);
  page.setCrumbs([[words.wiki.name, "#/wiki"], [browse && !params.q ? words.wiki.nav.browse : ws.title]]);
  const box = searchBox(words, params.q);
  const set = (key, value) => {
    const target = searchHref({ ...params, [key]: value }).slice(2);
    go(browse && !params.q ? target.replace(/^search\//, "browse/") : target);
  };
  const select = (key, label, options, fmt) => {
    const s = h("select", { class: "input" }, h("option", { value: "" }, `${label}: ${ws.any}`),
      options.map((o) => h("option", { value: o.value }, `${fmt ? fmt(o.value) : o.value}${o.count !== undefined ? ` (${o.count})` : ""}`)));
    s.value = params[key] || "";
    s.addEventListener("change", () => set(key, s.value));
    return s;
  };
  const f = data.facets;
  const sorts = Object.entries(ws.sorts).map(([value]) => ({ value }));
  const filters = h("div", { class: "filters" },
    select("type", ws.kind, data.phenomena.map((p) => ({ value: p.id, label: p.title })), (v) => (data.phenomena.find((p) => p.id === v) || {}).title || v),
    select("region", ws.region, f.region), select("decade", ws.decade, f.decade, (v) => `${v}s`),
    select("season", ws.season, f.season, (v) => words.wiki.seasons[v] || v),
    select("sort", ws.sort, sorts, (v) => ws.sorts[v]),
    Object.keys(params).some((k) => k !== "q" && params[k]) ? h("a", { href: browse && !params.q ? "#/browse" : searchHref({ q: params.q }) }, ws.clear) : null);
  const results = h("section", { class: "panel" }, h("ul", { class: "rows results" }, data.results.map((row) => h("li", {},
    h("span", { class: "badge" }, ws.badge[row.kind] || row.kind),
    h("a", { href: hrefFor(row) }, row.title),
    row.kind === "event" ? rarityChip(row) : null,
    seedTag(row, mixedSeed(data.results), words),
    h("div", { class: "note" }, row.kind === "event" ? [row.type_title, row.region, day(row.start), row.rarity_text].filter(Boolean).join(" · ")
      : row.kind === "run" ? [row.start, row.state].filter(Boolean).join(" · ")
        : row.kind === "place" ? (words.wiki.place.kinds[row.place_kind] || row.place_kind || "") : "")))));
  body.append(h("div", { class: "article" }, h("h1", {}, params.q ? `“${params.q}”` : browse ? words.wiki.nav.browse : ws.title),
    h("p", { class: "lede" }, ws.hint), box, filters,
    data.partial ? h("p", { class: "note" }, fill(ws.partial, { words: data.words.join(", ") })) : null,
    data.words && data.words.length && params.q && data.words.join(" ") !== params.q.trim().toLowerCase()
      ? h("p", { class: "note" }, fill(ws.looked_for, { words: data.words.join(", ") })) : null,
    allSeed(data.results, words),
    h("h2", { class: "sec" }, data.count === 1 ? ws.count_one : fill(ws.count, { count: data.count })),
    data.results.length ? results : h("p", { class: "note" }, ws.none)));
  if (!browse || params.q) box.input.focus();
}

// ---------------------------------------------------------------- Recent changes

async function changesPage(body, args, page) {
  const words = page.words;
  const wc = words.wiki.changes;
  const data = await api.get("/api/wiki/changes");
  page.setCrumbs([[words.wiki.name, "#/wiki"], [wc.title]]);
  const rows = data.changes.map((ch) => h("tr", {},
    h("td", { class: "mono" }, String(ch.when || "").replace("T", " ").slice(0, 16)),
    h("td", {}, h("span", { class: "tag" }, ch.what === "run" ? wc.run : wc.event)),
    h("td", {}, link(ch.what === "run" ? runHref(ch.id) : eventHref(ch.id), ch.title)),
    h("td", { class: "dim" }, ch.what === "run"
      ? [words.screens.states[ch.state] || ch.state, ch.events.length === 1 ? wc.covers_one : ch.events.length ? fill(wc.covers, { count: ch.events.length }) : null].filter(Boolean).join(", ")
      : ch.seed ? (mixedSeed(data.changes) ? wc.seed : "") : ch.origin || "")));
  body.append(h("div", { class: "article" }, h("h1", {}, wc.title), h("p", { class: "lede" }, wc.lede), allSeed(data.changes, words),
    h("section", { class: "panel" }, h("div", { class: "tablewrap" }, h("table", { class: "dtable" }, h("thead", {}, h("tr", {}, wc.columns.map((c) => h("th", {}, c)))),
      h("tbody", {}, rows))))));
}

const wikiTitle = () => "";
register("wiki", mainPage, { title: wikiTitle });
register("browse", (body, args, page) => searchPage(body, args, page, true), { title: wikiTitle });
register("places", placesPage, { title: wikiTitle });
register("changes", changesPage, { title: wikiTitle });
register("event", eventPage, { title: wikiTitle });
register("kind", kindPage, { title: wikiTitle });
register("place", placePage, { title: wikiTitle, side: "places" });
register("search", searchPage, { title: wikiTitle, side: "browse" });
register("run", runPage, { title: wikiTitle });
