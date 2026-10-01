// New forecast: five guided steps beside one map. Where (place the box on the map), When (any past date and hour
// on a calendar, with the data sources that hold it), How fine (a few named grids for a chosen card and the number
// of vertical levels, each checked with `gpuwm run-plan PLAN --resolve`), Physics (one table per part of the model,
// each scheme a row with its cost, the set checked by `gpuwm physics-catalog --check`), then Review and start, which
// writes the run folder and launches `gpuwm run-plan PLAN.json`. The assistant fills this form through bridge.js. "Find an event" above the steps searches the wiki and fills the box, date and
// length from the event's recipe. #/create/recipe/EVENT opens it filled in the same way.

import { h, button, fill, fold, spacing, duration, place } from "./core.js";
import * as api from "./api.js";
import { register, go, goWithNotice, runRoute, notice, errorText } from "./router.js";
import { actionButton } from "./command.js";
import { boxMap } from "./map.js";
import { calendar, parseTyped, dayText, todayText } from "./datepick.js";
import { utcText, parseTime } from "./time.js";
import { onFill, takeFill, setFormReader } from "./bridge.js";

const STEPS = ["where", "when", "fine", "physics", "review"];
const EARLIEST_DAY = "1940-01-01";

function stamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `forecast-${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}-${p(d.getUTCHours())}${p(d.getUTCMinutes())}`;
}

const cardWords = (card) => card.replace("gb", " GB");
// the card's own memory as people say it: a card that reports 9.8 GiB is a 10 GB card
const ownGb = (gib) => String(Math.round(gib));
const cycleWords = (cycle) => utcText(cycle);
const hourText = (n) => String(n).padStart(2, "0");
// One decimal below 100 GB, so a grid just over the budget never reads as "needs 11 of 11".
const gb = (x) => (x >= 100 ? x.toFixed(0) : x.toFixed(1));

// A checked physics set New forecast can start: every set the engine's check says runs.  One the data source offers
// by name goes into the plan as that set, the way a preset does; any other goes in as the picked schemes, which the
// engine writes into the forecast's experiment.
export function startable(result) {
  return !!(result && result.valid);
}

// The data source's own set a checked choice makes, or "" when it makes none the source offers.
export function offeredSet(result, source) {
  if (!startable(result) || !result.named_suite || !source) return "";
  return (source.profiles || []).some((p) => p.id === result.named_suite) ? result.named_suite : "";
}

// A readable default name for a run made from a wiki event: its title words, then the start date.
function eventRunName(recipe) {
  const months = /^(january|february|march|april|may|june|july|august|september|october|november|december)$/;
  const words = String(recipe.title || recipe.event).toLowerCase().match(/[a-z0-9]+/g) || ["event"];
  const kept = words.filter((w) => !months.test(w) && !(/^\d+$/.test(w) && [1, 2, 4].includes(w.length)));
  const tail = `${String(recipe.cycle || "").slice(0, 10)}${recipe.card_gb ? `-${recipe.card_gb}gb` : ""}-custom`;
  let head = kept[0] || "event";
  for (const w of kept.slice(1)) { if (head.length + 1 + w.length > 60 - tail.length) break; head = `${head}-${w}`; }
  return `${head}-${tail}`.replace(/-+/g, "-");
}

function choiceButton(label, sub, on, onclick, title) {
  const b = h("button", { type: "button", class: `btn${on ? " on" : ""}`, "aria-pressed": on ? "true" : "false",
    title: title || null }, label, sub ? h("small", {}, sub) : null);
  b.addEventListener("click", onclick);
  return b;
}

function field(label, input, hint) {
  return h("label", {}, label, input, hint ? h("span", { class: "hint" }, hint) : null);
}

// The newest six-hourly start at least six hours old, by this browser's clock: what When opens on at once, before
// the server has said which start the download accepts.
function localCycle() {
  const d = new Date(Date.now() - 6 * 3600000);
  return `${dayText(d)}T${hourText(Math.floor(d.getUTCHours() / 6) * 6)}`;
}

// Days back from now at a UTC hour, as YYYY-MM-DDTHH.
function daysAgo(days, hour) {
  const d = new Date(Date.now() - days * 86400000);
  return `${dayText(d)}T${hourText(hour)}`;
}

// The link keeps the draft: #/create/LAT,LON,WxH/STEP/at/CYCLE/src/SOURCE/from/EVENT/nz/LEVELS, each pair optional.
function readRoute(args) {
  const out = { box: null, step: null, cycle: null, source: null, event: null, card: null, nz: null };
  if (args[0] === "recipe") {
    out.event = args[1] || null;
    out.card = /^\d+$/.test(args[2] || "") ? args[2] : null;
    return out;
  }
  const spec = /^(-?[\d.]+),(-?[\d.]+),(\d+)x(\d+)$/.exec(args[0] || "");
  if (spec) {
    out.box = { lat: Number(spec[1]), lon: Number(spec[2]), width_km: Number(spec[3]), height_km: Number(spec[4]) };
    if (STEPS.includes(args[1])) out.step = args[1];
  }
  for (let i = 2; i + 1 < args.length; i += 2) {
    if (args[i] === "at" && /^\d{4}-\d{2}-\d{2}T\d{2}$/.test(args[i + 1])) out.cycle = args[i + 1];
    if (args[i] === "src") out.source = args[i + 1];
    if (args[i] === "from") out.event = args[i + 1];
    if (args[i] === "card" && /^\d+$/.test(args[i + 1])) out.card = args[i + 1];
    if (args[i] === "nz" && /^\d+$/.test(args[i + 1])) out.nz = Number(args[i + 1]);
  }
  return out;
}

// The draft kept for this browser tab, so a reload of New forecast comes back to every choice, not only the ones
// the link spells. Kept under the link it was drawn on: another link, or another tab, starts afresh. Storage can be
// missing or refused (a private window, blocked site data); the page then works as before, without the keeping.
const KEPT = "gpuwm.create.draft";
const KEPT_FIELDS = ["box", "source", "cycle", "hours", "grid", "dx", "card", "cardTouched", "nz", "name", "profile",
  "products", "productsNamed", "render_section", "ladder", "clock", "startHour", "picks", "machine", "wholeCycle"];
function keepDraft(link, draft) {
  try {
    sessionStorage.setItem(KEPT, JSON.stringify({ link, draft: Object.fromEntries(KEPT_FIELDS.map((k) => [k, draft[k]])) }));
  } catch (_) { /* nothing kept; a reload starts from the link */ }
}
function keptDraft(link) {
  try {
    const kept = JSON.parse(sessionStorage.getItem(KEPT) || "null");
    if (!kept || kept.link !== link || !kept.draft) return null;
    return Object.fromEntries(KEPT_FIELDS.filter((k) => k in kept.draft).map((k) => [k, kept.draft[k]]));
  } catch (_) {
    return null;
  }
}
function forgetDraft() {
  try { sessionStorage.removeItem(KEPT); } catch (_) { /* nothing was kept */ }
}

// "Find an event": a search over the wiki's events; picking one opens New forecast filled from its recipe.
function eventFinder(w) {
  const input = h("input", { class: "input", type: "search", placeholder: w.find_placeholder, spellcheck: "false",
    "aria-label": w.find_label, autocomplete: "off" });
  const list = h("div", { class: "finderlist", role: "listbox", hidden: true });
  let timer = null;
  let asked = 0;
  // The words the rows in the list were found for; a search asked for other words, or left, draws nothing.
  let shownFor = null;
  // No answer on its way may be drawn: the words were changed or cleared, or the search was put away.
  function forget() { clearTimeout(timer); asked += 1; }
  async function ask() {
    const q = input.value.trim();
    if (q.length < 2) { list.hidden = true; return; }
    const mine = ++asked;
    let rows = [];
    try {
      const data = await api.get(`/api/wiki/search?q=${encodeURIComponent(q)}&sort=score`);
      rows = (data.results || []).filter((r) => r.kind === "event").slice(0, 7);
    } catch (err) {
      rows = [];
    }
    if (mine !== asked) return;
    list.replaceChildren(...(rows.length ? rows.map((r) => {
      const item = h("button", { type: "button", class: "finderrow", role: "option" },
        h("b", {}, r.title), h("span", { class: "dim" }, [r.start ? String(r.start).slice(0, 10) : "", r.type_title || ""]
          .filter(Boolean).join(" · ")));
      item.addEventListener("click", () => { list.hidden = true; go(`create/recipe/${encodeURIComponent(r.id)}`); });
      return item;
    }) : [h("p", { class: "note" }, w.find_none)]));
    shownFor = q;
    list.hidden = false;
  }
  input.addEventListener("input", () => {
    forget();
    // the rows were found for the words before this change
    list.hidden = true;
    list.replaceChildren();
    shownFor = null;
    timer = setTimeout(ask, 220);
  });
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") { forget(); list.hidden = true; input.blur(); }
    // Only a choice on screen, found for these words, is opened.
    if (ev.key === "Enter" && !list.hidden && shownFor === input.value.trim()) {
      const first = list.querySelector(".finderrow");
      if (first) first.click();
    }
  });
  input.addEventListener("focus", () => {
    if (list.childElementCount && shownFor === input.value.trim()) list.hidden = false;
    else if (input.value.trim().length >= 2) { clearTimeout(timer); timer = setTimeout(ask, 0); }
  });
  const finder = h("div", { class: "finder" }, h("span", { class: "cap" }, w.find_label), input, list);
  const away = (ev) => { if (!finder.contains(ev.target)) { forget(); list.hidden = true; } };
  document.addEventListener("click", away);
  finder.close = () => { forget(); document.removeEventListener("click", away); };
  return finder;
}

async function render(body, args, page) {
  const w = page.words.screens.create;
  const ww = page.words.wiki.create;
  page.setCrumbs([[page.words.screens.shell.my_forecasts, "#/runs"], [w.title]]);
  // The map, the Where step and the When step draw at once; the data source list and the card come from the engine
  // (slow the first time), asked for separately, and only the source rows, How fine and Review wait on them.
  let sources = [];
  let cycleTouched = false;
  // The start the draft opened on by itself, which a start the download is known to take may still replace.
  let openedOn = null;
  let machines = null;
  let queueInfo = null;
  let queueTimer = null;
  let cards = [];
  let device = null;
  let ready = false;
  let loadError = null;

  const route = readRoute(args);
  // The one draft: every choice on every step lives here, what the page sends is built from it alone, and it is
  // kept for this tab so a reload comes back to the same choices.
  const draft = {
    box: route.box,
    source: route.source || "",
    cycle: route.cycle,
    hours: 6,
    // a named grid (its index), or a typed spacing in km under More settings, which wins over it
    grid: 1,
    dx: null,
    card: "16gb",
    cardTouched: false,
    // Vertical levels: null is the engine's own ladder; a number goes to the engine as its level count (--nz).
    nz: route.nz,
    // More settings
    name: "",
    profile: "",
    products: "",
    // "Named pictures or a cross-section" is its own choice, kept apart from the names typed under it: read off the
    // names, an emptied box was the standard set, so the run drew that set while the page still showed the choice.
    productsNamed: false,
    render_section: "",
    ladder: "",
    // How the run steps: "" is the engine's choice (--clock auto), else adaptive or fixed.
    clock: "",
    startHour: 0,
    // Run as the source posts (the default): start at the first hours and wait at an hour not posted yet. On,
    // "Wait for the whole cycle" starts once the cycle's last hour is posted.
    wholeCycle: false,
    // the physics schemes picked per family
    picks: {},
    machine: "this-computer",
  };
  let step = route.step || "where";
  let closed = false;
  let recipe = null;
  if (route.event) {
    try {
      recipe = await api.get(`/api/wiki/recipe/${encodeURIComponent(route.event)}${route.card ? `?card=${route.card}` : ""}`);
    } catch (err) {
      notice(errorText(err), "warn");
    }
  }
  const grids = w.grids;
  // The recipe's box as the draft holds it, so a box the person moved is told apart from the event's own.
  const recipeBox = recipe ? { lat: +Number(recipe.lat).toFixed(2), lon: +Number(recipe.lon).toFixed(2),
    width_km: Math.round(recipe.width_km), height_km: Math.round(recipe.height_km) } : null;
  if (recipe) {
    // The recipe's box, unless the link already holds one the person moved.
    if (!draft.box) draft.box = { ...recipeBox };
    draft.hours = recipe.hours;
    if (recipe.card) draft.card = recipe.card;
    if (recipe.nz && draft.nz === null) draft.nz = Number(recipe.nz);
    // A physics set or a forecast hour to start from, when the best run names one, is filled in as the rest is.
    if (recipe.profile) draft.profile = String(recipe.profile);
    if (recipe.start_hour) draft.startHour = Number(recipe.start_hour);
    const same = grids.findIndex((g) => g.dx_km === recipe.dx_km);
    if (same >= 0) draft.grid = same; else draft.dx = Number(recipe.dx_km);
    if (args[0] === "recipe") step = "when";
  }
  draft.name = recipe ? eventRunName(recipe) : stamp();

  // A reload brings back the draft this tab had on this same link. A link past When that this tab never drew
  // holds only the box, date and source, so the page starts again at Where and says so rather than showing a
  // Review of choices nobody made.
  const kept = keptDraft(location.hash);
  const restored = !!kept;
  if (kept) {
    Object.assign(draft, kept);
  } else if (["fine", "physics", "review"].includes(step)) {
    step = "where";
    notice(w.draft_reset, "warn");
  }

  // ---- the map, left
  const canvas = h("canvas", { class: "map", "aria-label": w.where_line });
  const tip = h("div", { class: "maptip" }, w.no_box);
  const zoomIn = button("+", { title: w.zoom_in, onclick: () => map.zoom(1.5) });
  const zoomOut = button("−", { title: w.zoom_out, onclick: () => map.zoom(1 / 1.5) });
  const zoomBox = button("□", { title: w.zoom_box, onclick: () => map.frameBox() });
  const where = h("span", { class: "where" });
  const credit = h("span", { class: "credit" }, w.map_credit);
  const mapPanel = h("section", { class: "panel flush" },
    h("div", { class: "mapwrap" }, canvas, tip, h("div", { class: "mapctl" }, zoomIn, zoomOut, zoomBox)),
    h("div", { class: "mapfoot" },
      h("span", { class: "key" }, h("i", { class: "sw" }), w.map_ask),
      h("span", { class: "key" }, h("i", { class: "sw fit" }), w.map_fit),
      where, credit));

  // ---- the step panel, right
  const stepsBar = h("nav", { class: "steps", "aria-label": "Steps" });
  const panel = h("section", { class: "panel steppanel" });
  const finder = eventFinder(w);
  body.append(h("div", { class: "createtop" }, stepsBar, finder), h("div", { class: "grid cols-map" }, mapPanel, panel));

  // ---- More settings: the fields a guided run leaves to the engine. Each writes into the draft as it changes,
  // and shows the draft's value whenever the draft is changed by anything else (a grid click, the assistant).
  const name = h("input", { class: "input", value: draft.name, spellcheck: "false" });
  const lat = h("input", { class: "input num", type: "number", step: "0.01", min: "-85", max: "85" });
  const lon = h("input", { class: "input num", type: "number", step: "0.01", min: "-180", max: "180" });
  const width = h("input", { class: "input num", type: "number", min: "50", max: "8000", step: "10" });
  const height = h("input", { class: "input num", type: "number", min: "50", max: "8000", step: "10" });
  const dx = h("input", { class: "input num", type: "number", min: "0.5", max: "50", step: "0.25", placeholder: "" });
  const profile = h("select", { class: "input" });
  const products = h("select", { class: "input" },
    h("option", { value: "" }, w.products_standard), h("option", { value: "all" }, w.products_all),
    h("option", { value: "none" }, w.products_none), h("option", { value: "custom" }, w.products_custom));
  const customProducts = h("input", { class: "input", spellcheck: "false", placeholder: "xsec:wa" });
  const section = h("input", { class: "input", spellcheck: "false", placeholder: "lat,lon,lat,lon" });
  const customFields = h("div", { class: "wide fields", style: "grid-column: 1 / -1" },
    h("label", { class: "wide" }, w.products_named, customProducts),
    h("label", { class: "wide" }, w.section, section, h("span", { class: "hint" }, w.section_hint)));
  const ladder = h("select", { class: "input" });
  const clock = h("select", { class: "input" }, h("option", { value: "" }, w.clock_auto),
    h("option", { value: "adaptive" }, w.clock_adaptive), h("option", { value: "fixed" }, w.clock_fixed));
  const startHour = h("input", { class: "input num", type: "number", min: "0", max: "384", step: "1", value: "0" });
  const wholeCycle = h("input", { type: "checkbox" });
  function showDraft() {
    name.value = draft.name;
    dx.value = draft.dx === null ? "" : String(draft.dx);
    const named = draft.productsNamed || !["", "all", "none"].includes(draft.products);
    products.value = named ? "custom" : draft.products;
    customProducts.value = named ? draft.products : "";
    section.value = draft.render_section || "";
    customFields.hidden = !named;
    startHour.value = String(draft.startHour);
    wholeCycle.checked = !!draft.wholeCycle;
    if ([...profile.options].some((o) => o.value === draft.profile)) profile.value = draft.profile;
    if ([...ladder.options].some((o) => o.value === draft.ladder)) ladder.value = draft.ladder;
    clock.value = ["adaptive", "fixed"].includes(draft.clock) ? draft.clock : "";
  }
  function fillLadders(names) {
    ladder.replaceChildren(h("option", { value: "" }, w.ladder_none),
      ...(names || []).map((id) => h("option", { value: id }, id === "auto" ? w.ladder_auto : id.split("-").map((km) => `${km} km`).join(", "))));
    if (![...ladder.options].some((o) => o.value === draft.ladder)) draft.ladder = "";
    ladder.value = draft.ladder;
  }
  // The physics composer: the families and schemes of the chosen source, and the engine's check of the whole set
  // at this grid, place and time. The picks themselves live in the draft.
  const physics = { catalogs: new Map(), checks: new Map(), timer: null, waiting: null };
  function fillProfiles() {
    // Before the source list is in there is nothing to offer, and a kept set must not be dropped for that.
    if (!sources.length) return;
    const s = sources.find((row) => row.id === draft.source);
    profile.replaceChildren(h("option", { value: "" }, w.profile_default),
      ...(s ? s.profiles.map((p) => h("option", { value: p.id, title: p.summary || "" }, p.id)) : []));
    if (![...profile.options].some((o) => o.value === draft.profile)) draft.profile = "";
    profile.value = draft.profile;
  }
  function setProfile(id) {
    draft.profile = id || "";
    profile.value = draft.profile;
  }
  const sourceName = (id) => (sources.find((s) => s.id === id) || {}).name || id;

  const dxOf = () => (draft.dx !== null ? draft.dx : grids[draft.grid] ? grids[draft.grid].dx_km : null);
  const picked = () => Object.keys(draft.picks).length > 0;
  const sameBox = (a, b) => !!(a && b && a.lat === b.lat && a.lon === b.lon && a.width_km === b.width_km
    && a.height_km === b.height_km);
  // The event's own layout at a spacing: its nests (the same chain and buffers the event page's button runs) while
  // the grid is the event's outer spacing and no nest ladder is set, and its storm-following nest while the box,
  // levels and physics are the event's own as well, since the cyclone setup sizes all three itself. Computed from
  // the spacing and levels asked about, never the ones on screen, so a choice's preview is the plan its click gives.
  function eventLayout(dxKm, nz) {
    const own = !!(recipe && !draft.ladder && dxKm === recipe.dx_km);
    const nests = own && !!recipe.chain;
    const follows = own && !!(recipe.following && recipe.cyclone_setup) && nz === null && !picked() && !draft.profile
      && sameBox(draft.box, recipeBox);
    // The layout is named by its event and the card size its Customise opened with while its grids are the event's
    // own: the server reads the rest of the best run from the storm wiki (how often each grid writes, a cyclone's
    // sea-surface flux, a following nest's cyclone setup), as the event page's button runs it.
    const named = own && !!recipe.layout;
    return { chain: nests ? recipe.chain : null, buffer_km: nests ? recipe.buffer_km || null : null,
      following: follows, event: named ? recipe.event : null, recipe_card_gb: named ? recipe.card_gb : null };
  }
  // The event's own grids at a spacing: as the fit priced them, or before it answers as the best run has them.
  function layoutSpacings(fit) {
    const doms = fit && fit.domains && fit.domains.length ? fit.domains
      : (recipe && recipe.layout && recipe.layout.domains) || [];
    return doms.map((d) => spacing(d.dx_km));
  }
  const keepsLayout = (layout) => !!(layout.chain || layout.following);
  // How often the event's best run writes each grid, in words.
  function every(seconds) {
    const s = Number(seconds);
    if (s === 3600) return "hour";
    return s % 3600 === 0 ? `${s / 3600} hours` : `${Math.round(s / 60)} minutes`;
  }
  // Whether the plan runs the rest of the event's best run (how often it writes, its physics): a cyclone setup's
  // only with its storm-following nest, since the setup is what sets them, and any other best run's while the
  // draft keeps its grids, since the server carries them then. Review and the Physics note name them only then.
  function bestRunRides() {
    const layout = eventLayout(dxOf(), draft.nz);
    return recipe && recipe.cyclone_setup ? layout.following : !!layout.event;
  }
  function outputWords() {
    const out = recipe && recipe.layout && recipe.layout.output;
    if (!out || !out.history_interval_s || !bestRunRides()) return null;
    return out.nest_history_interval_s
      ? fill(ww.output_event, { outer: every(out.history_interval_s), nests: every(out.nest_history_interval_s) })
      : fill(ww.output_event_one, { outer: every(out.history_interval_s) });
  }
  function picturesWords() {
    if (draft.productsNamed || !["", "all", "none"].includes(draft.products)) return draft.products || w.products_custom;
    return draft.products === "all" ? w.products_all : draft.products === "none" ? w.products_none : w.products_standard;
  }
  const payload = (dxKm = dxOf(), nz = draft.nz) => {
    const layout = eventLayout(dxKm, nz);
    return {
      name: draft.name.trim(), source: draft.source, cycle: draft.cycle,
      lat: draft.box ? draft.box.lat : null, lon: draft.box ? draft.box.lon : null,
      width_km: draft.box ? draft.box.width_km : 600, height_km: draft.box ? draft.box.height_km : 600,
      // A nest ladder fixes the spacing itself, so the grid's spacing is left out beside it.
      hours: draft.hours, dx_km: draft.ladder ? null : dxKm, nz, profile: draft.profile || null, card: draft.card,
      // Named with an empty list goes to the server as that, and the server refuses it rather than draw the standard set.
      products: draft.products || null, products_named: draft.productsNamed || null,
      render_section: draft.render_section || null,
      ladder: draft.ladder || null, clock: draft.clock || null, start_hour: draft.startHour,
      whole_cycle: draft.wholeCycle ? true : null,
      physics_choices: picked() ? { ...draft.picks } : null,
      ...layout,
      era5_provider: recipe && recipe.era5_provider && draft.source === recipe.source ? recipe.era5_provider : null,
    };
  };
  // What the event's layout does at the chosen grid, in one line: kept, or which choice drops it.
  function layoutLine() {
    if (!recipe || !(recipe.chain || (recipe.following && recipe.cyclone_setup))) return null;
    const now = eventLayout(dxOf(), draft.nz);
    const words = { dx: spacing(recipe.dx_km) };
    if (recipe.following && recipe.cyclone_setup) {
      return now.following ? h("p", { class: "note" }, fill(w.layout_following, words))
        : h("p", { class: "fitline no" }, fill(w.layout_following_dropped, words));
    }
    return now.chain ? h("p", { class: "note" }, fill(w.layout_nests, words))
      : h("p", { class: "fitline no" }, fill(w.layout_nests_dropped, words));
  }

  // ---- what each source holds for the chosen start and length, asked for whenever either changes. The server
  // answers at once; a row still being checked says so, and the page asks again until every row is in.
  const avail = new Map();
  let switched = null;
  const availKey = () => `${draft.cycle}|${draft.hours}`;
  function availNow() {
    const entry = avail.get(availKey());
    return entry && entry.done ? entry.done : null;
  }
  // While a row is checking the page asks every second; while a row is unconfirmed (its host was not heard in time),
  // or a newer start than the chosen source's newest found went unheard, it asks every 15 s, and the server puts
  // that host to the probe again once a minute.
  const POLL_MS = 1000;
  const RECHECK_MS = 15000;
  const POLL_LIMIT = 90;
  function askAvail(force = false) {
    const key = availKey();
    if (!draft.cycle) return;
    const had = avail.get(key);
    const idle = had && !had.flying && !had.timer;
    if (had && !force && !(idle && had.done && (had.done.error || had.done.pending > 0))) return;
    const entry = had || { done: null, polls: 0, flying: false, timer: null };
    avail.set(key, entry);
    entry.flying = true;
    clearTimeout(entry.timer);
    entry.timer = null;
    const [cycle, hours] = key.split("|");
    api.get(`/api/sources/availability?time=${encodeURIComponent(cycle)}&hours=${hours}`)
      .then((data) => { entry.done = data; })
      .catch((err) => {
        // A failed ask keeps the rows it already had; with none it says why and offers to ask again.
        if (!entry.done || entry.done.error) entry.done = { error: errorText(err), sources: [] };
      })
      .then(() => {
        entry.flying = false;
        if (closed) return;
        const data = entry.done;
        const live = data && !data.error;
        const wait = live && data.pending > 0 ? POLL_MS
          : live && data.sources.some((row) => row.basis === "unchecked"
            || (row.id === draft.source && row.newest_basis === "unchecked")) ? RECHECK_MS : 0;
        // Only the start on screen is asked again; going back to another one asks it afresh.
        if (wait && entry.polls < POLL_LIMIT && key === availKey()) {
          entry.polls += 1;
          entry.timer = setTimeout(() => { entry.timer = null; if (!closed && key === availKey()) askAvail(true); }, wait);
        }
        if (key !== availKey() || !live) { if (key === availKey()) availChanged(false); return; }
        const mine = data.sources.find((row) => row.id === draft.source);
        // A draft the person has not moved follows the opening start as the checks answer (openingOf).
        if (!cycleTouched && openedOn && openedOn === draft.cycle && mine && ready) {
          const target = openingOf(mine);
          if (target && target !== draft.cycle) {
            openedOn = target;
            setCycle(target, { user: false });
            return;
          }
        }
        // A source that does not hold this start is swapped for the first one that does, and the page says so;
        // only once the chosen source's own check is in, and only once the source list has named the page's own
        // first choice: rows that came back before it picked whichever source answered first.
        const settled = mine && !mine.checking;
        let moved = false;
        if (ready && (!mine || (settled && mine.starts === "no")) && data.best && data.best !== draft.source) {
          switched = draft.source && mine ? { from: draft.source, to: data.best, why: mine.why } : null;
          draft.source = data.best;
          fillProfiles();
          moved = true;
        }
        availChanged(moved);
      });
  }
  // An answer that arrives on its own changes only what it answers: the source rows, the Newest pick, the faint days
  // and Next, in place. Redrawing the whole step on every poll took the keyboard from the date box after each key.
  // A source swapped for another one redraws the step, since the hours and the physics follow the source.
  function availChanged(sourceMoved) {
    if (closed) return;
    if (sourceMoved || step !== "when" || paintedStep !== "when") {
      if (sourceMoved) paintStep();
      return;
    }
    fillWhen();
  }
  const sourceRow = (id) => (availNow() ? availNow().sources.find((row) => row.id === id) : null);
  // What the page knows of a source before its row has answered: its hours and usual publication delay.
  const factsOf = (id) => sourceRow(id) || sources.find((row) => row.id === id) || null;
  // The start before one, on the hours that serve the length.
  function earlierStart(row, cycle) {
    const d = new Date(`${cycle}:00:00Z`);
    for (let i = 0; i < 24 * 14; i++) {
      d.setUTCHours(d.getUTCHours() - 1);
      if (row.cycle_hours.includes(d.getUTCHours())) break;
    }
    return `${dayText(d)}T${hourText(d.getUTCHours())}`;
  }
  // The newest start past a source's usual publication delay, by this browser's clock.
  function dueOf(row) {
    if (!row || row.usual_delay_hours == null || !(row.cycle_hours || []).length) return null;
    const d = new Date(Date.now() - row.usual_delay_hours * 3600000);
    const at = `${dayText(d)}T${hourText(d.getUTCHours())}`;
    const mine = row.cycle_hours.includes(d.getUTCHours()) ? at : earlierStart(row, at);
    // A row's own due start also knows what the clock cannot: an analysis window's end, an archive's published end.
    return row.due && row.due < mine ? row.due : mine;
  }
  // The one rule (gpuwm/gui/availability.py): the page opens on the newest start Start takes, the newest start a
  // check confirmed whole or the newest start past the source's usual publication delay that no check found
  // missing, whichever is newer. Until a check confirms one, that is the delay's own start. A draft the person has
  // not moved follows it.
  function openingOf(row) {
    if (!row) return null;
    const missing = new Set(row.missing || []);
    let due = dueOf(row);
    for (let i = 0; due && missing.has(due) && i < missing.size; i++) due = earlierStart(row, due);
    const found = row.confirmed || null;
    return found && (!due || found >= due) ? found : due;
  }
  // The Newest run: the server's, and only while the page opens on it. The server names one only once a check found
  // it whole and every start after it not yet whole; a start found while a newer one went unheard is not named.
  function newestOf(row) {
    const at = openingOf(row);
    return at && row && row.newest_run && at === row.newest_run ? at : null;
  }

  // ---- fits: one per grid and level choice, cached on everything that changes the answer
  const fits = new Map();
  // The picks stay in the key: whether a cumulus scheme was picked decides the cumulus the fit is sized with.
  // A fit's key leaves the pictures out, so its body leaves out the named-pictures choice: an empty named list refused
  // under a key that ignores it would stay refused after the names were typed.
  const fitBody = (dxKm, nz) => ({ ...payload(dxKm, nz), products_named: null, whole_cycle: null });
  const fitKey = (dxKm, nz = draft.nz) => JSON.stringify({ ...fitBody(dxKm, nz), name: "", products: null });
  // The auto ladder's depth is its fit's. The grid its last answered fit landed on is kept for everything but the
  // physics, so a pick (which changes the fit's key) still has its check read the default of the ladder the run is
  // fitted to, and the check, Start and the night check name the physics the run carries.
  const autoGrids = new Map();
  const gridKey = (dxKm, nz = draft.nz) => {
    const { profile, physics_choices, ...rest } = fitBody(dxKm, nz);
    return JSON.stringify({ ...rest, name: "", products: null });
  };
  const fittedGrid = (fit) => {
    const spacings = ((fit && fit.domains) || []).map((d) => d.dx_km).filter((dx) => dx > 0);
    return spacings.length ? { finest_dx_km: Math.min(...spacings), domains: spacings.length } : null;
  };
  const autoGrid = () => (draft.ladder === "auto" ? autoGrids.get(gridKey(dxOf())) || null : null);
  function fitFor(dxKm, nz = draft.nz) {
    const key = fitKey(dxKm, nz);
    if (!fits.has(key)) {
      const auto = draft.ladder === "auto" ? gridKey(dxKm, nz) : null;
      const promise = api.post("/api/create/fit", fitBody(dxKm, nz))
        .then((reply) => ({ ok: true, fit: reply.fit, command: reply.command }))
        .catch((err) => ({ ok: false, message: err.message, fix: err.fix, memory: err.body ? err.body.memory : null }));
      fits.set(key, { promise, done: null });
      promise.then((result) => {
        fits.get(key).done = result;
        const grid = auto && result.ok ? fittedGrid(result.fit) : null;
        if (grid) autoGrids.set(auto, grid);
        if (!closed) paintStep();
      });
    }
    return fits.get(key);
  }
  // A step asks for fits while it draws; they start once the box has stopped moving, so dragging the box does not
  // start one engine call per mouse move.
  const wanted = new Map();
  let fitTimer = null;
  function want(dxKm, nz = draft.nz) {
    const entry = fits.get(fitKey(dxKm, nz));
    if (entry) return entry;
    wanted.set(fitKey(dxKm, nz), [dxKm, nz]);
    clearTimeout(fitTimer);
    fitTimer = setTimeout(() => {
      const list = [...wanted.values()];
      wanted.clear();
      for (const [d, n] of list) fitFor(d, n);
    }, 600);
    return { done: null };
  }
  const currentFit = () => {
    if (!draft.box) return null;
    const entry = fits.get(fitKey(dxOf()));
    return entry && entry.done && entry.done.ok ? entry.done.fit : null;
  };
  // The level count a fit resolved to: the engine's own number, never one the page assumes.
  const fitLevels = (fit) => (fit && fit.domains && fit.domains[0] ? fit.domains[0].nz : null);

  // A refusal that is not about memory (the engine and its data package disagree, say) is the same for every
  // choice. When every checked choice was refused for one such reason, the step says it once above the choices.
  const refusedWhy = (r) => (r && !r.ok && !(r.memory && !r.memory.fits) ? [r.message, r.fix].filter(Boolean).join(" ") : null);
  let sharedWhy = null;
  function sharedRefusal(results) {
    const done = results.filter(Boolean);
    const whys = done.map(refusedWhy);
    if (done.length < 2 || whys.some((why) => why === null)) return null;
    return new Set(whys).size === 1 ? whys[0] : null;
  }
  const sharedOff = (result) => sharedWhy !== null && refusedWhy(result) === sharedWhy;

  function fitLine(result, tooBig = w.too_big) {
    if (!result) return h("div", { class: "fit wait" }, w.checking);
    if (sharedOff(result)) return null;
    if (result.ok) {
      const f = result.fit;
      const root = f.domains[0];
      const mem = f.memory;
      const fitsText = mem ? fill(w.fits, { need: gb(mem.need_gib), budget: gb(mem.budget_gib) }) : w.fits_plain;
      // A fit of several grids names each one, so nests are never read as one grid.
      const points = f.domains.length > 1
        ? fill(w.points_many, { levels: root.nz, grids: f.domains.map((d) => fill(w.grid_points,
          { dx: spacing(d.dx_km), nx: d.nx, ny: d.ny })).join(", ") })
        : root ? fill(w.points, { nx: root.nx, ny: root.ny, levels: root.nz }) : "";
      return h("div", { class: "fit ok" }, `${fitsText}. `, points);
    }
    const mem = result.memory;
    return h("div", { class: "fit no" },
      mem && !mem.fits ? fill(tooBig, { need: gb(mem.need_gib), budget: gb(mem.budget_gib) }) : [result.message, result.fix].filter(Boolean).join(" "));
  }

  const map = boxMap(canvas, {
    defaultKm: 600,
    view: draft.box ? { lat: draft.box.lat, lon: draft.box.lon, k: 13 } : null,
    onchange(box) {
      draft.box = box;
      syncFields();
      paintStep();
    },
  });

  function syncFields() {
    const b = draft.box;
    lat.value = b ? String(b.lat) : ""; lon.value = b ? String(b.lon) : "";
    width.value = b ? String(b.width_km) : ""; height.value = b ? String(b.height_km) : "";
    tip.hidden = !!b;
    where.textContent = b ? `${b.lat.toFixed(2)}, ${b.lon.toFixed(2)}  ${b.width_km} × ${b.height_km} km` : "";
  }
  function fromFields() {
    if (lat.value === "" || lon.value === "") return;
    draft.box = { lat: Number(lat.value), lon: Number(lon.value),
      width_km: Number(width.value) || 600, height_km: Number(height.value) || 600 };
    map.setBox(draft.box);
    syncFields();
    paintStep();
  }
  for (const el of [lat, lon, width, height]) el.addEventListener("change", fromFields);
  name.addEventListener("input", () => { draft.name = name.value; keepNow(); });
  dx.addEventListener("change", () => {
    const n = Number(dx.value);
    draft.dx = dx.value === "" || !Number.isFinite(n) ? null : n;
    paintStep();
  });
  products.addEventListener("change", () => {
    draft.productsNamed = products.value === "custom";
    draft.products = draft.productsNamed ? (customProducts.value || "xsec:wa") : products.value;
    if (!draft.productsNamed) draft.render_section = "";
    showDraft();
    paintStep();
  });
  customProducts.addEventListener("input", () => { draft.products = customProducts.value; keepNow(); });
  section.addEventListener("input", () => { draft.render_section = section.value; keepNow(); });
  ladder.addEventListener("change", () => { draft.ladder = ladder.value; paintStep(); });
  clock.addEventListener("change", () => { draft.clock = clock.value; paintStep(); });
  startHour.addEventListener("change", () => {
    const n = Number(startHour.value || 0);
    draft.startHour = Number.isInteger(n) && n >= 0 ? n : 0;
    paintStep();
  });
  wholeCycle.addEventListener("change", () => { draft.wholeCycle = wholeCycle.checked; paintStep(); });
  // A set picked by name under More settings replaces the rows picked on the Physics step.
  profile.addEventListener("change", () => { draft.picks = {}; draft.profile = profile.value; paintStep(); });

  function setStep(next) {
    const early = step === "where" || step === "when";
    step = next;
    paintStep();
    // Past When, the map closes in on the box so the fitted grid shows beside it.
    if (early && (next === "fine" || next === "physics" || next === "review")) map.frameBox();
  }

  function nav2(backTo, nextTo, nextOn = true) {
    return h("div", { class: "nav2" },
      backTo ? button(w.back, { onclick: () => setStep(backTo) }) : h("span", {}),
      nextTo ? button(w.next, { kind: "primary", disabled: !nextOn, onclick: () => setStep(nextTo) }) : null);
  }

  function paintSteps() {
    const at = STEPS.indexOf(step);
    stepsBar.replaceChildren(...STEPS.map((s, i) => {
      const reachable = i === 0 || draft.box;
      const b = h("button", { type: "button", class: `${s === step ? "on" : ""}${i < at ? " done" : ""}`.trim() || null,
        disabled: reachable ? null : true }, h("b", {}, String(i + 1)), w.steps[s]);
      b.addEventListener("click", () => setStep(s));
      return b;
    }));
    page.setData(`${STEPS.indexOf(step) + 1} / ${STEPS.length}  ${w.steps[step]}`);
    const b = draft.box;
    const parts = [];
    // Before the source list answers, the link keeps a start and source it was opened with (or a start already
    // picked): a link rewritten without them lost them to the server's opening start when the app drew the page
    // again from the rewritten link (a second hashchange during the first load).
    if (draft.cycle && (ready || cycleTouched)) parts.push("at", draft.cycle);
    if (draft.source && (ready || route.source === draft.source)) parts.push("src", draft.source);
    if (recipe) parts.push("from", encodeURIComponent(recipe.event));
    if (recipe && recipe.card_gb) parts.push("card", String(recipe.card_gb));
    if (draft.nz) parts.push("nz", String(draft.nz));
    const link = b ? `#/create/${b.lat},${b.lon},${b.width_km}x${b.height_km}/${step}${parts.length ? `/${parts.join("/")}` : ""}` : "#/create";
    if (location.hash !== link) history.replaceState(null, "", link);
    keepNow();
  }
  // The draft is kept for this tab under the link it was drawn on; only once the source list is in, so a reload
  // taken before it never keeps a source the page had not yet checked.
  function keepNow() {
    if (!closed && ready) keepDraft(location.hash, draft);
  }

  function whereStep() {
    const b = draft.box;
    const km = b ? Math.max(b.width_km, b.height_km) : null;
    const sizes = h("div", { class: "choices" }, w.sizes.map((s) => choiceButton(s.name, `${s.km} km`, km === s.km && b.width_km === b.height_km, () => {
      const box = draft.box || { lat: map.view().lat, lon: map.view().lon };
      draft.box = { lat: +box.lat.toFixed(2), lon: +box.lon.toFixed(2), width_km: s.km, height_km: s.km };
      map.setBox(draft.box);
      syncFields();
      paintStep();
    })));
    return [
      h("h2", {}, w.where_title), h("p", { class: "note" }, w.where_line),
      h("div", { class: "cap sub" }, w.size), sizes,
      b ? h("dl", { class: "kv" }, h("dt", {}, w.centre), h("dd", { class: "mono" }, `${b.lat.toFixed(2)}, ${b.lon.toFixed(2)}`),
        h("dt", {}, w.size), h("dd", { class: "mono" }, `${b.width_km} × ${b.height_km} km`))
        : h("p", { class: "fitline wait" }, w.no_box),
      h("p", { class: "note" }, w.where_find),
      nav2(null, "when", !!b),
    ];
  }

  function waitStep(title, back) {
    return [
      h("h2", {}, title),
      loadError ? h("p", { class: "fitline no" }, loadError) : h("p", { class: "fitline wait" }, w.reading_sources),
      nav2(back, null),
    ];
  }

  // The wiki event this draft came from, and its recipe. On When it also says the start data, start and length
  // there are the event's and are changed right under it.
  function recipeBanner(when = false) {
    if (!recipe) return null;
    return h("div", { class: "fromevent" },
      h("b", {}, fill(ww.from_event, { title: recipe.title })), " ",
      h("a", { href: `#/event/${encodeURIComponent(recipe.event)}` }, ww.back),
      recipe.runnable
        ? h("p", { class: "note" }, fill(ww.recipe_line, { start: cycleWords(recipe.start_cycle), hours: recipe.hours, dx: spacing(recipe.dx_km),
          card: recipe.card_gb || cardWords(draft.card).replace(" GB", ""),
          nests: recipe.chain ? fill(ww.nests, { finest: spacing(recipe.dx_km / recipe.chain.split(",").reduce((a, r) => a * Number(r), 1)) }) : "" }))
        : h("p", { class: "fitline no" }, fill(ww.not_offered, { source: sourceName(recipe.source),
          date: cycleWords(recipe.cycle), hours: recipe.hours, dx: spacing(recipe.dx_km) })),
      when && recipe.runnable ? h("p", { class: "note" }, fill(ww.recipe_change, { source: sourceName(recipe.start_source) })) : null);
  }

  // ---- When: shortcuts, a typed date and hour, the calendar, then the sources that hold that start
  const typed = h("input", { class: "input num", type: "text", inputmode: "numeric", spellcheck: "false",
    placeholder: "YYYY-MM-DD", "aria-label": w.date, autocomplete: "off" });
  const typedNote = h("p", { class: "note typednote" });
  const hourPick = h("select", { class: "input num", "aria-label": w.hour },
    Array.from({ length: 24 }, (_, i) => h("option", { value: String(i) }, `${hourText(i)} UTC`)));
  function setCycle(cycle, { viewMonth = true, user = true } = {}) {
    if (user) cycleTouched = true;
    // A date half typed stays as typed when the page itself sets the start (the source list arriving, say).
    const editing = !user && document.activeElement === typed && typed.value !== String(draft.cycle || "").slice(0, 10);
    if (cycle !== draft.cycle) switched = null;
    draft.cycle = cycle;
    if (!editing) {
      typed.value = cycle.slice(0, 10);
      typedNote.textContent = "";
      typedNote.classList.remove("no");
    }
    hourPick.value = String(Number(cycle.slice(11, 13)));
    if (viewMonth) cal.set(cycle.slice(0, 10)); else cal.paint();
    paintStep();
  }
  function typedDate() {
    const read = parseTyped(typed.value);
    const bad = (text) => { typedNote.textContent = text; typedNote.classList.add("no"); };
    if (!read) return bad(w.date_bad);
    if (read.day < EARLIEST_DAY) return bad(fill(w.date_early, { day: EARLIEST_DAY }));
    if (read.day > todayText()) return bad(w.date_future);
    setCycle(`${read.day}T${hourText(read.hour === null ? Number(hourPick.value) : read.hour)}`);
  }
  // Typing a date is picking one: nothing moves the draft to another start under the person's keys.
  typed.addEventListener("input", () => { cycleTouched = true; });
  typed.addEventListener("change", typedDate);
  typed.addEventListener("keydown", (ev) => { if (ev.key === "Enter") typedDate(); });
  // The hour applies to the date in the box, typed or picked, so typing a date and then choosing the hour keeps both.
  hourPick.addEventListener("change", () => {
    const read = parseTyped(typed.value);
    const day = read && read.day >= EARLIEST_DAY && read.day <= todayText() ? read.day : draft.cycle.slice(0, 10);
    setCycle(`${day}T${hourText(Number(hourPick.value))}`, { viewMonth: day !== draft.cycle.slice(0, 10) });
  });
  const cal = calendar({
    value: todayText(), min: EARLIEST_DAY, words: w,
    // Faint: outside what the chosen source holds.
    faint(day) {
      const row = sourceRow(draft.source);
      if (!row) return false;
      // Past the newest run only when it is known to be the newest: past a start found while a newer one went
      // unheard, the days may hold published starts.
      const end = newestOf(row);
      return (row.earliest && day < row.earliest.slice(0, 10)) || (end && day > end.slice(0, 10));
    },
    onpick(day) { setCycle(`${day}T${hourText(Number(hourPick.value))}`, { viewMonth: false }); },
  });

  function shortcuts() {
    const newest = newestOf(sourceRow(draft.source));
    const picks = [];
    if (recipe && recipe.runnable) picks.push([ww.event_cycle, recipe.start_cycle]);
    if (newest) picks.push([w.newest, newest]);
    picks.push([w.yesterday, daysAgo(1, 12)], [w.week_ago, daysAgo(7, 12)], [w.year_ago, daysAgo(365, 12)]);
    return picks;
  }

  function sourceList() {
    const data = availNow();
    if (!data) return h("p", { class: "fitline wait" }, w.checking_date);
    if (data.error) {
      return h("div", { class: "btns" }, h("p", { class: "fitline no" }, data.error),
        button(w.sources_retry, { small: true, onclick: () => { askAvail(true); paintStep(); } }));
    }
    const has = data.sources.filter((row) => row.state === "yes");
    const maybe = data.sources.filter((row) => row.state === "unknown");
    const lacks = data.sources.filter((row) => row.state === "no");
    const option = (row) => {
      const on = row.id === draft.source;
      const own = sources.find((x) => x.id === row.id) || {};
      const facts = own.record_kind === "reanalysis" ? w.kind_reanalysis : own.record_kind === "analysis" ? w.kind_analysis
        : own.regional ? w.kind_regional : w.kind_global;
      // The age of an answer in whole minutes: "4 min", not "4 min 00 s".
      const mins = row.basis === "checked" && row.checked_age_s >= 60 ? Math.round(row.checked_age_s / 60) : 0;
      const age = mins ? fill(w.checked_ago, { age: mins >= 60 ? duration(mins * 60) : `${mins} min` }) : null;
      const el = h("button", { type: "button", class: `option src${on ? " on" : ""}${row.state === "yes" ? "" : " maybe"}`,
        "aria-pressed": on ? "true" : "false" },
      h("b", {}, row.name), h("span", { class: `v${row.checking ? " dim" : ""}` },
        row.checking ? w.checking_row : row.state === "yes" ? w.has_it
          : row.starts === "queue" && row.basis === "checked" ? w.not_yet : w.may_have_it),
      h("span", { class: "why" }, facts, age ? ` ${age}.` : ""),
      // What the server says of this start: why it may not have it, or how it knows it has it.
      row.note ? h("span", { class: "why" }, row.note)
        : row.state === "unknown" && row.why && !row.checking ? h("span", { class: "why" }, row.why) : null);
      el.addEventListener("click", () => { draft.source = row.id; switched = null; fillProfiles(); paintStep(); });
      return el;
    };
    const selectedLacks = lacks.find((row) => row.id === draft.source);
    // One line per reason: when every source says the same thing (a time still to come), it is said once.
    const reasons = new Map();
    for (const row of lacks) reasons.set(row.why, [...(reasons.get(row.why) || []), row.name]);
    const waitNote = !ready ? h("p", { class: "note" }, loadError || w.sources_wait) : null;
    const oneReason = reasons.size === 1 && !has.length && !maybe.length;
    // The first few in the suggested order, and the chosen one; the rest behind one fold. The chosen one shows here
    // also when it only may have the start (its data server was not heard, say), so its answer is in view.
    const shown = has.filter((row, i) => i < 3 || row.id === draft.source);
    const others = has.filter((row) => !shown.includes(row));
    const mineMaybe = maybe.find((row) => row.id === draft.source);
    const restMaybe = maybe.filter((row) => row !== mineMaybe);
    const top = mineMaybe ? [mineMaybe, ...shown] : shown;
    return [
      waitNote,
      switched ? h("p", { class: "fitline" }, fill(w.switched, { from: sourceName(switched.from), to: sourceName(switched.to), why: switched.why })) : null,
      top.length ? h("div", { class: "options" }, top.map(option))
        : oneReason ? null : h("p", { class: "fitline no" }, w.none_has),
      others.length ? keyed("others", fold(fill(w.others_fold, { n: others.length }), h("div", { class: "options" }, others.map(option)))) : null,
      selectedLacks && !oneReason ? h("p", { class: "fitline no" }, selectedLacks.why) : null,
      oneReason ? h("p", { class: "fitline no" }, lacks[0].why) : null,
      restMaybe.length ? keyed("maybe", fold(fill(w.maybe_fold, { n: restMaybe.length }), h("p", { class: "note" }, w.maybe_line),
        h("div", { class: "options" }, restMaybe.map(option)))) : null,
      lacks.length && !oneReason ? keyed("lacks", fold(fill(w.lacks_fold, { n: lacks.length }),
        h("ul", { class: "plain lacks" }, [...reasons].map(([why, names]) => h("li", {},
          h("b", {}, names.join(", ")), " ", h("span", { class: "dim" }, why)))))) : null,
    ];
  }
  // The source list's folds carry a name, so a redraw of the list keeps each open one open.
  function keyed(key, el) { el.dataset.fold = key; return el; }

  // The parts of When that answers change are kept from draw to draw and filled in place (fillWhen), so the date box,
  // the hour, the calendar's month and year boxes and a source row being clicked are never swapped out underneath
  // the person using them.
  const dateRow = h("div", { class: "datetime" }, field(w.date, typed), field(w.hour, hourPick));
  const quickBox = h("div", { class: "choices" });
  const srcCap = h("div", { class: "cap sub" });
  const srcBox = h("div", { class: "part" });
  const whenNote = h("p", { class: "note" });
  whenNote.hidden = true;
  let whenNext = null;
  const shown = { quick: "", sources: "", faint: "" };
  // Next goes on for any start Start or Queue it takes, checked or not: a check still out never blocks choosing.
  // Review offers only Queue it for a start Start does not take yet.
  const nextOk = () => { const row = sourceRow(draft.source); return !!row && row.starts !== "no"; };
  function fillWhen(force = false) {
    const picks = shortcuts();
    const quickKey = JSON.stringify([picks, draft.cycle]);
    if (force || quickKey !== shown.quick) {
      shown.quick = quickKey;
      quickBox.replaceChildren(...picks.map(([label, cycle]) => choiceButton(label, cycleWords(cycle),
        cycle === draft.cycle, () => setCycle(cycle))));
    }
    const data = availNow();
    srcCap.textContent = data ? fill(w.sources_for, { when: cycleWords(draft.cycle) }) : w.source;
    // Only a change a person would see redraws the rows; a poll that brings the same answers changes nothing.
    const rowsKey = JSON.stringify([ready, loadError, sources.length, draft.source, switched,
      data && (data.error || data.sources.map((r) => [r.id, r.state, r.starts, r.checking, r.basis, r.why, r.note,
        r.confirmed, r.newest_run, (r.missing || []).join(),
        r.basis === "checked" && r.checked_age_s >= 60 ? Math.round(r.checked_age_s / 60) : 0]))]);
    if (force || rowsKey !== shown.sources) {
      shown.sources = rowsKey;
      const open = new Set([...srcBox.querySelectorAll("details[data-fold]")].filter((d) => d.open).map((d) => d.dataset.fold));
      srcBox.replaceChildren(...[sourceList()].flat().filter(Boolean));
      srcBox.querySelectorAll("details[data-fold]").forEach((d) => { if (open.has(d.dataset.fold)) d.open = true; });
    }
    const own = sourceRow(draft.source);
    const faintKey = own ? `${draft.source}|${own.earliest}|${newestOf(own)}` : "";
    if (faintKey !== shown.faint) {
      shown.faint = faintKey;
      cal.shade();
    }
    if (whenNext) whenNext.disabled = !nextOk();
    const say = own && own.starts === "queue" && !own.checking ? w.start_queue_only : "";
    whenNote.textContent = say;
    whenNote.hidden = !say;
  }

  function whenStep() {
    askAvail();
    const src = sources.find((s) => s.id === draft.source);
    const horizon = src && !src.analysis && src.horizon_hours ? src.horizon_hours : 384;
    const hourChoices = [...new Set([...w.hour_choices, ...(recipe ? [recipe.hours] : [])])].sort((a, b) => a - b);
    const nav = nav2("where", "fine", nextOk());
    whenNext = nav.lastElementChild;
    nav.prepend(whenNote);
    fillWhen();
    const hoursRow = [h("div", { class: "cap sub" }, w.hours),
      h("div", { class: "choices" }, hourChoices.filter((x) => x <= horizon || (recipe && x === recipe.hours)).map((x) =>
        choiceButton(fill(w.hours_unit, { hours: x }), null, x === draft.hours, () => { draft.hours = x; paintStep(); })))];
    if (recipe && recipe.runnable) {
      // From an event the start is already picked: its start data and length come first, in view without
      // scrolling, and another start under them, the calendar folded.
      return [
        h("h2", {}, w.when_title), recipeBanner(true),
        srcCap, srcBox, ...hoursRow,
        h("div", { class: "cap sub" }, ww.other_start), quickBox, dateRow, typedNote,
        fold(ww.calendar_fold, cal.el),
        nav,
      ];
    }
    return [
      h("h2", {}, w.when_title), recipeBanner(true), h("p", { class: "note" }, w.when_line),
      h("div", { class: "cap sub" }, w.quick), quickBox,
      dateRow, typedNote,
      cal.el,
      srcCap,
      srcBox,
      ...hoursRow,
      nav,
    ];
  }

  function cardChoices() {
    const found = device && device.memory_total_gib ? device : null;
    return h("div", { class: "choices" }, cards.map((c) => choiceButton(cardWords(c),
      found && page.system && page.system.card === c ? fill(w.this_computer, { own: ownGb(found.memory_total_gib) }) : null,
      c === draft.card,
      () => { draft.card = c; draft.cardTouched = true; paintStep(); })));
  }

  function fineStep() {
    let budget = null;
    const custom = draft.nz !== null && !w.levels.some((l) => l.nz === draft.nz);
    sharedWhy = sharedRefusal([...grids.map((g) => want(g.dx_km).done), ...w.levels.map((l) => want(dxOf(), l.nz).done),
      ...(custom ? [want(dxOf(), draft.nz).done] : [])]);
    // The grid that keeps the event's layout is drawn as that layout, every grid of it named, and comes first: shown
    // as its outer grid alone, the event's three grids read as one.
    const drawn = grids.map((g, i) => {
      const entry = want(g.dx_km);
      const mem = entry.done && (entry.done.ok ? entry.done.fit.memory : entry.done.memory);
      if (mem && budget === null) budget = mem.budget_gib;
      const on = draft.dx === null && draft.grid === i;
      const layout = eventLayout(g.dx_km, draft.nz);
      const mine = keepsLayout(layout);
      const spacings = mine ? layoutSpacings(entry.done && entry.done.ok ? entry.done.fit : null) : [];
      const el = h("button", { type: "button", class: `option${on ? " on" : ""}${sharedOff(entry.done) ? " off" : ""}`, "aria-pressed": on ? "true" : "false" },
        h("b", {}, mine ? ww.layout_title : g.name),
        h("span", { class: "v" }, spacings.length > 1 ? spacings.join(" / ") : spacing(g.dx_km)),
        h("span", { class: "why" }, !mine ? g.why : fill(layout.following ? ww.layout_why_following : ww.layout_why,
          { grid: g.name, dx: spacing(g.dx_km), nests: spacings.slice(1).join(", ") })),
        fitLine(entry.done));
      el.addEventListener("click", () => { draft.grid = i; draft.dx = null; dx.value = ""; paintStep(); });
      return { el, mine };
    });
    const options = h("div", { class: "options" }, [...drawn.filter((d) => d.mine), ...drawn.filter((d) => !d.mine)].map((d) => d.el));
    const chosen = currentFit();
    return [
      h("h2", {}, w.fine_title), h("p", { class: "note" }, w.fine_line),
      h("div", { class: "cap sub" }, w.card), cardChoices(),
      h("p", { class: "note" }, budget === null ? fill(w.card_line, { card: cardWords(draft.card) })
        : fill(w.card_line_usable, { card: cardWords(draft.card), budget: gb(budget) })),
      sharedWhy !== null ? h("p", { class: "fitline no" }, sharedWhy) : null,
      options,
      layoutLine(),
      h("div", { class: "cap sub" }, w.levels_title), h("p", { class: "note" }, w.levels_line),
      levelChoices(),
      chosen ? h("p", { class: "fitline" }, chosen.words) : null,
      nav2("when", "physics", !!chosen),
    ];
  }

  // Vertical levels: the engine's default and a few named counts, each priced by the engine at the chosen grid,
  // and any other whole number, handed to the engine as it is.
  const otherLevels = h("input", { class: "input num", type: "number", min: "1", step: "1", inputmode: "numeric",
    "aria-label": w.levels_other });
  const otherNote = h("span", { class: "hint" }, w.levels_other_hint);
  otherLevels.addEventListener("change", () => {
    const n = Number(otherLevels.value);
    if (otherLevels.value === "" || !Number.isInteger(n) || n < 1) {
      otherNote.textContent = otherLevels.value === "" ? w.levels_other_hint : w.levels_other_bad;
      return;
    }
    otherNote.textContent = w.levels_other_hint;
    draft.nz = n;
    paintStep();
  });
  function levelChoices() {
    const named = w.levels.map((l) => l.nz);
    const options = w.levels.map((l) => {
      const entry = want(dxOf(), l.nz);
      const fit = entry.done && entry.done.ok ? entry.done.fit : null;
      const count = l.nz || fitLevels(fit);
      const on = draft.nz === l.nz;
      const el = h("button", { type: "button", class: `option${on ? " on" : ""}${sharedOff(entry.done) ? " off" : ""}`, "aria-pressed": on ? "true" : "false" },
        h("b", {}, l.name), h("span", { class: "v" }, count ? fill(w.levels_n, { n: count }) : "…"),
        h("span", { class: "why" }, l.why), fitLine(entry.done, w.too_big_levels));
      el.addEventListener("click", () => { draft.nz = l.nz; otherLevels.value = ""; paintStep(); });
      return el;
    });
    const custom = draft.nz !== null && !named.includes(draft.nz);
    if (custom && otherLevels.value === "") otherLevels.value = String(draft.nz);
    const customEntry = custom ? want(dxOf(), draft.nz) : null;
    const other = h("div", { class: `option${custom ? " on" : ""}` },
      h("b", {}, w.levels_other), h("span", { class: "v" }, custom ? fill(w.levels_n, { n: draft.nz }) : ""),
      h("div", { class: "why custom" }, otherLevels, otherNote), custom ? fitLine(customEntry.done, w.too_big_levels) : null);
    return h("div", { class: "options levels" }, options, other);
  }

  // ---- Physics: one table per family, each scheme a row with its cost; the engine checks the whole set
  function catalogFor(source) {
    if (!physics.catalogs.has(source)) {
      const entry = { done: null };
      physics.catalogs.set(source, entry);
      api.get(`/api/physics?source=${encodeURIComponent(source)}`)
        .then((doc) => { entry.done = doc; })
        .catch((err) => { entry.done = { error: errorText(err), families: [] }; })
        .then(() => { if (!closed) paintStep(); });
    }
    return physics.catalogs.get(source).done;
  }
  // The draft's nests go too, so the check reads the default at the finest grid, the set the run gets with none picked.
  const checkBody = () => {
    const out = { source: draft.source, cycle: draft.cycle, hours: draft.hours, nz: draft.nz,
      dx_km: draft.ladder ? null : dxOf(), ladder: draft.ladder || null, chain: eventLayout(dxOf(), draft.nz).chain,
      ...(autoGrid() || {}) };
    if (draft.box) { out.lat = draft.box.lat; out.lon = draft.box.lon; }
    if (picked()) out.choices = { ...draft.picks };
    else if (draft.profile) out.suite = draft.profile;
    return out;
  };
  // The data source's own set the checked picks make, or "" when they make none it offers.
  const setOfPicks = (result) => offeredSet(result, sources.find((row) => row.id === draft.source));
  // One check per distinct set, asked for once the picks stop changing. A set waits (physics.waiting) until the
  // picks hold still; only a check actually sent is kept (physics.checks), so a set whose wait was cut short by
  // another pick is asked about afresh when it is picked again.
  function physicsCheck() {
    if (!draft.source) return null;
    const body = checkBody();
    const key = JSON.stringify(body);
    if (physics.checks.has(key)) return physics.checks.get(key);
    if (physics.waiting && physics.waiting.key === key) return physics.waiting.entry;
    clearTimeout(physics.timer);
    const entry = { done: null };
    physics.waiting = { key, entry };
    physics.timer = setTimeout(() => {
      physics.waiting = null;
      physics.checks.set(key, entry);
      api.post("/api/physics/check", body)
        .then((reply) => { entry.done = reply.check; })
        .catch((err) => { entry.done = { valid: false, words: errorText(err), error: true }; })
        .then(() => {
          if (closed) return;
          // A set the source offers under the name the engine gave goes into the plan as that set; any other
          // mix goes in as the picks themselves.
          if (key === JSON.stringify(checkBody()) && picked()) {
            setProfile(setOfPicks(entry.done));
          }
          paintStep();
        });
    }, 250);
    return entry;
  }
  // A set by its plain name; its id is the hover title.
  function setName(id, label) {
    if (!label) {
      const catalog = draft.source ? catalogFor(draft.source) : null;
      const row = catalog && (catalog.suites || []).find((x) => x.id === id);
      label = row ? row.label : "";
    }
    return label && label !== id ? h("span", { title: id }, label) : h("span", { class: "mono" }, id);
  }
  // The picked schemes by their plain names, for a mix no set of the source matches.
  function mixWords() {
    const catalog = draft.source ? catalogFor(draft.source) : null;
    const names = Object.entries(draft.picks).map(([family, choice]) => {
      const rows = catalog && (catalog.families || []).find((f) => f.id === family);
      const row = rows && (rows.schemes || []).find((x) => x.choice === choice);
      return row ? row.label : choice;
    });
    return fill(w.physics_mix_plain, { schemes: names.join(", ") });
  }
  function physicsWords() {
    const entry = physicsCheck();
    const result = entry && entry.done;
    if (picked() && result && startable(result)) {
      return setOfPicks(result) ? setName(result.named_suite, result.named_suite_label) : mixWords();
    }
    if (draft.profile) return setName(draft.profile);
    return w.physics_default_plain;
  }
  const costText = (cost) => (!cost ? "" : cost.measured ? h("span", { class: "mono" }, `${Number(cost.relative).toFixed(2)}×`)
    : h("span", { class: "dim" }, w.physics_cost_unmeasured));
  // Picking a scheme repaints the step; the keyboard's place is put back on the scheme it picked.
  let refocus = null;
  function pickScheme(family, choice, toggle) {
    if (toggle && draft.picks[family] === choice) delete draft.picks[family];
    else draft.picks[family] = choice;
    if (!picked()) setProfile("");
    paintStep();
  }
  function familyTable(family, result) {
    const resolved = (result && result.resolved) || {};
    const running = resolved[family.id];
    const pick = draft.picks[family.id];
    const isRunning = (scheme) => running === scheme.choice || running === scheme.id;
    const rows = (family.schemes || []).map((scheme) => {
      const usable = scheme.implemented !== false && !scheme.blocker;
      const on = pick ? pick === scheme.choice : isRunning(scheme);
      const tags = [];
      // The default set for this grid, as the check names it (below 1 km it is not the source's own).
      const isDefault = result && result.default_suite ? (scheme.suites || []).includes(result.default_suite)
        : scheme.is_default;
      if (isDefault) tags.push(h("span", { class: "tag" }, w.physics_default_row));
      if (pick === scheme.choice) tags.push(h("span", { class: "tag a" }, w.physics_picked_row));
      else if (isRunning(scheme)) tags.push(h("span", { class: "tag a" }, w.physics_runs_row));
      // A native radio per scheme, one group per family: Tab reaches the family, the arrow keys pick in it.
      const radio = h("input", { type: "radio", class: "pick", name: `physics-${family.id}`, value: scheme.choice,
        "aria-label": scheme.label, checked: on, disabled: !usable });
      radio.addEventListener("change", () => {
        if (!radio.checked) return;
        refocus = { name: radio.name, value: radio.value };
        pickScheme(family.id, scheme.choice, false);
      });
      const tr = h("tr", { class: `row${on ? " on" : ""}${usable ? "" : " off"}` },
        h("td", {}, h("div", { class: "btns" }, radio, h("b", {}, scheme.label), tags),
          h("span", { class: "note" }, usable ? scheme.description : `${scheme.description} ${scheme.blocker || w.physics_unavailable}`)),
        h("td", { class: "num" }, costText(scheme.cost)));
      if (usable) {
        // A click elsewhere on the row picks it too, and a click on the picked row lets it go again.
        tr.addEventListener("click", (ev) => {
          if (ev.target === radio) return;
          pickScheme(family.id, scheme.choice, true);
        });
      }
      return tr;
    });
    const shown = (family.schemes || []).find((x) => (pick ? x.choice === pick : isRunning(x)));
    return fold(`${family.name}: ${shown ? shown.label : "…"}`,
      h("p", { class: "note" }, family.what),
      h("section", { class: "panel" }, h("div", { class: "tablewrap" }, h("table", { class: "dtable" },
        h("thead", {}, h("tr", {}, w.physics_cols.map((c, i) => h("th", { class: i ? "num" : null }, c)))),
        h("tbody", {}, rows)))));
  }
  function physicsStep() {
    const catalog = catalogFor(draft.source);
    const entry = physicsCheck();
    const result = entry && entry.done;
    const advisory = (a) => (typeof a === "string" ? a : a.words || a.message || a.headline || "");
    const verdict = !result ? h("p", { class: "fitline wait" }, w.physics_checking)
      : !result.valid ? h("p", { class: "fitline no" }, result.words)
        : h("div", { class: "levels" }, h("p", { class: "fitline" }, result.words),
          result.cost && result.cost.words ? h("p", { class: "note" }, fill(w.physics_cost_line, { words: result.cost.words })) : null,
          ...(result.advisories || []).map((a) => h("p", { class: "note" }, advisory(a))),
          picked() ? (setOfPicks(result) ? h("p", { class: "fitline" }, w.physics_lands, " ", setName(result.named_suite, result.named_suite_label))
            : h("p", { class: "fitline" }, w.physics_mix_lands)) : h("p", { class: "note" }, w.physics_default_set));
    const nextOn = !picked() || (result && startable(result));
    // The physics of the event's best run, while the plan runs it: the rest of the best run rides, the start data
    // is the event's own (another source runs its own default set) and no other set or scheme is picked.
    const own = recipe && recipe.layout && recipe.layout.physics && recipe.layout.physics.profile
      && bestRunRides() && !picked() && draft.source === recipe.source
      && (draft.profile || "") === (recipe.profile || "");
    return [
      h("h2", {}, w.physics_title), h("p", { class: "note" }, w.physics_line),
      own ? h("div", { class: "fromevent" }, h("p", { class: "note" }, fill(ww.physics_event,
        { profile: recipe.layout.physics.profile, why: recipe.layout.physics.why || "" }))) : null,
      verdict,
      picked() ? h("div", { class: "btns" }, button(w.physics_reset, { small: true, onclick: () => { draft.picks = {}; setProfile(""); paintStep(); } })) : null,
      // The Cost column is measured against the catalog's own reference, the data source's default set. Below 1 km the
      // grid's default is another set, so the rows tagged default are not the ones at 1.00x, and the step says so.
      catalog && !catalog.error && result && result.default_suite && catalog.default_suite
        && result.default_suite !== catalog.default_suite
        ? h("p", { class: "note" }, w.physics_cost_reference, " ", setName(catalog.default_suite), ".") : null,
      !catalog ? h("p", { class: "fitline wait" }, w.physics_checking)
        : catalog.error ? h("p", { class: "fitline no" }, catalog.error)
          : (catalog.families || []).map((family) => familyTable(family, result)),
      nav2("fine", "review", !!nextOn),
    ];
  }

  // What the assistant filled in, with its one-line reasons.
  let assisted = null;
  function assistantBanner() {
    if (!assisted) return null;
    const reasons = Object.entries(assisted.reasons || {});
    return h("div", { class: "fromevent" }, h("b", {}, w.from_assistant),
      reasons.length ? h("ul", { class: "plain" }, reasons.map(([k, v]) => h("li", { class: "note" }, h("b", {}, k), ` ${v}`))) : null,
      ...(assisted.fixes || []).map((fix) => h("p", { class: "fitline no" }, fix.words)));
  }
  function applyFill(detail) {
    if (!detail) return;
    const f = detail.fields || {};
    if (f.lat !== undefined && f.lon !== undefined && f.lat !== null && f.lon !== null) {
      draft.box = { lat: Number(f.lat), lon: Number(f.lon), width_km: Number(f.width_km || (draft.box && draft.box.width_km) || 600),
        height_km: Number(f.height_km || (draft.box && draft.box.height_km) || 600) };
      map.setBox(draft.box);
      map.frameBox();
      syncFields();
    }
    if (f.source) { draft.source = f.source; fillProfiles(); }
    if (f.hours) draft.hours = Number(f.hours);
    if (f.card) { draft.card = f.card; draft.cardTouched = true; }
    if (f.name) draft.name = String(f.name);
    if ("dx_km" in f) {
      const same = grids.findIndex((g) => g.dx_km === Number(f.dx_km));
      if (f.dx_km === null || f.dx_km === undefined) draft.dx = null;
      else if (same >= 0) { draft.grid = same; draft.dx = null; } else draft.dx = Number(f.dx_km);
    }
    if ("profile" in f) { draft.picks = {}; draft.profile = f.profile || ""; }
    if ("products" in f) {
      draft.products = f.products || "";
      draft.productsNamed = !["", "all", "none"].includes(draft.products);
    }
    if ("render_section" in f) draft.render_section = f.render_section || "";
    if ("ladder" in f) draft.ladder = f.ladder || "";
    if ("clock" in f) draft.clock = f.clock || "";
    if ("start_hour" in f) draft.startHour = Number(f.start_hour || 0);
    showDraft();
    assisted = { reasons: { ...((assisted && assisted.reasons) || {}), ...(detail.reasons || {}) }, fixes: detail.fixes || [] };
    if (f.cycle && ready) setCycle(f.cycle);
    else if (f.cycle) draft.cycle = f.cycle;
    if (draft.box && ready) setStep("review"); else paintStep();
  }

  // ---- where it runs and when it starts: this computer or a Machines node, now or after the queue. Asked every few
  // seconds while Review is open and drawn in place, so typing a name is never interrupted.
  const runsOn = h("td", {});
  const startsAt = h("td", {});
  const goBox = h("div", { class: "btns" });
  const goNote = h("p", { class: "note" });
  goNote.hidden = true;
  const machinePick = h("select", { class: "input", "aria-label": w.runs_on });
  machinePick.addEventListener("change", () => { draft.machine = machinePick.value; queueInfo = null; keepNow(); askQueue(); paintQueue(); });
  const machineField = h("label", { class: "wide" }, w.runs_on, machinePick);
  machineField.hidden = true;
  const needGib = () => { const f = currentFit(); return f && f.memory ? f.memory.need_gib : null; };
  function askQueue() {
    const need = needGib();
    const asked = draft.machine;
    // The start on the page rides along, so the answer also says whether Start takes it or only Queue it does.
    const start = draft.source && draft.cycle
      ? `&source=${encodeURIComponent(draft.source)}&cycle=${encodeURIComponent(draft.cycle)}&hours=${draft.hours}` : "";
    return api.get(`/api/queue?machine=${encodeURIComponent(asked)}${need ? `&need_gib=${need}` : ""}${start}`)
      .then((data) => { if (asked === draft.machine) queueInfo = data; })
      .catch(() => {})
      .then(() => { if (!closed) paintQueue(); });
  }
  function watchQueue() {
    if (queueTimer) return;
    const tick = () => {
      askQueue().then(() => {
        queueTimer = !closed && step === "review" ? setTimeout(tick, 3000) : null;
      });
    };
    queueTimer = setTimeout(tick, 0);
    if (machines === null) {
      machines = [];
      api.get("/api/machines").then((data) => {
        machines = (data.machines || []).filter((m) => m.name !== "this-computer");
        machinePick.replaceChildren(h("option", { value: "this-computer" }, w.here.replace(" ({host})", "")),
          ...machines.map((m) => h("option", { value: m.name }, m.name)));
        // A machine kept from before a reload that is no longer listed falls back to this computer.
        if (draft.machine !== "this-computer" && !machines.some((m) => m.name === draft.machine)) draft.machine = "this-computer";
        machinePick.value = draft.machine;
        machineField.hidden = machines.length === 0;
      }).catch(() => {});
    }
  }
  function stopQueueWatch() { clearTimeout(queueTimer); queueTimer = null; }
  function whereWords() {
    if (draft.machine !== "this-computer") return draft.machine;
    const e = queueInfo && queueInfo.expect;
    const host = queueInfo ? queueInfo.host : "";
    return e && e.card_name ? fill(w.here_card, { host, card: e.card_name }) : fill(w.here, { host });
  }
  // When the start's data is posted: the engine's readiness answer for this draft (gui/posting.py), asked once per
  // start, window, box and choice. On an engine that runs every start on the whole cycle it says so.
  const postings = new Map();
  const postingCell = h("td", {});
  // The opt-out is offered only where it changes something: an engine that starts every run on the whole cycle
  // has nothing to opt out of, and its Data posting row says so.
  const wholeCycleChoice = h("label", { class: "check" }, wholeCycle, " ", w.whole_cycle,
    h("span", { class: "hint" }, w.whole_cycle_hint));
  function showPosting(p) {
    postingCell.textContent = postingWords(p);
    wholeCycleChoice.hidden = !!(p && p.available === false && !draft.wholeCycle);
  }
  const postingKey = () => {
    const b = payload();
    return JSON.stringify([b.source, b.cycle, b.hours, b.start_hour, b.whole_cycle, b.lat, b.lon, b.width_km,
      b.height_km, b.dx_km, b.ladder, b.chain || null]);
  };
  const at = (text) => (text ? utcText(text) : w.posting_unknown_time);
  function postingWords(p) {
    if (!p) return w.posting_checking;
    if (p.failed) return fill(w.posting_failed, { why: p.failed });
    if (!p.available) return draft.wholeCycle ? w.posting_engine_whole : w.posting_engine_whole_default;
    if (p.error) return fill(w.posting_failed, { why: p.error });
    if (p.state === "refused") return p.refusal || w.posting_refused;
    const late = p.posting && p.posting.late_after_minutes;
    if (!p.as_posted) return fill(w.posting_whole, { final: at(p.expected_final_at) });
    if (p.posting && p.posting.streams === false) {
      return fill(w.posting_cycle, { final: at(p.expected_ready_at || p.expected_final_at), why: p.posting.why || "" });
    }
    if (p.state === "unprobeable") return fill(w.posting_unprobeable, { final: at(p.expected_final_at) });
    const lateWords = late ? fill(w.posting_late, { late }) : "";
    const first = p.ready ? w.posting_ready : fill(w.posting_waiting, { ready: at(p.expected_ready_at) });
    return [first, fill(w.posting_rest, { final: at(p.expected_final_at) }), lateWords].filter(Boolean).join(" ");
  }
  function askPosting() {
    const key = postingKey();
    if (!postings.has(key)) {
      postings.set(key, null);
      api.post("/api/create/posting", payload())
        .then((reply) => reply.posting)
        .catch((err) => ({ failed: err.message }))
        .then((answer) => {
          postings.set(key, answer);
          if (!closed && postingKey() === key) showPosting(answer);
        });
    }
    showPosting(postings.get(key));
  }

  // What takes the start on the page: "now" Start, "queue" only Queue it (the forecast waits for the start to be
  // confirmed), "no" neither.
  const startsBy = (e) => (e && e.data ? e.data.starts : "now");
  function whenWords() {
    const e = queueInfo && queueInfo.expect;
    if (!e || e.checking) return w.starts_checking;
    if (e.held) return e.held;
    if (startsBy(e) === "no") return e.data.why;
    const parts = [];
    if (startsBy(e) === "queue") parts.push(w.starts_for_data);
    else if (!e.busy && !e.ahead) return w.starts_now;
    if (e.running) {
      parts.push(e.seconds_left ? fill(w.starts_after_run, { run: e.running, left: duration(Math.round(e.seconds_left / 60) * 60) })
        : fill(w.starts_after_run_plain, { run: e.running }));
    } else if (e.busy) {
      parts.push(fill(w.starts_when_free, { why: e.why || "" }).trim());
    }
    if (e.ahead) parts.push(e.ahead === 1 ? w.starts_ahead_one : fill(w.starts_ahead, { n: e.ahead }));
    return parts.join(" ");
  }
  // The buttons: Start when the card is free and nothing waits; otherwise Queue it, and Start now beside it.
  let goState = { f: null, blocked: false };
  // What Start and Queue it send, and what their Show command asks for: one body, so the line shown is the one the
  // button runs, on the machine picked.
  // An auto ladder carries the grid its fit landed on, so Start checks the physics the run gets there.
  const startBody = (queue) => ({ ...payload(), queue, need_gib: needGib(),
    ...(draft.ladder === "auto" ? fittedGrid(currentFit()) || autoGrid() || {} : {}),
    ...(draft.machine !== "this-computer" ? { machine: draft.machine } : {}) });
  async function launch(queue) {
    for (const b of goBox.querySelectorAll("button")) b.disabled = true;
    try {
      const reply = await api.post("/api/create/start", startBody(queue));
      // Started or queued: the draft is spent, and New forecast opens fresh next time.
      forgetDraft();
      if (reply.queued) {
        goWithNotice("runs", fill(reply.waits_for_data ? w.queued_data : w.queued, { place: reply.place }));
      } else {
        const warning = reply.assistant && reply.assistant.warning;
        goWithNotice(runRoute("watch", reply.run), [w.started, warning].filter(Boolean).join(" "), warning ? "warn" : "");
      }
    } catch (err) {
      notice(errorText(err), "stop");
      goShown = "";
      paintQueue();
    }
  }
  // The buttons are drawn again only when what they offer changes: the queue is asked every few seconds, and a
  // button replaced on each answer lost a click that spanned it and closed its Show command.
  let goShown = "";
  function paintQueue() {
    runsOn.textContent = whereWords();
    startsAt.textContent = whenWords();
    const e = queueInfo && queueInfo.expect;
    const by = startsBy(e);
    // A start the queue would hold (no GPU runtime here, too little disk, too small a card) is off whatever the
    // card is doing: started, it fails. When it starts says why.
    const off = !goState.f || goState.blocked || by === "no" || !!(e && e.held);
    // Start now is off while the card is taken, or while the start is one only Queue it takes.
    const nowOk = !!e && by === "now" && e.start_now;
    const shape = JSON.stringify([!e || (!e.busy && !e.ahead), off, !!(e && e.held), nowOk, by, e && e.data ? e.data.why : ""]);
    if (shape === goShown && goBox.childElementCount) return;
    goShown = shape;
    // Drawn again (as on picking another machine, whose queue is asked afresh), a Show command left open stays open
    // on the button drawn in its place and asks for that button's line; the one replaced stops asking.
    const shown = [...goBox.children].map((el) => el.cmd).filter((cmd) => cmd && cmd.retire);
    const open = shown.some((cmd) => cmd.isOpen());
    for (const cmd of shown) cmd.retire();
    const request = (queue) => () => (draft.box ? { path: "/api/create/start", body: startBody(queue) }
      : { error: w.need_box });
    const watch = [name, lat, lon, width, height, dx, profile, products, customProducts, section, ladder, clock, startHour, machinePick];
    if (by !== "queue" && (!e || (!e.busy && !e.ahead))) {
      const start = actionButton(w.start, { kind: "primary", request: request(false), watch, open, disabled: off,
        onclick: () => launch(false) });
      start.button.classList.add("big");
      goBox.replaceChildren(start);
      goNote.textContent = by === "no" ? e.data.why : "";
      goNote.title = "";
      goNote.hidden = by !== "no";
      return;
    }
    const queueIt = actionButton(w.queue_it, { kind: "primary", disabled: off || !!e.held, onclick: () => launch(true),
      request: request(true), watch, open });
    queueIt.button.classList.add("big");
    const whyOff = by === "queue" ? fill(w.start_off_data, { why: e.data.why }) : w.start_now_off;
    const now = button(w.start_now, { disabled: off || !nowOk, onclick: () => launch(false), title: nowOk ? null : whyOff });
    goBox.replaceChildren(now, queueIt);
    // One short line in the button bar; the whole sentence is its title and Start now's.
    goNote.textContent = nowOk ? "" : by === "queue" ? w.start_off_data_short : w.start_now_off_short;
    goNote.title = nowOk ? "" : whyOff;
    goNote.hidden = nowOk;
  }

  function reviewStep() {
    const b = draft.box;
    const f = currentFit();
    const entry = want(dxOf());
    const g = grids[draft.grid];
    const mem = f && f.memory;
    // The forecast starts at the data's cycle plus the forecast hour it starts from; the cycle is said on its own row.
    const cycleAt = parseTime(draft.cycle);
    const startsFrom = cycleAt ? new Date(cycleAt.getTime() + draft.startHour * 3600000) : null;
    const layout = eventLayout(dxOf(), draft.nz);
    const gridWords = draft.ladder ? fill(w.ladder_row, { ladder: ladder.selectedOptions[0] ? ladder.selectedOptions[0].textContent : draft.ladder })
      : keepsLayout(layout) ? fill(ww.layout_grid, { spacings: layoutSpacings(f).join(", ") })
        : draft.dx !== null ? spacing(draft.dx) : `${g.name}, ${spacing(g.dx_km)}`;
    const written = outputWords();
    const rows = [
      [w.rows.where, `${b.lat.toFixed(2)}, ${b.lon.toFixed(2)}, ${b.width_km} × ${b.height_km} km`],
      [w.rows.start, `${utcText(startsFrom || draft.cycle)}, ${fill(w.hours_unit, { hours: draft.hours })}`],
      [w.rows.data, sourceName(draft.source)],
      [w.rows.cycle, fill(draft.startHour ? w.cycle_lead : w.cycle_only, { cycle: utcText(draft.cycle), hour: draft.startHour })],
      [w.rows.grid, gridWords],
      [w.rows.levels, draft.nz ? fill(w.levels_n, { n: draft.nz })
        : fitLevels(f) ? fill(w.levels_default, { n: fitLevels(f) }) : w.levels_default_plain],
      [w.rows.physics, physicsWords()],
      ...(written ? [[ww.output_row, written]] : []),
      [ww.pictures_row, picturesWords()],
      [w.rows.memory, mem ? `${cardWords(draft.card)}: ${fill(w.fits, { need: gb(mem.need_gib), budget: gb(mem.budget_gib) })}` : cardWords(draft.card)],
    ];
    const table = h("div", { class: "tablewrap" }, h("table", { class: "review" }, h("tbody", {},
      rows.map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", {}, v))),
      h("tr", {}, h("td", {}, w.rows.runs_on), runsOn), h("tr", {}, h("td", {}, w.rows.starts), startsAt),
      h("tr", {}, h("td", {}, w.rows.posting), postingCell))));
    askPosting();
    const more = fold(w.more,
      h("div", { class: "fields" },
        field(w.lat, lat), field(w.lon, lon), field(w.width, width), field(w.height, height),
        h("label", { class: "wide" }, w.dx, dx, h("span", { class: "hint" }, w.dx_hint)),
        h("label", { class: "wide" }, w.profile, profile), h("label", { class: "wide" }, w.products, products),
        customFields,
        h("label", {}, w.ladder, ladder, h("span", { class: "hint" }, w.ladder_hint)),
        h("label", {}, w.clock, clock, h("span", { class: "hint" }, w.clock_hint)),
        h("label", {}, w.start_hour, startHour, h("span", { class: "hint" }, w.start_hour_hint))));
    const fitButton = button(w.fit, { onclick: () => {
      fits.delete(fitKey(dxOf()));
      fitFor(dxOf());
      paintStep();
    } });
    const check = physicsCheck();
    const physicsBlocked = picked() && !(check && check.done && startable(check.done));
    goState = { f, blocked: physicsBlocked };
    paintQueue();
    watchQueue();
    // The name comes first, above the table: at 1366 by 900 the button bar, with its line about Start now, covered
    // the Name box when it sat below the table. The map's legend line goes under the table, so Runs on and Starts
    // stay in view too.
    return [
      h("h2", {}, w.review_title), recipeBanner(), assistantBanner(),
      h("div", { class: "fields sub" }, h("label", { class: "wide" }, w.name, name, h("span", { class: "hint" }, w.name_hint)),
        machineField),
      table,
      wholeCycleChoice,
      layoutLine(),
      entry.done ? (entry.done.ok ? h("p", { class: "fitline" }, f.words) : h("p", { class: "fitline no" }, fitLine(entry.done)))
        : h("p", { class: "fitline wait" }, w.fit_waiting),
      h("p", { class: "note" }, w.review_line),
      more,
      // The line that says why Start now is off sits with the buttons, above them, where the sticky bar keeps it.
      h("div", { class: "nav2" }, goNote, button(w.back, { onclick: () => setStep("physics") }), h("div", { class: "btns" }, fitButton, goBox)),
    ];
  }

  // The last fit the map was framed on (paintNow).
  let framedFit = null;
  let openFolds = [];
  // A paint asked for while one is drawing runs right after it rather than inside it.
  let painting = false;
  function paintStep() {
    if (closed) return;
    if (painting) { queueMicrotask(paintStep); return; }
    painting = true;
    try { paintNow(); } finally { painting = false; }
  }
  let paintedStep = null;
  function paintNow() {
    // Folds keep their open state across repaints of one step; another step starts with its folds closed.
    openFolds = paintedStep === step ? [...panel.querySelectorAll("details.fold:not([data-fold])")].map((d) => d.open) : [];
    const entering = paintedStep !== step;
    paintedStep = step;
    if (step !== "where" && !draft.box) step = "where";
    paintSteps();
    const titles = { when: w.when_title, fine: w.fine_title, physics: w.physics_title, review: w.review_title };
    const kids = step === "where" ? whereStep()
      : step === "when" ? whenStep()
        : !ready ? waitStep(titles[step], STEPS[STEPS.indexOf(step) - 1])
          : step === "fine" ? fineStep() : step === "physics" ? physicsStep() : reviewStep();
    if (step !== "review") stopQueueWatch();
    // The parts a step keeps (the date box, the hour, the calendar, the source rows) stay where they are in the
    // document, so the keyboard stays where it was; should one still have to move, it gets the keyboard and its
    // selection back.
    const had = panel.contains(document.activeElement) ? document.activeElement : null;
    const selection = had && typeof had.selectionStart === "number" ? [had.selectionStart, had.selectionEnd] : null;
    place(panel, kids);
    if (had && had.isConnected && panel.contains(had) && document.activeElement !== had) {
      had.focus({ preventScroll: true });
      if (selection) { try { had.setSelectionRange(selection[0], selection[1]); } catch (err) { /* not a text box */ } }
    }
    panel.querySelectorAll("details.fold:not([data-fold])").forEach((d, i) => { if (openFolds[i]) d.open = true; });
    // A physics scheme picked by keyboard is drawn afresh with its family; the keyboard goes back to it.
    if (refocus) {
      const again = [...panel.querySelectorAll("input[type=radio]")].find((r) => r.name === refocus.name && r.value === refocus.value);
      refocus = null;
      if (again) again.focus({ preventScroll: true });
    }
    // The calendar draws itself when its day changes and shades its days in place as answers come; it is drawn
    // afresh only on the way into When, so a redraw never closes its month or year box.
    if (step === "when" && entering) cal.paint();
    const shownFit = step === "fine" || step === "physics" || step === "review" ? currentFit() : null;
    map.setFit(shownFit);
    // A fit of several grids is framed whole when it comes in, so the map shows the outer grids around the box
    // and not the box alone.
    if (shownFit && shownFit !== framedFit && shownFit.domains && shownFit.domains.length > 1) {
      framedFit = shownFit;
      map.frameBox();
    }
  }

  if (draft.box) { map.setBox(draft.box); map.frameBox(); }
  syncFields();
  // When draws at once on a start from this browser's clock (or the link's); the server's answer replaces it
  // below unless the person has already picked one.
  if (!draft.cycle) { draft.cycle = localCycle(); openedOn = draft.cycle; } else cycleTouched = true;
  showDraft();
  setCycle(draft.cycle, { user: false });

  api.get("/api/system").then((sys) => {
    if (closed || !sys) return;
    device = sys.devices && sys.devices.length ? sys.devices[0] : null;
    page.system = sys;
    if (!draft.cardTouched && !(recipe && recipe.card) && sys.card) draft.card = sys.card;
    paintStep();
  }).catch(() => {});

  api.get("/api/sources").then((data) => {
    if (closed) return;
    sources = data.sources;
    cards = data.cards;
    const offered = (id) => sources.some((row) => row.id === id);
    if (!offered(draft.source)) draft.source = (data.preferred || []).find(offered) || (sources[0] || {}).id || "";
    // The event's own start and source when New forecast offers that source; the link's own when it holds one;
    // otherwise the newest start, unless the person already picked a date.
    if (restored) {
      // A kept draft keeps its own source and start.
    } else if (recipe && recipe.runnable && offered(recipe.start_source) && !route.cycle) {
      draft.source = recipe.start_source;
      if (!cycleTouched) draft.cycle = recipe.start_cycle;
    } else if (!cycleTouched) {
      // The opening start by this browser's clock (openingOf): until a row's check answers, the newest start past
      // the source's usual publication delay. The rows then move it as their checks answer.
      draft.cycle = openingOf(factsOf(draft.source)) || draft.cycle;
      openedOn = draft.cycle;
    }
    if (recipe && recipe.card && !draft.cardTouched) draft.card = recipe.card;
    fillProfiles();
    fillLadders(data.ladders);
    showDraft();
    ready = true;
    setCycle(draft.cycle, { user: false, viewMonth: !cycleTouched });
    // Rows that came back before the source list are read again for the source it named.
    const early = avail.get(availKey());
    if (early && early.done && !early.flying) askAvail(true);
    // A plan the assistant made before this page was open.
    applyFill(takeFill());
  }).catch((err) => {
    if (closed) return;
    loadError = errorText(err);
    paintStep();
  });
  const unFill = onFill((detail) => { takeFill(); applyFill(detail); });
  const unRead = setFormReader(() => payload());
  return () => {
    closed = true; clearTimeout(fitTimer); clearTimeout(physics.timer); stopQueueWatch();
    for (const entry of avail.values()) clearTimeout(entry.timer);
    finder.close(); map.close(); unFill(); unRead();
  };
}

register("create", render, { title: (words) => words.screens.create.title });
