// The page shell: the header (the mark, the path to this page, search, New forecast), the sidebar (the wiki, the
// forecasts, and whatever the open page adds: the events of this kind, the pages of this forecast) and a hash
// router. Each screen registers itself in router.js and draws into the content area.

import { h } from "./core.js";
import * as api from "./api.js";
import { SCREENS, setNoticeBox, notice, errorText } from "./router.js";
import { openPalette } from "./palette.js";
import "./runs.js";
import "./create.js";
import "./viewpage.js";
import "./explore.js";
import "./wikipages.js";
import "./machines.js";
import "./settings.js";
import { attachAssistant, openAssistant, assistantOpen, showPanel, assistantEnabled, noteEnabled } from "./assistant.js";

const SEARCH_ICON = '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" ' +
  'aria-hidden="true"><circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5 14 14"/></svg>';

let cleanup = null;
let routing = 0;
// The open page's sidebar painter: the assistant link lights up when its panel opens or closes.
let repaintSide = null;
document.addEventListener("assistant", () => { if (repaintSide) repaintSide(); });

// A route is #/NAME/ARG/ARG?QUERY. The query (the search route's filters) is cut off before anything is decoded
// and handed on as it is, so a "?" or "&" typed into a search stays part of the words: decoded first, the words'
// "?" read as the start of the filters and everything after it was lost. A segment that is not valid percent
// encoding (a hand-typed "%") stays as typed rather than stopping the page.
function decodePart(text) {
  try { return decodeURIComponent(text); } catch (_) { return text; }
}

function parseHash() {
  const raw = (location.hash || "").replace(/^#\/?/, "");
  const at = raw.indexOf("?");
  const [name, ...rest] = (at < 0 ? raw : raw.slice(0, at)).split("/");
  return { name: name || "wiki", args: rest.map(decodePart), query: at < 0 ? "" : raw.slice(at + 1), raw };
}

// What the sidebar counts: the wiki's kinds and events, and the forecasts on disk. Asked for once, and the
// forecast count again on every page change.
const counts = { wiki: null, runs: null };
async function readCounts() {
  const [wiki, runs] = await Promise.all([
    counts.wiki ? Promise.resolve(counts.wiki) : api.get("/api/wiki").catch(() => null),
    api.get("/api/runs").catch(() => null),
  ]);
  counts.wiki = wiki;
  counts.runs = runs;
  return counts;
}

function sideLink(href, label, opts = {}) {
  return h("a", { href, class: opts.on ? "on" : null, title: opts.title || null },
    opts.dot ? h("i", { class: opts.dot === true ? null : opts.dot }) : null,
    h("span", { class: "t" }, label),
    opts.n !== undefined && opts.n !== null ? h("span", { class: "n" }, String(opts.n)) : null);
}

function drawSide(side, words, s, here, context) {
  const ww = words.wiki.nav;
  const sh = words.screens.shell;
  const w = counts.wiki;
  const rows = counts.runs ? counts.runs.runs : [];
  const on = (href) => here === href;
  const kids = [h("h4", {}, ww.wiki_group),
    sideLink("#/wiki", ww.wiki, { on: on("#/wiki") }),
    sideLink("#/browse", ww.browse, { on: on("#/browse"), n: w ? w.counts.events : null })];
  for (const k of (w && w.phenomena) || []) {
    kids.push(sideLink(`#/kind/${encodeURIComponent(k.id)}`, k.plural || k.title, { on: on(`#/kind/${k.id}`), n: k.count }));
  }
  kids.push(sideLink("#/places", ww.places, { on: on("#/places") }), sideLink("#/changes", ww.changes, { on: on("#/changes") }));
  kids.push(h("h4", {}, ww.forecasts_group),
    sideLink("#/runs", sh.my_forecasts, { on: on("#/runs"), n: (counts.runs && counts.runs.total) || rows.length || null }),
    sideLink("#/create", sh.new_forecast, { on: on("#/create") }),
    sideLink("#/explore", sh.files, { on: on("#/explore") }),
    sideLink("#/machines", sh.machines, { on: on("#/machines") }));
  // The assistant is a panel beside every page, not a page: its link opens and closes it. It is optional and off
  // until turned on; then the link says so, with one line on how to turn it on.
  const asking = assistantEnabled();
  const ask = sideLink("#", asking ? sh.assistant : sh.assistant_off, { on: assistantOpen(),
    title: asking ? sh.open_assistant_sub : sh.assistant_off_line });
  ask.setAttribute("role", "button");
  ask.addEventListener("click", (ev) => { ev.preventDefault(); if (assistantOpen()) showPanel(false); else openAssistant(words); });
  kids.push(h("h4", {}, sh.ask_group), ask);
  if (!asking) kids.push(h("p", { class: "side-line" }, sh.assistant_off_line));
  kids.push(sideLink("#/settings", sh.settings, { on: on("#/settings") }));
  if (context && context.items && context.items.length) {
    kids.push(h("h4", { title: context.title }, context.title));
    for (const item of context.items) {
      kids.push(sideLink(item.href, item.label, { on: item.on, dot: item.dot === undefined ? true : item.dot, title: item.title }));
    }
  }
  kids.push(h("div", { class: "foot" }, h("div", { class: "mono" }, `${s.bind}:${s.port}`), h("div", {}, `gpuwm ${s.version}`)));
  side.replaceChildren(...kids);
}

function frame(words, s, entry, route) {
  const sh = words.screens.shell;
  const crumb = h("nav", { class: "crumb", "aria-label": sh.path });
  const search = h("button", { type: "button", class: "hsearch", "aria-label": sh.search, title: sh.search });
  search.innerHTML = SEARCH_ICON;
  search.append(h("span", {}, sh.search), h("kbd", {}, sh.search_key));
  search.addEventListener("click", () => openPalette(words));
  const logo = h("a", { class: "logo", href: "#/wiki" }, h("b", {}), words.screens.app.name);
  const header = h("header", { class: "hdr" }, logo, crumb, search,
    h("a", { class: "btn", href: "#/create" }, sh.new_forecast));
  const side = h("nav", { class: "side", "aria-label": sh.sections });
  const title = h("h1", {});
  const data = h("div", { class: "data" });
  const right = h("div", { class: "right" });
  const head = h("div", { class: "phead" }, h("div", {}, title, data), right);
  const noticeBox = h("div", { class: "noticebox", role: "status", "aria-live": "polite" });
  setNoticeBox(noticeBox);
  const body = h("div", { class: "pbody-root" });
  const content = h("main", { class: `content${entry.bleed ? " bleed" : ""}` }, noticeBox, entry.bleed ? null : head, body);
  document.body.replaceChildren(header, h("div", { class: "shell" }, side, content));
  attachAssistant(words);
  const here = `#/${route}`;
  let context = null;
  const paintSide = () => drawSide(side, words, s, entry.side ? `#/${entry.side}` : here, context);
  paintSide();
  readCounts().then(paintSide);
  repaintSide = paintSide;
  const setTitle = (text, opts = {}) => {
    title.textContent = text || "";
    title.classList.toggle("name", !!opts.name);
    document.title = text ? `${text} · ${words.screens.app.name}` : words.screens.app.name;
  };
  const setCrumbs = (items) => {
    const parts = [];
    for (const [label, href] of items.filter(Boolean)) {
      parts.push(h("em", {}, "/"), href ? h("a", { href }, label) : h("span", {}, label));
    }
    crumb.replaceChildren(...parts);
  };
  setCrumbs([[words.wiki.name, "#/wiki"]]);
  return {
    body, right, setTitle, setCrumbs,
    setData: (text) => { data.textContent = text || ""; },
    setContext: (ctx) => { context = ctx; paintSide(); },
    refreshSide: () => readCounts().then(paintSide),
  };
}

async function shell() {
  let words;
  let s;
  try {
    [words, s] = await Promise.all([api.copy(), api.session()]);
  } catch (err) {
    document.body.replaceChildren(h("main", { class: "content" }, h("p", { class: "notice stop" }, errorText(err))));
    return;
  }
  noteEnabled(!!(s.assistant && s.assistant.enabled));

  async function route() {
    const ticket = ++routing;
    if (cleanup) { try { cleanup(); } catch (_) { /* a screen's cleanup never blocks the next */ } }
    cleanup = null;
    const { name, args, query, raw } = parseHash();
    const screen = SCREENS.has(name) ? name : "wiki";
    const entry = SCREENS.get(screen);
    const f = frame(words, s, entry, raw || "wiki");
    f.setTitle(entry.title ? entry.title(words) : "");
    document.body.dataset.screen = screen;
    delete document.body.dataset.ready;
    notice("");
    const page = { words, session: s, query, ...f };
    try {
      const done = await entry.render(f.body, args, page);
      if (ticket !== routing) { if (typeof done === "function") done(); return; }
      if (typeof done === "function") cleanup = done;
      document.body.dataset.ready = "1";
    } catch (err) {
      notice(`${words.screens.app.failed_draw} ${errorText(err)}`, "stop");
      document.body.dataset.ready = "error";
    }
  }
  window.addEventListener("hashchange", route);
  // Ctrl K (or / outside a text box) opens search over the wiki and your forecasts.
  window.addEventListener("keydown", (ev) => {
    const typing = ev.target.closest && ev.target.closest("input, select, textarea, [contenteditable]");
    if ((ev.key === "k" || ev.key === "K") && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); openPalette(words); }
    else if (ev.key === "/" && !typing && !ev.ctrlKey && !ev.metaKey && !ev.altKey) { ev.preventDefault(); openPalette(words); }
  });
  route();
}

shell();
