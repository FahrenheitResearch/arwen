// Search, opened with Ctrl K or the header's search box: the wiki's pages and your forecasts as you type, the
// app's own pages, arrows to move, Enter to open, Escape to close. The last row opens the full search page for
// the typed words.

import { h } from "./core.js";
import * as api from "./api.js";
import { hrefFor, searchHref, day } from "./wikikit.js";
import { openAssistant, assistantEnabled } from "./assistant.js";

let open = null;

export function openPalette(words) {
  if (open) { open.input.focus(); return; }
  const sh = words.screens.shell;
  const ws = words.wiki.search;
  const input = h("input", { type: "text", placeholder: sh.palette_placeholder, "aria-label": sh.search, spellcheck: "false",
    autocomplete: "off" });
  const icon = h("span", { "aria-hidden": "true" });
  icon.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4">' +
    '<circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5 14 14"/></svg>';
  const pin = h("div", { class: "pin" }, icon, input, h("kbd", {}, "Esc"));
  const list = h("div", { class: "plist", role: "listbox" });
  const foot = h("div", { class: "pfoot" }, h("span", {}, sh.palette_keys), h("span", {}, sh.palette_enter));
  const pal = h("div", { class: "pal", role: "dialog", "aria-label": sh.search }, pin, list, foot);
  const back = h("div", { class: "pal-back" }, pal);
  document.body.append(back);
  input.focus();

  const pages = [
    { label: words.wiki.nav.wiki, href: "#/library" },
    { label: words.wiki.nav.browse, href: "#/browse" },
    { label: words.wiki.nav.places, href: "#/places" },
    { label: words.wiki.nav.changes, href: "#/changes" },
    { label: sh.my_forecasts, href: "#/runs" },
    { label: sh.new_forecast, href: "#/create" },
    { label: sh.files, href: "#/explore" },
    { label: sh.machines, href: "#/machines" },
    { label: sh.settings, href: "#/settings" },
  ];
  // Things the palette does rather than opens: the assistant is a panel beside the page.
  const actions = [
    assistantEnabled()
      ? { label: sh.open_assistant, sub: sh.open_assistant_sub, keywords: `${sh.assistant} ask help`, run: () => openAssistant(words) }
      : { label: sh.assistant_off, sub: sh.assistant_off_line, keywords: `${sh.assistant} ask help turn on`, run: () => openAssistant(words) },
  ];
  let runs = [];
  api.get("/api/runs").then((data) => { runs = data.runs || []; paint(); }).catch(() => {});

  let items = [];
  let at = 0;
  let found = [];
  let asked = 0;
  let timer = null;

  function close() {
    // an answer still on its way has no palette to be drawn in
    asked += 1;
    clearTimeout(timer);
    back.remove();
    document.removeEventListener("keydown", onKey, true);
    open = null;
  }
  function choose(item) {
    close();
    if (item.run) item.run();
    else if (item.href) location.hash = item.href.slice(1);
  }
  function row(item, group) {
    const b = h("button", { type: "button", class: "pitem", role: "option" }, h("b", {}, item.label), item.sub ? h("span", {}, item.sub) : null);
    b.addEventListener("click", () => choose(item));
    b.addEventListener("mousemove", () => { if (items[at] !== item) { at = items.indexOf(item); mark(); } });
    item.el = b;
    item.group = group;
    return b;
  }
  function mark() {
    items.forEach((item, i) => item.el.classList.toggle("on", i === at));
    const el = items[at] && items[at].el;
    if (el) el.scrollIntoView({ block: "nearest" });
  }
  function paint() {
    const q = input.value.trim().toLowerCase();
    const groups = [];
    const pagesHit = pages.filter((p) => !q || p.label.toLowerCase().includes(q));
    const runsHit = runs.filter((r) => !q || String(r.title || r.id).toLowerCase().includes(q) || r.id.toLowerCase().includes(q))
      .slice(0, q ? 6 : 4).map((r) => ({ label: r.title || r.id, href: `#/run/${encodeURIComponent(r.id)}`,
        sub: [words.screens.states[r.status.state] || r.status.state, r.status.start_time].filter(Boolean).join(" · ") }));
    const wikiHit = found.filter((r) => r.kind !== "run").slice(0, 8).map((r) => ({ label: r.title, href: hrefFor(r),
      sub: r.kind === "event" ? [r.type_title, r.region, day(r.start)].filter(Boolean).join(" · ") : ws.badge[r.kind] || r.kind }));
    if (q && wikiHit.length) groups.push([sh.palette_wiki, wikiHit]);
    if (runsHit.length) groups.push([sh.palette_runs, runsHit]);
    const actionsHit = actions.filter((a) => !q || `${a.label} ${a.keywords}`.toLowerCase().includes(q));
    if (actionsHit.length) groups.push([sh.palette_actions, actionsHit]);
    if (pagesHit.length) groups.push([sh.palette_pages, pagesHit]);
    if (q) groups.push([sh.palette_more, [{ label: `${ws.button}: “${input.value.trim()}”`, href: searchHref({ q: input.value.trim() }) }]]);
    items = [];
    const kids = [];
    for (const [name, rows] of groups) {
      kids.push(h("div", { class: "pgrp" }, name));
      for (const item of rows) { kids.push(row(item, name)); items.push(item); }
    }
    if (!items.length) kids.push(h("p", { class: "pempty" }, ws.none));
    list.replaceChildren(...kids);
    at = Math.min(at, Math.max(0, items.length - 1));
    mark();
  }
  async function ask() {
    const q = input.value.trim();
    if (q.length < 2) { found = []; paint(); return; }
    const mine = ++asked;
    try {
      const data = await api.get(`/api/library/search?q=${encodeURIComponent(q)}&sort=score`);
      if (mine !== asked) return;
      found = data.results || [];
    } catch (_) {
      if (mine !== asked) return;
      found = [];
    }
    paint();
  }
  input.addEventListener("input", () => {
    // The answers so far, and any still on the way, are for the words before this change: shown under the new
    // ones, Enter opened them.
    asked += 1;
    found = [];
    at = 0;
    paint();
    clearTimeout(timer);
    timer = setTimeout(ask, 160);
  });
  function onKey(ev) {
    if (ev.key === "Escape") { ev.preventDefault(); close(); return; }
    if (ev.key === "ArrowDown") { ev.preventDefault(); at = Math.min(items.length - 1, at + 1); mark(); }
    else if (ev.key === "ArrowUp") { ev.preventDefault(); at = Math.max(0, at - 1); mark(); }
    else if (ev.key === "Enter") { ev.preventDefault(); if (items[at]) choose(items[at]); }
  }
  document.addEventListener("keydown", onKey, true);
  back.addEventListener("pointerdown", (ev) => { if (ev.target === back) close(); });
  window.addEventListener("hashchange", () => { if (open) close(); }, { once: true });
  open = { input, close };
  paint();
}
