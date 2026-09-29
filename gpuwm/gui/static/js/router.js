// The screen table, the hash router's helpers and the notice line. Screens import this (never app.js), so there
// is no import cycle: app.js imports the screens, the screens register themselves here.

export const SCREENS = new Map();
// The pages of one open forecast, listed under its name in the sidebar: its article, its map, its files.
export const RUN_PAGES = ["run", "results", "explore"];

// Routes that moved keep answering: an address whose first part is an old name forwards to the new
// name, the rest of the address kept. The main page's route took the page's name, the Weather Library.
export const MOVED_ROUTES = new Map([["wiki", "library"]]);

// The route an old address forwards to (wiki/x to library/x), or null when the address has not moved.
export function forwardedRoute(raw) {
  const [name, ...rest] = String(raw || "").split("/");
  return MOVED_ROUTES.has(name) ? [MOVED_ROUTES.get(name), ...rest].join("/") : null;
}

// register(name, render, {title, side, crumb, bleed}). render(body, args, page) draws into body and may return a
// cleanup function. page.setTitle(text) and page.setData(text) fill the page head; page.right holds its actions;
// page.setCrumbs([[label, href], ...]) sets the path in the header; page.setContext({title, items}) adds a group of
// links to the sidebar (the events of this kind, the pages of this forecast). side names the sidebar link the
// screen lights up; bleed gives the screen the whole content area with no padding (the map).
export function register(name, render, opts = {}) {
  SCREENS.set(name, { render, ...opts });
}

export function go(route) {
  const target = `#/${route}`;
  if (location.hash === target) window.dispatchEvent(new HashChangeEvent("hashchange"));
  else location.hash = target;
}

// Go to a route and show a notice on the page it opens. Drawing a route clears the notice line, so a notice set
// just before go() (a start's "Started", or the warning that the assistant's model is still on the card) was
// cleared before anyone saw it; this one is set once the new page has been drawn.
export function goWithNotice(route, text, kind = "") {
  window.addEventListener("hashchange", () => notice(text, kind), { once: true });
  go(route);
}

export function runRoute(page, id) {
  return `${page}/${encodeURIComponent(id)}`;
}

// Where a forecast opens from a list: its map while it runs or once it has pictures, its article otherwise.
export function openRoute(row) {
  const state = row.status.state;
  if (state === "running" || state === "ready") return runRoute("watch", row.id);
  if (row.card && row.card.pictures) return runRoute("results", row.id);
  return runRoute("run", row.id);
}

let noticeBox = null;

export function setNoticeBox(el) {
  noticeBox = el;
}

// One line under the header: what just happened, and what to do (kind: "", "warn" or "stop").
export function notice(text, kind = "") {
  if (!noticeBox) return;
  noticeBox.replaceChildren();
  if (!text) return;
  const p = document.createElement("p");
  p.className = `notice${kind ? ` ${kind}` : ""}`;
  p.textContent = text;
  noticeBox.append(p);
}

// Take the notice line down while it still says text. A line set since by something else stays: a page back in
// touch with the server takes its own "lost touch" line down, never the line a start or a Remove left.
export function clearNotice(text) {
  if (!noticeBox) return;
  const line = noticeBox.querySelector(".notice");
  if (line && line.textContent === text) noticeBox.replaceChildren();
}

export function errorText(err) {
  return `${(err && err.message) || err || ""} ${(err && err.fix) || ""}`.trim();
}

// The sidebar group of one open forecast: its article, its map, its files.
export function runContext(words, runId, title, active) {
  const sh = words.screens.shell;
  const labels = { run: sh.article, results: sh.map, explore: sh.files };
  return {
    title: title || runId,
    items: RUN_PAGES.map((p) => ({ href: `#/${runRoute(p, runId)}`, label: labels[p], on: p === active || (active === "watch" && p === "results") })),
  };
}

// The header's path for a forecast's pages.
export function runCrumbs(words, runId, title, pageLabel) {
  return [[words.screens.shell.my_forecasts, "#/runs"], [title || runId, `#/${runRoute("run", runId)}`], pageLabel ? [pageLabel] : null];
}
