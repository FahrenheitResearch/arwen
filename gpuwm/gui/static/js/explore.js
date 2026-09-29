// Files: a forecast's folder as the engine and the renderer left it. The render tree <domain>/<product>/<valid-day>,
// one folder at a time, named and grouped the way the map names them (the renderer's own folder name shows on
// hover), each picture opening as the file the renderer wrote; the run folder and the commands that made it,
// folded away. Without a forecast named, the list of forecasts to pick from.

import { h, fill, fold, copyline, chip, button } from "./core.js";
import * as api from "./api.js";
import { register, go, runRoute, runContext, runCrumbs, errorText } from "./router.js";
import { productName, kindOf } from "./looks.js";

async function pickRun(body, page) {
  const w = page.words.screens;
  page.setTitle(w.explore.title);
  page.setData(w.explore.intro);
  page.setCrumbs([[w.shell.my_forecasts, "#/runs"], [w.shell.files]]);
  const data = await api.get("/api/runs");
  // Each forecast is a link, so the keyboard reaches and opens it; a click anywhere on its row opens it too.
  const rows = data.runs.map((row) => {
    const tr = h("tr", { class: "row" }, h("td", {}, h("a", { class: "rowlink", href: `#/${runRoute("explore", row.id)}` }, row.title || row.id)),
      h("td", {}, chip(row.status.state, w.states[row.status.state])),
      h("td", { class: "num mono" }, row.card && row.card.pictures ? row.card.pictures.toLocaleString("en-US") : ""));
    tr.addEventListener("click", (ev) => { if (!ev.target.closest("a")) go(runRoute("explore", row.id)); });
    return tr;
  });
  body.append(h("section", { class: "panel" }, rows.length
    ? h("div", { class: "tablewrap" }, h("table", { class: "dtable" },
      h("thead", {}, h("tr", {}, w.explore.pick_columns.map((c, i) => h("th", { class: i === 2 ? "num" : null }, c)))), h("tbody", {}, rows)))
    : h("p", { class: "pbody note" }, w.runs.empty)));
}

async function render(body, args, page) {
  const w = page.words.screens;
  const runId = args[0];
  if (!runId) { await pickRun(body, page); return; }
  const [detail, index] = await Promise.all([api.get(api.runPath(runId)), api.get(`${api.runPath(runId)}/pictures`)]);
  const title = detail.title || detail.name || runId;
  page.setTitle(title, { name: true });
  page.setData(w.explore.intro);
  page.setCrumbs(runCrumbs(page.words, runId, title, w.shell.files));
  page.setContext(runContext(page.words, runId, title, "explore"));

  const folder = h("section", { class: "panel pad" },
    h("h2", { class: "sec" }, w.results.files),
    fold(w.results.show_folder, copyline(detail.folder, w.results.folder)),
    Object.keys(detail.reattach || {}).length ? fold(w.results.reattach, ...Object.values(detail.reattach).map((path) => copyline(path))) : null);
  const log = detail.commands_log ? h("section", { class: "panel" }, h("div", { class: "phd" }, h("b", {}, w.results.log)),
    h("pre", { class: "log" }, detail.commands_log)) : null;

  if (!index.count) {
    body.append(h("section", { class: "panel pad" }, h("p", { class: "note" }, w.explore.none)), h("div", { class: "wsec" }, folder, log));
    return;
  }
  const looks = page.words.looks;
  const kindOrder = [...(looks.kinds || []).map((k) => k.name), looks.other_kind, looks.raw_kind];
  const shown = h("div", {});
  // Each folder's list belongs to the pick that asked for it: a reply that lands after another folder was picked
  // (or the page was left) is dropped, so the pictures shown are always the picked row's. A list that fails says
  // why in place, with Try again, instead of leaving the last folder's pictures under the new pick.
  let request = 0;
  let closed = false;
  async function open(group, tr) {
    const mine = ++request;
    for (const row of tbody.children) {
      row.classList.toggle("on", row === tr);
      const b = row.querySelector("button.rowlink");
      if (b) b.setAttribute("aria-pressed", row === tr ? "true" : "false");
    }
    const q = new URLSearchParams({ domain: group.domain, episode: group.episode || "", product: group.product, day: group.day });
    shown.replaceChildren(h("p", { class: "note", role: "status" }, w.explore.loading));
    let data;
    try {
      data = await api.get(`${api.runPath(runId)}/pictures/list?${q}`);
    } catch (err) {
      if (closed || mine !== request) return;
      shown.replaceChildren(h("p", { class: "notice stop", role: "status" }, errorText(err)),
        button(w.explore.retry, { onclick: () => open(group, tr) }));
      return;
    }
    if (closed || mine !== request) return;
    shown.replaceChildren(h("section", { class: "panel wsec" },
      h("div", { class: "phd" }, h("b", {}, productName(group.product, looks)), h("span", { class: "mono" }, `${gridName(group)} · ${group.day} · ${data.count}`)),
      h("div", { class: "thumbs" }, data.pictures.map((p) => h("a", { href: api.filePath(runId, p.path), target: "_blank", rel: "noopener" },
        h("img", { src: api.filePath(runId, p.path), alt: p.name, loading: "lazy" }), p.label)))));
    shown.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  // A nest that retired and re-armed has one row per life: "d02-3km, life 2".
  const gridName = (g) => (g.episode ? fill(w.explore.life, { domain: g.domain, number: Number(g.episode.split("-").pop()) }) : g.domain);
  const named = index.groups.map((g) => ({ g, kind: kindOf(g.product, looks), name: productName(g.product, looks) }));
  named.sort((a, b) => kindOrder.indexOf(a.kind) - kindOrder.indexOf(b.kind) || a.name.localeCompare(b.name)
    || a.g.domain.localeCompare(b.g.domain) || (a.g.episode || "").localeCompare(b.g.episode || "") || a.g.day.localeCompare(b.g.day));
  const tbody = h("tbody", {});
  for (const { g, kind, name } of named) {
    // The folder's name is a button, so the keyboard reaches and opens it; a click anywhere on its row opens it too.
    const pick = h("button", { type: "button", class: "rowlink", "aria-pressed": "false",
      "aria-label": `${name}, ${gridName(g)}, ${g.day}` }, name);
    const tr = h("tr", { class: "row" }, h("td", { class: "dim" }, kind), h("td", { title: g.product }, pick), h("td", { class: "mono" }, gridName(g)),
      h("td", { class: "mono" }, g.day), h("td", { class: "num mono" }, String(g.count)));
    tr.addEventListener("click", () => open(g, tr));
    tbody.append(tr);
  }
  body.append(
    h("h2", { class: "sec" }, w.explore.tree, h("span", {}, fill(w.explore.count, { count: index.count.toLocaleString("en-US") }))),
    h("section", { class: "panel" }, h("div", { class: "tablewrap" }, h("table", { class: "dtable" },
      h("thead", {}, h("tr", {}, w.explore.columns.map((c, i) => h("th", { class: i === 4 ? "num" : null }, c)))), tbody))),
    shown, h("div", { class: "wsec" }, folder, log));
  return () => { closed = true; request += 1; };
}

register("explore", render, { title: (words) => words.screens.explore.title });
