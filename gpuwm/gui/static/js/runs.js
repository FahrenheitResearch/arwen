// My forecasts: one row per forecast folder under the root, newest first, each with its latest picture, its
// state and its shape. Read from disk every few seconds, so forecasts started from a terminal show here too.
// Forecasts waiting for the card are listed above, in the order they will start, with Move up, Move down and
// Remove.

import { h, chip, bar, fill, spacing, paceLine, prepLine, waitLine, button } from "./core.js";
import * as api from "./api.js";
import { register, go, openRoute, runRoute, notice, clearNotice } from "./router.js";

function newest(a, b) {
  const ta = a.status.updated_utc || a.status.start_time || "";
  const tb = b.status.updated_utc || b.status.start_time || "";
  return tb.localeCompare(ta) || a.id.localeCompare(b.id);
}

function row(r, w) {
  const st = r.status;
  const c = r.card || {};
  const thumb = h("div", { class: "thumb" }, c.thumb ? h("img", { src: api.filePath(r.id, c.thumb), alt: "", loading: "lazy" }) : null);
  const running = st.state === "running";
  const known = st.percent !== null && st.percent !== undefined;
  // A forecast waiting at a seam says what it waits on in place of its pace.
  const waiting = running ? waitLine(st, w.wait).join(" · ") : "";
  const pace = waiting || (running ? paceLine(st, w.mapviewer) : "");
  // A forecast still preparing says its stage and step here, as its map page does, rather than a 0% bar.
  const preparing = running && st.stage && !["forecast", "finalize"].includes(st.stage)
    ? [w.stages[st.stage] || st.stage, ...prepLine(st, w.preparation)].join(" · ") : "";
  const progress = preparing ? h("span", { class: "dim" }, preparing)
    : running && known ? h("div", { class: "prog", title: pace || null }, bar(st.percent, true), h("span", {}, `${st.percent.toFixed(0)}%`),
      pace ? h("span", { class: "dim" }, pace) : null)
    : c.pictures ? h("span", { class: "dim" }, fill(w.runs.pictures, { count: c.pictures.toLocaleString("en-US") })) : null;
  // A machine's run whose follower ended shows the last state received; the row says so.
  const lost = st.follow_lost ? h("div", { class: "fitline no" }, w.runs.updates_stopped) : null;
  const name = h("td", { class: "name" }, h("a", { href: `#/${runRoute("run", r.id)}` }, r.title || r.id),
    r.title && r.title !== r.id ? h("span", { class: "dim mono" }, r.id) : null);
  // A failed row says why on hover, the refusal and its remedy, as its Results page does.
  const end = st.end || {};
  const why = st.state === "failed" && end.message ? [end.message, end.remedy].filter(Boolean).join(" ") : "";
  const tr = h("tr", { class: "row", title: why || w.state_help[st.state] || "" },
    h("td", {}, thumb), name, h("td", {}, chip(st.state, w.states[st.state])),
    h("td", { class: "mono" }, st.start_time || ""),
    h("td", { class: "num" }, c.hours ? fill(w.runs.hours, { hours: +c.hours.toFixed(2) }) : ""),
    h("td", {}, (c.dx_km || []).map(spacing).join(" / ")),
    h("td", { class: "num" }, (c.levels || []).join(" / ")),
    h("td", {}, progress, lost));
  tr.addEventListener("click", (ev) => { if (!ev.target.closest("a")) go(openRoute(r)); });
  return tr;
}

// The queue: place, name, machine, when it was queued, why it waits (or is held), and the three buttons.
function queuePanel(queue, w, onchange) {
  const q = w.runs;
  const items = queue.items || [];
  if (!items.length) return null;
  const act = (run, action) => async () => {
    try {
      const reply = await api.post(`/api/queue/${encodeURIComponent(run)}/${action}`);
      // Out of the line, but a folder that could not be deleted is said, with what to do about it.
      if (action === "remove") notice(reply.kept ? reply.message : fill(q.removed, { run }), reply.kept ? "warn" : "");
    } catch (err) {
      notice(`${err.message} ${err.fix || ""}`.trim(), "stop");
    }
    onchange();
  };
  const rows = items.map((it, i) => h("tr", { class: "row" },
    h("td", { class: "num mono" }, String(i + 1)),
    h("td", { class: "name" }, h("a", { href: `#/${runRoute("run", it.run)}` }, it.title || it.run)),
    h("td", {}, it.machine === "this-computer" ? q.this_computer : it.machine),
    h("td", { class: "mono" }, String(it.queued_utc || "").replace("T", " ").replace("Z", "")),
    // Held only until its start is published is an ordinary wait; held for want of disk or card needs the person.
    h("td", { class: "why" }, it.held && !it.awaits_data ? h("span", { class: "fitline no" }, it.held)
      : h("span", { class: "dim" }, it.held || it.waiting || q.waits)),
    h("td", {}, h("div", { class: "btns" },
      button(q.move_up, { small: true, disabled: i === 0, onclick: act(it.run, "up") }),
      button(q.move_down, { small: true, disabled: i === items.length - 1, onclick: act(it.run, "down") }),
      button(q.remove, { small: true, onclick: act(it.run, "remove") })))));
  return h("div", {}, h("h2", { class: "cap" }, q.queue_title), h("p", { class: "note" }, q.queue_line),
    h("section", { class: "panel" }, h("div", { class: "tablewrap" }, h("table", { class: "dtable queuelist" },
      h("thead", {}, h("tr", {}, q.queue_columns.map((c, i) => h("th", { class: i === 0 ? "num" : null }, c)))),
      h("tbody", {}, rows)))));
}

function render(body, args, page) {
  const w = page.words.screens;
  page.setCrumbs([[w.shell.my_forecasts]]);
  page.right.append(h("a", { class: "btn primary", href: "#/create" }, w.runs.new));
  const queued = h("div", {});
  const holder = h("div", {});
  body.append(queued, holder, h("p", { class: "note wsec" }, w.runs.folder_note));
  let queueShown = "";
  let stopped = false;
  let shown = "";
  // Reads overlap (the timer, and each Move up, Move down or Remove), and their replies can arrive in any order.
  // Each read has a number; a reply is shown only when no later read's reply has been, and a queue change
  // retires every read begun before it, so a reply read before a Move up cannot put the old order back.
  let asked = 0;
  let settled = 0;

  async function refresh() {
    const mine = ++asked;
    let data;
    let queue;
    try {
      [data, queue] = await Promise.all([api.get("/api/runs"), api.get("/api/queue").catch(() => ({ items: [] }))]);
    } catch (err) {
      // A read that fails after the page was left says nothing on the page opened since.
      if (!stopped && mine > settled) {
        settled = mine;
        notice(w.app.lost, "warn");
      }
      return;
    }
    if (stopped || mine <= settled) return;
    settled = mine;
    // In touch again: the first good read takes the lost line down, and only that line, so the place a queued
    // forecast was given, or a Remove, is still said.
    clearNotice(w.app.lost);
    const queueKey = JSON.stringify(queue.items || []);
    if (queueKey !== queueShown) {
      queueShown = queueKey;
      queued.replaceChildren(...[queuePanel(queue, w, () => { queueShown = ""; settled = asked; refresh(); })].filter(Boolean));
      holder.className = (queue.items || []).length ? "wsec" : "";
    }
    // A queued forecast is listed in the queue above, not twice.
    const rows = data.runs.filter((r) => r.status.state !== "queued").sort(newest);
    const waiting = (queue.items || []).length;
    // Every folder counts, queued ones too, so this line and the sidebar give the same number.
    const all = data.runs.length;
    // A very large folder lists its newest forecasts and says how many there are in all.
    page.setData(data.truncated ? fill(w.runs.cut, { shown: all.toLocaleString("en-US"), total: (data.total || 0).toLocaleString("en-US") })
      : !all ? "" : waiting ? fill(w.runs.data_queued, { count: all, queued: waiting }) : fill(w.runs.data, { count: all }));
    const key = JSON.stringify([waiting > 0, ...rows.map((r) => [r.id, r.status.state, Math.round(r.status.percent || 0), r.card && r.card.thumb,
      r.status.grids && r.status.grids.pace_label, Math.round(r.status.speed_x || 0), !!r.status.follow_lost])]);
    if (key === shown) return;
    shown = key;
    // With forecasts waiting above, an empty list below says nothing: "No forecasts yet" would be wrong.
    if (!rows.length && waiting) {
      holder.replaceChildren();
      return;
    }
    if (!rows.length) {
      holder.replaceChildren(h("section", { class: "panel pad" }, h("h3", {}, w.runs.empty), h("p", { class: "note" }, w.runs.new_line),
        h("div", { class: "btns wsec" }, h("a", { class: "btn primary", href: "#/create" }, w.runs.new), h("a", { class: "btn", href: "#/browse" }, w.runs.from_wiki))));
      return;
    }
    holder.replaceChildren(h("section", { class: "panel" }, h("div", { class: "tablewrap" }, h("table", { class: "dtable runlist" },
      h("thead", {}, h("tr", {}, w.runs.columns.map((c, i) => h("th", { class: [4, 6].includes(i) ? "num" : null }, c)))),
      h("tbody", {}, rows.map((r) => row(r, w)))))));
  }

  refresh();
  const timer = setInterval(() => { if (!stopped) refresh(); }, 3000);
  return () => { stopped = true; clearInterval(timer); };
}

register("runs", render, { title: (words) => words.screens.runs.title });
