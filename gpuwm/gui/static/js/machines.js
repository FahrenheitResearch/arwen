// Machines: the computers that run and draw forecasts. This computer first, then every row of the machines table
// (an SSH host or a cloud machine), each checked: its card, what it is doing, the gpuwm version there. Add a
// computer by its SSH host (checked over SSH before it is saved), check or remove a row, and send any forecast's
// pictures to be drawn on any machine. Every button is one request to /api/machines or /api/runs/RUN/render.

import { h, button, chip, fill } from "./core.js";
import * as api from "./api.js";
import { register, notice, errorText, runRoute } from "./router.js";
import { actionButton } from "./command.js";

const REFRESH_MS = 15000;

function cardText(row, w) {
  const cards = row.cards || [];
  if (!cards.length) return h("span", { class: "dim" }, row.card_error || w.no_card);
  return cards.map((c) => h("div", {}, fill(w.card_line, {
    name: c.name || "card", gib: ((c.memory_total_mib || (c.memory_total_gib || 0) * 1024) / 1024).toFixed(0) })));
}

function versionText(row, w) {
  if (row.kind === "local") return h("span", { class: "mono" }, row.version_here || "");
  if (!row.version_there) return h("span", { class: "dim" }, row.reachable === false ? "" : w.version_none);
  // The version answers before its requirements are in, so an install still going or one that left them out is
  // said here instead of "same as here".
  if (row.version_matches && row.not_ready) return h("span", { class: "fitline no" }, fill(w.version_not_ready, { there: row.version_there, why: row.not_ready.label }));
  if (row.version_matches) return h("span", {}, h("span", { class: "mono" }, row.version_there), " ", h("span", { class: "dim" }, w.version_ok));
  return h("span", { class: "fitline no" }, fill(w.version_other, { there: row.version_there, here: row.version_here }));
}

function doing(row) {
  const bits = [];
  // This computer's line names the forecast that holds its card and opens that forecast's page.
  if (row.detail) bits.push(h("div", {}, row.run ? h("a", { href: `#/${runRoute("run", row.run)}` }, row.detail) : row.detail));
  if (row.fix) bits.push(h("div", { class: "note" }, row.fix));
  const jobs = row.jobs || {};
  for (const f of jobs.forecasts || []) bits.push(h("div", { class: "note" }, `${f.run}: ${f.state}`));
  for (const r of jobs.renders || []) bits.push(h("div", { class: "note" }, `${r.run}: ${r.state}`));
  return bits;
}

// The install request for one machine, or {error} naming what the page still needs. The wheel folder is required:
// the engine and data wheels of this computer's version are copied from it (on this computer, or on the source
// machine named) to the target.
export function installRequest(name, folder, from, w) {
  const wheelhouse = String(folder || "").trim();
  if (!wheelhouse) return { error: w.install_need_folder };
  const body = { wheelhouse };
  if (from) body.from_machine = from;
  return { path: `/api/machines/${encodeURIComponent(name)}/install`, body };
}

// The Install box under the list: the wheel folder, where it is, a check that finds the two wheels without copying
// anything, and Install. `done` runs after an install starts.
export function installBox(target, rows, w, { folder = "", done = () => {} } = {}) {
  const folderIn = h("input", { class: "input", spellcheck: "false", autocomplete: "off", value: folder || null,
    "data-field": "wheelhouse", placeholder: w.install_folder_example });
  const fromPick = h("select", { class: "input", "data-field": "from_machine" },
    rows.filter((r) => r.name !== target.name && (r.kind === "local" || r.reachable !== false))
      .map((r) => h("option", { value: r.kind === "local" ? "" : r.name }, r.kind === "local" ? w.this_computer : r.name)));
  const words = h("div", { class: "install-words" });
  const say = (text, kind = "") => words.replaceChildren(h("p", { class: `fitline ${kind}`.trim() }, text));
  const request = () => installRequest(target.name, folderIn.value, fromPick.value, w);
  const run = async (dry) => {
    const req = request();
    if (req.error) { say(req.error, "no"); folderIn.focus(); return null; }
    say(dry ? w.install_checking : w.install_starting, "wait");
    try {
      const reply = await api.post(req.path, dry ? { ...req.body, dry_run: true } : req.body);
      if (dry) {
        const names = (reply.wheels || []).map((path) => String(path).split(/[\\/]/).pop());
        say(fill(w.install_found, { wheels: names.join(", "), machine: target.name }));
      } else {
        say(reply.message || "");
        done(reply);
      }
      return reply;
    } catch (err) {
      say(errorText(err), "no");
      return null;
    }
  };
  const box = h("section", { class: "panel pad", "data-install": target.name },
    h("h3", {}, fill(w.install_title, { name: target.name })),
    h("p", { class: "note" }, target.offer && target.offer.words ? target.offer.words : w.install_line),
    h("div", { class: "fields wsec" },
      h("label", { class: "wide" }, w.install_folder, folderIn, h("span", { class: "hint" }, w.install_folder_hint)),
      h("label", {}, w.install_from, fromPick, h("span", { class: "hint" }, w.install_from_hint))),
    words,
    h("div", { class: "btns wsec" },
      button(w.install_check, { onclick: () => run(true) }),
      button(w.install, { kind: "primary", onclick: () => run(false) })));
  box.run = run;
  box.folder = folderIn;
  box.from = fromPick;
  return box;
}

function render(body, args, page) {
  const w = page.words.screens.machines;
  const st = page.words.screens;
  page.setCrumbs([[w.title]]);
  let closed = false;
  let rows = [];
  let runs = [];
  let preset = args[0] ? decodeURIComponent(args[0]) : "";
  let wheelhouse = "";

  const list = h("section", { class: "panel" });
  const addBox = h("section", { class: "panel pad", hidden: true });
  const installSlot = h("div", {});
  const drawBox = h("section", { class: "panel pad" });
  const foot = h("p", { class: "note wsec" });
  const addButton = button(w.add, { kind: "primary", onclick: () => { addBox.hidden = !addBox.hidden; if (!addBox.hidden) nameIn.focus(); } });
  page.right.append(addButton);
  body.append(h("p", { class: "lede" }, w.line), addBox, list, installSlot, h("h2", { class: "sec wsec" }, w.draw_title), drawBox, foot);

  // ---- add a computer by its SSH host
  const nameIn = h("input", { class: "input", spellcheck: "false", autocomplete: "off" });
  const hostIn = h("input", { class: "input", spellcheck: "false", autocomplete: "off", placeholder: "me@gpu-box" });
  const portIn = h("input", { class: "input num", type: "number", min: "1", max: "65535", placeholder: "22" });
  const workIn = h("input", { class: "input", spellcheck: "false", placeholder: "~/gpuwm-machine" });
  const geogIn = h("input", { class: "input", spellcheck: "false" });
  const addWords = h("div", {});
  const row = () => {
    const out = { name: nameIn.value.trim(), kind: "ssh", host: hostIn.value.trim() };
    if (portIn.value) out.port = Number(portIn.value);
    if (workIn.value.trim()) out.workspace = workIn.value.trim();
    if (geogIn.value.trim()) out.geog_root = geogIn.value.trim();
    return out;
  };
  const save = actionButton(w.save, { kind: "primary", onclick: async () => {
    save.button.disabled = true;
    addWords.replaceChildren(h("p", { class: "fitline wait" }, w.saving));
    try {
      const reply = await api.post("/api/machines/add", row());
      addWords.replaceChildren(h("p", { class: "fitline" }, reply.message));
      addBox.hidden = true;
      refresh(true);
    } catch (err) {
      addWords.replaceChildren(h("p", { class: "fitline no" }, errorText(err)));
    } finally {
      save.button.disabled = false;
    }
  } });
  addBox.append(
    h("h3", {}, w.add_title), h("p", { class: "note" }, w.add_line),
    h("div", { class: "fields wsec" },
      h("label", {}, w.name, nameIn, h("span", { class: "hint" }, w.name_hint)),
      h("label", {}, w.host, hostIn, h("span", { class: "hint" }, w.host_hint)),
      h("label", {}, w.port, portIn),
      h("label", {}, w.workspace, workIn, h("span", { class: "hint" }, w.workspace_hint)),
      h("label", { class: "wide" }, w.geog_root, geogIn, h("span", { class: "hint" }, w.geog_hint))),
    addWords,
    h("div", { class: "btns wsec" }, save, button(w.cancel, { onclick: () => { addBox.hidden = true; } })));

  // ---- the list
  async function act(name, action, bodyOut = {}) {
    try {
      const reply = await api.post(`/api/machines/${encodeURIComponent(name)}/${action}`, bodyOut);
      notice(reply.message || (action === "remove" ? fill(w.removed, { name }) : ""));
    } catch (err) {
      notice(errorText(err), "stop");
    }
    refresh(true);
  }
  function actions(r) {
    const out = [];
    if (r.kind !== "local") {
      out.push(button(w.check, { small: true, onclick: () => act(r.name, "check") }));
      if (r.kind !== "ssh") {
        out.push(button(w.start, { small: true, onclick: () => act(r.name, "start") }),
          button(w.stop, { small: true, onclick: () => act(r.name, "stop") }));
      }
      if (r.offer && r.offer.action === "install") {
        out.push(button(w.install, { small: true, title: r.offer.words, onclick: () => openInstall(r) }));
      }
      out.push(button(w.remove, { small: true, kind: "danger", onclick: () => act(r.name, "remove") }));
    }
    out.push(button(w.draw, { small: true, onclick: () => { preset = r.name; paintDraw(); drawBox.scrollIntoView({ block: "nearest" }); } }));
    return h("div", { class: "btns" }, out);
  }
  function openInstall(r) {
    const box = installBox(r, rows, w, { folder: wheelhouse, done: () => refresh(true) });
    installSlot.replaceChildren(box, h("div", { class: "btns" }, button(w.cancel, { small: true, onclick: () => installSlot.replaceChildren() })));
    box.folder.focus();
    box.scrollIntoView({ block: "nearest" });
  }
  function paintList() {
    const trs = rows.map((r) => {
      const state = r.state || (r.reachable === false ? "offline" : "idle");
      const kind = state === "running" || state === "rendering" ? "running" : state === "offline" ? "failed" : state === "busy" ? "stale" : "ready";
      return h("tr", {},
        h("td", {}, h("b", {}, r.kind === "local" ? w.this_computer : r.name),
          h("span", { class: "note mono" }, r.kind === "local" ? r.name : [r.host, r.kind !== "ssh" ? ` (${r.kind})` : ""].join(""))),
        h("td", {}, chip(kind, state)),
        h("td", {}, cardText(r, w)),
        h("td", {}, versionText(r, w)),
        h("td", {}, doing(r)),
        h("td", {}, actions(r)));
    });
    list.replaceChildren(...[h("div", { class: "tablewrap" }, h("table", { class: "dtable" },
      h("thead", {}, h("tr", {}, [...w.cols, ""].map((c) => h("th", {}, c)))), h("tbody", {}, trs))),
    rows.length < 2 ? h("div", { class: "pft" }, h("span", {}, w.empty)) : null].filter(Boolean));
  }

  // ---- draw a forecast on a machine
  const runPick = h("select", { class: "input" });
  const machinePick = h("select", { class: "input" });
  const products = h("select", { class: "input" }, h("option", { value: "" }, w.draw_products_default), h("option", { value: "all" }, w.draw_products_all));
  const drawWords = h("div", {});
  const drawRequest = () => (runPick.value && machinePick.value
    ? { path: `${api.runPath(runPick.value)}/render`, body: { machine: machinePick.value, products: products.value || null } }
    : { error: w.no_runs });
  const drawButton = actionButton(w.draw, { kind: "primary", request: drawRequest, watch: [runPick, machinePick, products],
    onclick: async () => {
      const req = drawRequest();
      if (req.error) { drawWords.replaceChildren(h("p", { class: "fitline no" }, req.error)); return; }
      drawButton.button.disabled = true;
      try {
        await api.post(req.path, req.body);
        const shown = (rows.find((r) => r.name === machinePick.value) || {}).kind === "local" ? w.this_computer : machinePick.value;
        drawWords.replaceChildren(h("p", { class: "fitline" }, fill(w.drawing, { machine: shown }), " ",
          h("a", { href: `#/${runRoute("results", runPick.value)}` }, st.shell.map)));
      } catch (err) {
        drawWords.replaceChildren(h("p", { class: "fitline no" }, errorText(err)));
      } finally {
        drawButton.button.disabled = false;
      }
    } });
  function paintDraw() {
    const keepRun = runPick.value;
    runPick.replaceChildren(...runs.map((r) => h("option", { value: r.id }, `${r.title || r.id} (${st.states[r.status.state] || r.status.state})`)));
    if (runs.some((r) => r.id === keepRun)) runPick.value = keepRun;
    const keepMachine = preset || machinePick.value;
    machinePick.replaceChildren(...rows.map((r) => h("option", { value: r.name }, r.kind === "local" ? w.this_computer : r.name)));
    if (rows.some((r) => r.name === keepMachine)) machinePick.value = keepMachine;
    drawBox.replaceChildren(...[h("p", { class: "note" }, w.draw_line),
      runs.length ? h("div", { class: "fields wsec" }, h("label", {}, w.draw_run, runPick), h("label", {}, w.draw_machine, machinePick),
        h("label", {}, w.draw_products, products)) : h("p", { class: "fitline wait" }, w.no_runs),
      drawWords, runs.length ? h("div", { class: "btns wsec" }, drawButton) : null].filter(Boolean));
  }

  async function refresh(fresh = false) {
    try {
      const [m, r] = await Promise.all([api.get(`/api/machines${fresh ? "?fresh=1" : ""}`), api.get("/api/runs").catch(() => ({ runs: [] }))]);
      if (closed) return;
      rows = m.machines || [];
      wheelhouse = m.wheelhouse || wheelhouse;
      runs = (r.runs || []).slice().sort((a, b) => String(b.status.updated_utc || "").localeCompare(String(a.status.updated_utc || "")));
      // The table's file by name; its full place is in the tooltip, not the page's words.
      foot.textContent = fill(w.file_note, { file: String(m.file || "").split(/[\\/]/).pop() });
      foot.title = m.file || "";
      paintList();
      paintDraw();
      document.body.dataset.machines = String(rows.length);
    } catch (err) {
      if (!closed) list.replaceChildren(h("p", { class: "notice stop" }, errorText(err)));
    }
  }
  const done = refresh();
  const timer = setInterval(() => { if (!closed) refresh(); }, REFRESH_MS);
  return done.then(() => () => { closed = true; clearInterval(timer); });
}

register("machines", render, { title: (words) => words.screens.machines.title });
