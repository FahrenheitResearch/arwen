// The assistant: a panel docked beside every page, opened from the sidebar or the Ctrl K palette. It sends the
// person's words to /api/assistant/say and shows what came back where the person can see it: the reply, where it
// was read from (the wiki, the run folder, the engine's fit), each typed decision under its letters with how sure
// it was, and any Start or Stop button. Those buttons send the page's own request, and only on the person's click.
// With no model on this computer the panel says so and says how to get one; it never waits on a model that is
// not there.
//
// The assistant is optional and ships off. Off, the panel and Settings show the model this card would get, its
// size and licence, and a Turn on button; nothing is downloaded, started or asked until the person turns it on,
// and even then every download is its own click on a list of files with their sizes and licences.

import { h, button, num, fill, fold, bar } from "./core.js";
import * as api from "./api.js";
import { go, goWithNotice, runRoute, notice, errorText } from "./router.js";
import { fillCreate, readForm } from "./bridge.js";
import { showCommand } from "./command.js";

const POLL_MS = 1500;
let panel = null;
// On or off, as the session first said and every status read since has said. The sidebar and Ctrl K read it.
let enabled = null;

export function assistantEnabled() {
  return enabled === true;
}

export function noteEnabled(on) {
  const value = on === true;
  if (enabled === value) return;
  enabled = value;
  document.dispatchEvent(new CustomEvent("assistant", { detail: { open: assistantOpen(), enabled: value } }));
}

function stamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `forecast-${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}-${p(d.getUTCHours())}${p(d.getUTCMinutes())}`;
}

const gb = (bytes) => `${num(bytes / 2 ** 30, 1)} GB`;
const share = (p) => (p === null || p === undefined ? "" : `${num(p * 100, 0)}%`);
const licenceLink = (licence) => (licence && licence.url
  ? h("a", { href: licence.url, target: "_blank", rel: "noreferrer" }, licence.id) : (licence && licence.id) || "");

// What Set up downloads, each file with its size and licence, and one Download button: nothing is fetched before
// that click. done() runs after the click or on Not now.
export async function installView(w, modelId, done) {
  let plan;
  try { plan = await api.dryRun("/api/assistant/install", { model: modelId }); } catch (err) {
    return h("p", { class: "notice stop" }, errorText(err));
  }
  if (!(plan.items || []).length) return h("p", { class: "note" }, w.all_here);
  const rows = plan.items.map((item) => h("li", {}, `${item.what}: ${gb(item.bytes)}, `, licenceLink(item.licence)));
  const confirm = button(`${w.download} ${gb(plan.bytes || 0)}`, { kind: "primary", onclick: async () => {
    confirm.disabled = true;
    try { await api.post("/api/assistant/install", { model: modelId, confirm: true }); } catch (err) {
      notice(errorText(err), "stop");
    }
    done();
  } });
  return h("div", { class: "notice" }, h("p", { class: "strong" }, w.download_title), h("ul", { class: "plain" }, rows),
    h("p", { class: "note" }, `${w.download_note} ${plan.home}.`), h("div", { class: "btns" }, confirm,
      button(w.not_now, { small: true, onclick: done })));
}

// On or off, with the way to change it. Off: the model this card would get, its size and licence, and Turn on.
// On: Turn off, and Remove its model when one was downloaded. changed(reply) runs after either, and the view it
// draws next says the new state to the sidebar, Ctrl K and the other view through noteEnabled: a view records what
// it shows before it draws this, so the event it causes does not send it back to the server.
export function switchView(w, s, changed) {
  noteEnabled(s.enabled);
  if (!s.enabled) {
    const m = s.model || {};
    const card = s.card_gib ? fill(w.off_card, { card: num(s.card_gib, 0) }) : w.off_no_card;
    const on = button(w.turn_on, { kind: "primary", onclick: async () => {
      on.disabled = true;
      try {
        const reply = await api.post("/api/assistant/enable", { on: true });
        notice("");
        changed(reply);
      } catch (err) {
        notice(errorText(err), "stop");
        on.disabled = false;
      }
    } });
    return h("div", { class: "notice", "data-assistant": "off" },
      h("p", { class: "strong" }, w.off_title), h("p", { class: "note" }, w.off_line),
      h("p", {}, card, " ", fill(w.off_model, { name: m.name || "", size: m.bytes ? gb(m.bytes) : "" }), " ",
        licenceLink(m.licence), "."),
      h("p", { class: "note" }, w.off_asks), h("div", { class: "btns" }, on));
  }
  const off = button(w.turn_off, { small: true, onclick: async () => {
    off.disabled = true;
    try {
      const reply = await api.post("/api/assistant/enable", { on: false });
      notice(reply.stopped ? w.turned_off_unloaded : w.turned_off);
      changed(reply);
    } catch (err) {
      notice(errorText(err), "stop");
      off.disabled = false;
    }
  } });
  const row = h("div", { class: "btns" }, off);
  if (s.downloaded_bytes) {
    const remove = button(fill(w.remove, { size: gb(s.downloaded_bytes) }), { small: true, kind: "danger" });
    remove.addEventListener("click", () => {
      const yes = button(fill(w.remove_yes, { size: gb(s.downloaded_bytes) }), { small: true, kind: "danger", onclick: async () => {
        yes.disabled = true;
        try {
          const reply = await api.post("/api/assistant/remove");
          notice(fill(w.removed, { size: gb(reply.freed_bytes || 0) }));
          changed(reply);
        } catch (err) {
          notice(errorText(err), "stop");
          yes.disabled = false;
        }
      } });
      row.replaceChildren(off, yes, button(w.keep, { small: true, onclick: () => changed(s) }));
    });
    row.append(remove);
  }
  return h("div", { "data-assistant": "on" }, h("p", { class: "note" }, w.on_line), row);
}

// One typed decision: the question, then every option under its letter, the chosen one marked, each with its share.
function decisionView(d, w) {
  const letters = w.option_letters || "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const options = d.options || [];
  const rows = options.map((id, i) => {
    const p = d.probabilities ? d.probabilities[id] : null;
    return h("li", { class: id === d.choice ? "on" : null, title: id },
      h("span", { class: "ltr" }, letters[i] || "?"), h("span", { class: "t" }, id),
      p === null || p === undefined ? h("span", {}) : bar(p * 100), h("span", { class: "p" }, share(p)));
  });
  const at = options.indexOf(d.choice);
  const sure = d.method === "unmeasured" ? w.decision_unmeasured : share(d.probability);
  return h("div", { class: "decision" },
    h("div", { class: "q" }, d.question),
    h("p", { class: "note" }, fill(w.decision_line, { choice: `${letters[at] || "?"} ${d.choice}`, p: sure, backend: d.backend })),
    rows.length ? h("ul", { class: "lettered" }, rows) : null,
    d.reason ? h("p", { class: "note" }, d.reason) : null);
}

function actionView(action, w, after) {
  if (action.type !== "confirm") return null;
  const box = h("div", { class: "panel pad" });
  let nameInput = null;
  if (action.action === "start" && action.needs_name) {
    nameInput = h("input", { class: "input", value: stamp(), spellcheck: "false", "aria-label": w.name });
    box.append(h("div", { class: "fields" }, h("label", { class: "wide" }, w.name, nameInput)));
  }
  const b = button(action.label || w.confirm_start, { kind: action.action === "stop" ? "danger" : "primary" });
  b.addEventListener("click", async () => {
    b.disabled = true;
    // One Start, sending what is on screen: the New forecast form as the person left it, and the plan as made only
    // when the form is not open.
    const onScreen = action.action === "start" ? readForm() : null;
    const body = onScreen && onScreen.lat !== null && onScreen.lon !== null ? { ...onScreen } : { ...(action.request.body || {}) };
    if (nameInput && !body.name) body.name = nameInput.value.trim();
    try {
      const reply = await api.post(action.request.path, body);
      if (action.action === "start") {
        const warning = reply.assistant && reply.assistant.warning;
        goWithNotice(runRoute("watch", reply.run),
          [w.started, warning || (reply.assistant && reply.assistant.message)].filter(Boolean).join(" "), warning ? "warn" : "");
      } else {
        notice(reply.message || w.stopped);
      }
      after();
    } catch (err) {
      notice(errorText(err), "stop");
      b.disabled = false;
    }
  });
  box.append(h("div", { class: "btns" }, b));
  if (action.action === "start") box.append(h("p", { class: "note" }, w.start_sends_form));
  if (action.command) box.append(showCommand(action.command));
  if (action.problem) box.append(h("p", { class: "fitline no" }, action.problem));
  return box;
}

// Where a turn's answer was read from, in the order the tools ran, each named once.
function fromLine(turn, w) {
  const names = { search_wiki: w.from_wiki, read_event: w.from_wiki, read_run: w.from_run, run_status: w.from_run,
    list_runs: w.from_run, check_fit: w.from_fit, plan_forecast: w.from_fit, list_sources: w.from_engine,
    list_physics: w.from_engine, this_computer: w.from_engine };
  const seen = [];
  for (const call of turn.tool_calls || []) {
    const label = names[call.name];
    if (label && call.ok !== false && !seen.includes(label)) seen.push(label);
  }
  return seen.length ? h("p", { class: "from" }, `${w.answered_from}: ${seen.join(", ")}`) : null;
}

function build(words) {
  const w = words.assistant || {};
  const el = h("aside", { class: "drawer docked", "aria-label": w.title, hidden: true });
  const close = button(w.close || "Close", { small: true, onclick: () => showPanel(false) });
  const status = h("div", { class: "status" });
  const log = h("div", { class: "chat" }, h("p", { class: "note" }, w.intro));
  const input = h("textarea", { class: "input", rows: "3", placeholder: w.placeholder, "aria-label": w.placeholder });
  const send = button(w.send, { kind: "primary" });
  let conversation = null;
  let timer = null;
  let detected = null;
  // On or off as this panel last drew it; a switch made in Settings redraws an open panel.
  let shown = null;
  document.addEventListener("assistant", (ev) => {
    const on = ev.detail && ev.detail.enabled;
    if (typeof on === "boolean" && on !== shown && !el.hidden) refresh();
  });

  // ---- settings: where the model runs, and who answers the typed decisions
  const where = h("select", { class: "input" }, h("option", { value: "bundled" }, w.where_bundled), h("option", { value: "endpoint" }, w.where_endpoint));
  const url = h("input", { class: "input", placeholder: "http://127.0.0.1:1234/v1", spellcheck: "false" });
  const model = h("input", { class: "input", spellcheck: "false" });
  const key = h("input", { class: "input", type: "password", autocomplete: "off" });
  const who = h("select", { class: "input" }, h("option", { value: "local" }, w.who_local), h("option", { value: "jev" }, w.who_jev),
    h("option", { value: "kev" }, w.who_kev));
  const dUrl = h("input", { class: "input", spellcheck: "false" });
  const dKey = h("input", { class: "input", type: "password", autocomplete: "off" });
  // The fields the person has edited since the server last filled them. The server's answer fills every other
  // field, Settings open or not, so one opened while the answer was on its way still shows the saved values; a
  // save sends only the edited ones, so a field that was never filled in never overwrites what is saved.
  const fields = { backend: where, decisions: who, endpoint_url: url, endpoint_model: model, decision_url: dUrl };
  const edited = new Set();
  for (const [name, input] of Object.entries(fields)) {
    for (const kind of ["input", "change"]) input.addEventListener(kind, () => edited.add(name));
  }
  async function saveSettings(extra = null) {
    let body = extra;
    if (!body) {
      body = Object.fromEntries([...edited].map((name) => [name, fields[name].value.trim()]));
      if (key.value) body.key = key.value;
      if (dKey.value) body.decision_key = dKey.value;
    }
    // A field edited again while the save is on its way holds a value this save did not send: it stays edited, so
    // the status read after the save does not put the older saved value back over it, and the next Save sends it.
    const sent = Object.keys(body).filter((name) => fields[name]).map((name) => [name, fields[name].value]);
    try {
      await api.post("/api/assistant/settings", body);
      for (const [name, value] of sent) if (fields[name].value === value) edited.delete(name);
    } catch (err) {
      notice(errorText(err), "stop");
    }
    refresh();
  }
  const power = h("div", {});
  const settings = fold(w.settings, power,
    h("div", { class: "fields" },
      h("label", { class: "wide" }, w.where, where), h("label", { class: "wide" }, w.endpoint_url, url),
      h("label", {}, w.endpoint_model, model), h("label", {}, w.key, key),
      h("label", { class: "wide" }, w.who, who), h("label", {}, w.decision_url, dUrl), h("label", {}, w.decision_key, dKey)),
    h("div", { class: "btns" }, button(w.save, { small: true, onclick: () => saveSettings() })));

  // ---- what the panel says about its model: loaded, off the card, downloading, or none at all
  function noModel(s) {
    const found = (detected || []).filter((row) => row.models && row.models.length);
    const kids = [
      h("p", { class: "strong" }, w.no_model_title), h("p", { class: "note" }, w.no_model_line),
      h("ul", { class: "plain" },
        h("li", {}, fill(w.no_model_download, { name: s.model.name, size: gb(s.model.bytes), licence: s.model.licence.id, home: s.home }),
          " ", button(w.set_up, { small: true, onclick: () => setUp(s.model.id) })),
        h("li", {}, w.no_model_server)),
    ];
    for (const row of found) {
      kids.push(h("p", { class: "note" }, fill(w.found_server, { name: row.name || row.id, url: row.url }), " ",
        button(w.use_server, { small: true, onclick: () => saveSettings({ backend: "endpoint", endpoint_url: row.url,
          endpoint_model: row.models[0], endpoint_kind: row.id }) })));
    }
    return h("div", { class: "notice warn" }, kids);
  }

  async function refresh() {
    let s;
    try { s = await api.get("/api/assistant"); } catch (err) {
      status.replaceChildren(h("p", { class: "notice stop" }, errorText(err)));
      return;
    }
    // Settings shows what is saved, on or off: the fold is there either way, and a field it never filled in would
    // read as a saved choice gone.
    const saved = { backend: s.backend, decisions: s.decisions, endpoint_url: s.endpoint || "",
      endpoint_model: s.endpoint_model || "", decision_url: s.decision_url || "" };
    for (const [name, input] of Object.entries(fields)) if (!edited.has(name)) input.value = saved[name];
    // Off: only the switch. The box stays shut so nothing can be sent to a model that is not wanted.
    shown = s.enabled === true;
    input.disabled = send.disabled = !s.enabled;
    input.placeholder = s.enabled ? w.placeholder : w.off_placeholder;
    power.replaceChildren(s.enabled ? switchView(w, s, afterSwitch) : "");
    if (!s.enabled) {
      log.hidden = true;
      status.replaceChildren(switchView(w, s, afterSwitch));
      return;
    }
    log.hidden = false;
    const parts = [];
    if (s.backend === "endpoint") {
      parts.push(h("p", { class: "note" }, `${s.endpoint_model || "model"}, ${s.endpoint || ""}`));
    } else if (s.loaded) {
      parts.push(h("p", { class: "note" }, `${s.model.name}: ${w.loaded}${s.vram_gib ? `, ${num(s.vram_gib, 1)} GB` : ""} `,
        button(w.unload, { small: true, onclick: async () => { await api.post("/api/assistant/unload"); refresh(); } })));
    } else if (s.downloaded && s.server_found) {
      parts.push(h("p", { class: "note" }, `${s.model.name}: ${w.not_loaded}`));
    } else if (["downloading", "unpacking", "starting"].includes(s.install.state)) {
      const got = s.install.bytes ? ` ${gb(s.install.bytes)} / ${gb(s.install.total)}` : "";
      parts.push(h("p", { class: "note" }, `${w.downloading} ${s.install.what || ""}${got}`));
      clearTimeout(timer);
      timer = setTimeout(refresh, POLL_MS);
    } else {
      if (s.install.state === "failed") parts.push(h("p", { class: "notice stop" }, s.install.message));
      if (detected === null) {
        detected = [];
        api.get("/api/assistant/catalog").then((c) => { detected = c.detected || []; refresh(); }).catch(() => {});
      }
      parts.push(noModel(s));
    }
    if (s.note) parts.push(h("p", { class: "notice warn" }, s.note));
    status.replaceChildren(...parts);
  }

  async function setUp(modelId) {
    status.replaceChildren(await installView(w, modelId, refresh));
  }

  // Just turned on with nothing downloaded: the download list comes up at once, to be said yes or no to.
  function afterSwitch(reply) {
    if (reply && reply.enabled && reply.install && (reply.install.items || []).length) {
      shown = true;
      input.disabled = send.disabled = false;
      log.hidden = false;
      input.placeholder = w.placeholder;
      power.replaceChildren(switchView(w, reply, afterSwitch));
      setUp(reply.model.id);
      return;
    }
    refresh();
  }

  // Pages first, then fills: a fill made before New forecast draws waits for it in bridge.js.
  function apply(turn) {
    for (const action of turn.actions || []) {
      if (action.type !== "go") continue;
      const route = action.run ? runRoute(action.page, action.run)
        : action.event ? `event/${encodeURIComponent(action.event)}` : action.page;
      if (location.hash.replace(/^#\/?/, "") !== route) go(route);
    }
    for (const action of turn.actions || []) {
      if (action.type === "fill_create") fillCreate(action);
    }
  }

  function showTurn(turn) {
    const decisions = (turn.decisions || []).map((d) => decisionView(d, w));
    const fixes = ((turn.plan && turn.plan.fixes) || []).map((fix) => h("li", {}, fix.words));
    const confirm = (turn.actions || []).map((a) => actionView(a, w, refresh)).filter(Boolean);
    const steps = (turn.tool_calls || []).map((c) => h("li", {}, h("code", {}, c.name),
      ` ${num(c.seconds, 1)} s${c.ok === false ? `: ${c.error}` : ""}`));
    log.append(h("div", { class: "turn" },
      h("p", {}, turn.reply || turn.error || ""),
      fromLine(turn, w),
      fixes.length ? h("div", { class: "notice warn" }, h("b", {}, w.given_up), h("ul", { class: "plain" }, fixes)) : null,
      confirm,
      decisions.length ? h("div", {}, h("p", { class: "cap" }, `${w.choices} (${decisions.length})`), decisions) : null,
      steps.length ? fold(`${w.steps}: ${num(turn.seconds, 1)} s`, h("ul", { class: "plain" }, steps)) : null));
    body.scrollTop = body.scrollHeight;
  }

  // One message at a time: Enter in the box reaches say() without the Send button, so the button being disabled
  // is not enough on its own to stop a second message while the first reply is pending.
  let sending = false;
  async function say() {
    if (sending) return;
    const text = input.value.trim();
    if (!text) return;
    sending = true;
    input.value = "";
    log.append(h("p", { class: "said" }, text));
    const waiting = h("p", { class: "note" }, w.thinking);
    log.append(waiting);
    body.scrollTop = body.scrollHeight;
    send.disabled = true;
    try {
      const turn = await api.post("/api/assistant/say", { text, conversation, form: readForm() });
      conversation = turn.conversation;
      waiting.remove();
      showTurn(turn);
      apply(turn);
    } catch (err) {
      waiting.remove();
      log.append(h("div", { class: "turn" }, h("p", { class: "notice warn" }, errorText(err))));
    } finally {
      sending = false;
      send.disabled = false;
      refresh();
    }
  }
  send.addEventListener("click", say);
  input.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); say(); } });

  const body = h("div", { class: "drawer-body" }, status, log);
  el.append(
    h("div", { class: "drawer-head" }, h("b", {}, w.title), close),
    body,
    h("div", { class: "drawer-foot" }, input, h("div", { class: "btns" }, h("span", { class: "hint" }, w.shortcut), send), settings));
  el.refresh = refresh;
  el.focusInput = () => input.focus();
  return el;
}

// The panel outlives every page: the shell puts it back after each page change.
export function attachAssistant(words) {
  if (!panel) panel = build(words);
  if (panel.parentNode !== document.body) document.body.append(panel);
  document.body.classList.toggle("with-panel", !panel.hidden);
  return panel;
}

export function showPanel(open) {
  if (!panel) return;
  panel.hidden = !open;
  document.body.classList.toggle("with-panel", open);
  document.dispatchEvent(new CustomEvent("assistant", { detail: { open } }));
  if (open) { panel.refresh(); panel.focusInput(); }
}

export function openAssistant(words) {
  attachAssistant(words);
  showPanel(true);
}

export function assistantOpen() {
  return !!(panel && !panel.hidden);
}
