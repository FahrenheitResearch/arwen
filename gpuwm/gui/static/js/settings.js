// Settings: the choices that belong to this computer rather than to one forecast. Today that is the assistant, which
// is optional and off until turned on here or in its panel. Off, this page shows the model this card would get, its
// size and licence; on, it offers the download (a list of files with their sizes and licences, then one click),
// Turn off, and Remove its model. The tables below list every model the assistant can download for each card and
// the models a person can bring on their own server, each with its licence.

import { h, button, num, table, fold } from "./core.js";
import * as api from "./api.js";
import { register, errorText } from "./router.js";
import { switchView, installView, openAssistant } from "./assistant.js";

const gb = (bytes) => `${num(bytes / 2 ** 30, 1)} GB`;
const link = (href, text) => h("a", { href, target: "_blank", rel: "noreferrer" }, text);

async function render(body, args, page) {
  const w = page.words.assistant || {};
  const ws = page.words.screens.settings || {};
  const box = h("div", {});
  // The download list, once asked for: files, sizes, licences and one Download button.
  const offered = h("div", {});
  const tables = h("div", {});
  let timer = null;
  body.append(h("section", { class: "panel pad" }, h("h2", { class: "cap" }, ws.assistant), h("p", { class: "note" }, ws.assistant_line),
    box, offered), tables);

  // On or off as this page last drew it; a switch made in the assistant panel redraws the page.
  let shown = null;
  const heard = (ev) => {
    const on = ev.detail && ev.detail.enabled;
    if (typeof on === "boolean" && on !== shown) refresh();
  };

  async function refresh(reply = null) {
    clearTimeout(timer);
    let s = reply && reply.model ? reply : null;
    try { if (!s || s.dry_run) s = await api.get("/api/assistant"); } catch (err) {
      box.replaceChildren(h("p", { class: "notice stop" }, errorText(err)));
      return;
    }
    shown = s.enabled === true;
    const kids = [switchView(w, s, (next) => {
      if (next && typeof next.enabled === "boolean") shown = next.enabled;
      if (next && next.enabled && next.install && (next.install.items || []).length) {
        refresh().then(() => offer(next.model.id));
      } else {
        refresh();
      }
    })];
    if (s.enabled) {
      const m = s.model;
      const install = s.install || {};
      const btns = [];
      kids.push(h("p", {}, `${m.name}, ${m.quant || ""}, ${gb(m.bytes)}, `, link(m.licence.url, m.licence.id),
        `, ${num((m.context || 0) / 1024, 0)}K ${ws.context}.`));
      if (m.why) kids.push(h("p", { class: "note" }, m.why));
      if (s.backend === "endpoint") {
        kids.push(h("p", { class: "note" }, `${ws.uses_server} ${s.endpoint || ""}`));
      } else if (["starting", "downloading", "unpacking"].includes(install.state)) {
        const got = install.total ? ` ${gb(install.bytes || 0)} / ${gb(install.total)}` : "";
        kids.push(h("p", { class: "note" }, `${w.downloading} ${install.what || ""}${got}`));
        timer = setTimeout(refresh, 1500);
      } else if (!s.downloaded) {
        if (install.state === "failed" && install.message) kids.push(h("p", { class: "notice stop" }, install.message));
        if (!offered.childElementCount) btns.push(button(ws.set_up, { kind: "primary", onclick: () => offer(m.id) }));
      } else {
        kids.push(h("p", { class: "note" }, ws.downloaded));
      }
      btns.push(button(ws.open_panel, { small: true, onclick: () => openAssistant(page.words) }));
      kids.push(h("div", { class: "btns" }, btns));
    } else {
      offered.replaceChildren();
    }
    box.replaceChildren(...kids);
  }

  async function offer(modelId) {
    const view = await installView(w, modelId, () => { offered.replaceChildren(); refresh(); });
    offered.replaceChildren(view);
    refresh();
  }

  try {
    const c = await api.get("/api/assistant/catalog");
    const rows = (c.models || []).slice().sort((a, b) => a.min_card_gib - b.min_card_gib)
      .map((m) => h("tr", {}, h("td", {}, `${m.min_card_gib} GB`), h("td", {}, m.name), h("td", {}, m.quant || ""),
        h("td", {}, gb(m.bytes)), h("td", {}, `${num((m.context || 0) / 1024, 0)}K`), h("td", {}, link(m.licence.url, m.licence.id)),
        h("td", {}, link(`https://huggingface.co/${m.repo}`, m.repo || ""))));
    const brought = (c.brought || []).map((m) => h("tr", {}, h("td", {}, link(m.where, m.name)), h("td", {}, m.maker || ""),
      h("td", {}, `${m.min_card_gib} GB`), h("td", {}, link(m.licence.url, m.licence.id)), h("td", {}, m.why_not_bundled)));
    tables.append(
      fold(ws.models_title, h("p", { class: "note" }, ws.models_line), table(ws.models_cols, rows)),
      fold(ws.brought_title, h("p", { class: "note" }, ws.brought_line), table(ws.brought_cols, brought)));
  } catch (err) {
    tables.append(h("p", { class: "notice stop" }, errorText(err)));
  }
  await refresh();
  document.addEventListener("assistant", heard);
  return () => { clearTimeout(timer); document.removeEventListener("assistant", heard); };
}

register("settings", render, { title: (words) => words.screens.settings.title });
