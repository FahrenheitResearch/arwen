// "Show command": every action on this page is one gpuwm command. This is the one component that shows it:
// a toggle that reveals the exact line, copyable. Opening it asks the server for the line with
// {"dry_run": true}, so nothing runs and nothing changes.
//
//   showCommand(line)          a line already known (a reply's command)
//   liveCommand(request)       a line that depends on what the form holds; request() returns {path, body},
//                              or {error} with what to fix first. {open: true} draws it open, asking at once;
//                              isOpen() says whether it is open; retire() stops a line taken off the page from
//                              asking again when a watched field changes
//   actionButton(label, opts)  a button with its Show command beside it

import { h, copyline, button, append } from "./core.js";
import * as api from "./api.js";

export function showCommand(command, opts = {}) {
  if (!command) return null;
  const line = copyline(command, "The command this runs");
  line.hidden = !opts.open;
  const label = opts.label || "Show command";
  const toggle = h("button", { type: "button", class: "link", "aria-expanded": opts.open ? "true" : "false" },
    opts.open ? "Hide command" : label);
  toggle.addEventListener("click", () => {
    line.hidden = !line.hidden;
    toggle.textContent = line.hidden ? label : "Hide command";
    toggle.setAttribute("aria-expanded", line.hidden ? "false" : "true");
  });
  return h("div", { class: "showcmd", dataset: { command } }, toggle, line);
}

export function liveCommand(request, opts = {}) {
  const label = opts.label || "Show command";
  const out = h("div", { class: "livecmd" });
  out.hidden = !opts.open;
  const toggle = h("button", { type: "button", class: "link", "aria-expanded": opts.open ? "true" : "false" },
    opts.open ? "Hide command" : label);
  const box = h("div", { class: "showcmd live", dataset: { command: "" } }, toggle, out);
  let ticket = 0;
  let retired = false;
  const say = (text) => h("p", { class: "note" }, text);
  async function refresh() {
    if (out.hidden || retired) return;
    const mine = ++ticket;
    let req;
    try { req = request(); } catch (err) { req = { error: err.message }; }
    if (!req || req.error) {
      out.replaceChildren(say((req && req.error) || "Fill in the form first; the command shows here."));
      return;
    }
    out.replaceChildren(say("Asking the server for the exact line."));
    try {
      const reply = await api.dryRun(req.path, req.body || {});
      if (mine !== ticket) return;
      box.dataset.command = reply.command || "";
      out.replaceChildren(copyline(reply.shown || reply.command, "The command this runs", reply.command));
    } catch (err) {
      if (mine !== ticket) return;
      out.replaceChildren();
      append(out, [err.command ? copyline(err.shown || err.command, null, err.command) : null, say(`${err.message} ${err.fix || ""}`.trim())]);
    }
  }
  let timer = null;
  const soon = () => { clearTimeout(timer); timer = setTimeout(refresh, 250); };
  const watched = [].concat(opts.watch || []).filter(Boolean);
  for (const el of watched) {
    el.addEventListener("input", soon);
    el.addEventListener("change", soon);
  }
  toggle.addEventListener("click", () => {
    out.hidden = !out.hidden;
    toggle.textContent = out.hidden ? label : "Hide command";
    toggle.setAttribute("aria-expanded", out.hidden ? "false" : "true");
    refresh();
  });
  // Drawn open, it asks once the caller has put it on the page.
  if (opts.open) setTimeout(refresh, 0);
  box.refresh = refresh;
  box.isOpen = () => !out.hidden;
  box.retire = () => {
    retired = true;
    clearTimeout(timer);
    for (const el of watched) {
      el.removeEventListener("input", soon);
      el.removeEventListener("change", soon);
    }
  };
  return box;
}

export function actionButton(label, opts = {}) {
  const b = button(label, { kind: opts.kind, small: opts.small, disabled: opts.disabled, onclick: opts.onclick });
  const cmd = opts.command ? showCommand(opts.command)
    : opts.request ? liveCommand(opts.request, { watch: opts.watch, open: opts.open }) : null;
  const wrap = h("div", { class: "action" }, b, cmd);
  wrap.button = b;
  wrap.cmd = cmd;
  return wrap;
}
