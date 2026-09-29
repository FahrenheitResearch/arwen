// The server's JSON endpoints and a run's live events. Every screen talks to the server through this module.
//
// GETs ride on the HttpOnly cookie the first link set. POSTs also carry the token in the X-ArWen-Token header,
// which /api/session hands to this page (same origin only). Every POST reply carries `command`: the one
// gpuwm command it ran, exactly, which Copy takes; and `shown`, the same command with plain names, which
// "Show command" draws.

const KINDS = ["plan_accepted", "resolved_plan", "stage_started", "stage_finished", "model_progress",
  "output_committed", "first_products_ready", "live_products_ready", "fetch_started", "fetch_progress", "fetch_completed",
  "warning", "completed", "failed", "status"];

export class ApiError extends Error {
  constructor(status, body) {
    super((body && body.message) || `The server answered ${status}.`);
    this.status = status;
    this.body = body || {};
    this.fix = this.body.fix || "";
    this.command = this.body.command || null;
    this.shown = this.body.shown || null;
  }
}

async function read(response) {
  let body = null;
  try { body = await response.json(); } catch (_) { body = null; }
  if (!response.ok) throw new ApiError(response.status, body);
  return body;
}

let sessionPromise = null;

export function session(refresh = false) {
  if (!sessionPromise || refresh) {
    sessionPromise = fetch("/api/session", { credentials: "same-origin", cache: "no-store" }).then(read);
    sessionPromise.catch(() => { sessionPromise = null; });
  }
  return sessionPromise;
}

let copyPromise = null;

// the page's words, from gui/copy/*.json: copy().screens.runs.title ...
export function copy() {
  if (!copyPromise) copyPromise = get("/api/copy");
  return copyPromise;
}

export async function get(path) {
  return read(await fetch(path, { credentials: "same-origin", cache: "no-store" }));
}

export async function post(path, body = {}) {
  const s = await session();
  const response = await fetch(path, {
    method: "POST", credentials: "same-origin", cache: "no-store",
    headers: { "Content-Type": "application/json", [s.token_header]: s.token },
    body: JSON.stringify(body),
  });
  return read(response);
}

// The exact command a POST would run, with nothing run and nothing changed: what "Show command" shows.
export function dryRun(path, body = {}) {
  return post(path, { ...body, dry_run: true });
}

export const runPath = (id) => `/api/runs/${encodeURIComponent(id)}`;
export const filePath = (id, rel) => `${runPath(id)}/files/${rel.split("/").map(encodeURIComponent).join("/")}`;

// Live events of one run. handlers: {model_progress, output_committed, status, ..., any(kind, data), error()}.
// Returns a function that closes the stream. The browser reconnects on its own and the server resumes after
// the last event it sent (Last-Event-ID).
export function follow(runId, handlers) {
  const source = new EventSource(`${runPath(runId)}/events`);
  for (const kind of KINDS) {
    source.addEventListener(kind, (ev) => {
      let data;
      try { data = JSON.parse(ev.data); } catch (_) { return; }
      if (handlers[kind]) handlers[kind](data);
      if (handlers.any) handlers.any(kind, data);
    });
  }
  source.onerror = () => { if (handlers.error) handlers.error(source.readyState); };
  return () => source.close();
}
