// Small DOM and number helpers every screen shares. No framework, no network.

export const MINUS = "−";

// h("div", {class: "x", onclick: fn}, child, "text", [more children])
export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  append(el, kids);
  return el;
}

export function append(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

// Puts `kids` in `el` in order and moves only what has to move. A child kept from the last draw stays where it is in
// the document, so a box someone is typing in keeps the keyboard and its text through a redraw; a new child goes in
// its place, and a child no longer listed leaves.
export function place(el, kids) {
  const want = kids.flat(Infinity).filter((kid) => kid !== null && kid !== undefined && kid !== false)
    .map((kid) => (kid instanceof Node ? kid : document.createTextNode(String(kid))));
  const keep = new Set(want);
  for (const old of [...el.childNodes]) if (!keep.has(old)) old.remove();
  want.forEach((kid, i) => {
    const at = el.childNodes[i] || null;
    if (at !== kid) el.insertBefore(kid, at);
  });
  return el;
}

// a number with a true minus sign and thousands separators
export function num(x, nd = 0) {
  if (x === null || x === undefined || Number.isNaN(x)) return "";
  const s = Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: nd, maximumFractionDigits: nd });
  return (x < 0 && Number(s.replace(/,/g, "")) !== 0 ? MINUS : "") + s;
}

// seconds as "2 h 05 min" or "4 min 10 s"
export function duration(seconds) {
  if (seconds === null || seconds === undefined) return "";
  const s = Math.max(0, Math.round(seconds));
  const hrs = Math.floor(s / 3600);
  const min = Math.floor((s % 3600) / 60);
  if (hrs > 0) return `${hrs} h ${String(min).padStart(2, "0")} min`;
  if (min > 0) return `${min} min ${String(s % 60).padStart(2, "0")} s`;
  return `${s} s`;
}

export function panel(cap, ...kids) {
  return h("section", { class: "panel" }, cap ? h("h2", { class: "cap" }, cap) : null, kids);
}

// a state as a dot and short mono words; colour only where something is live or wrong
export function chip(state, label) {
  return h("span", { class: `st ${state}` }, label || state);
}

export function button(label, opts = {}) {
  const cls = ["btn", opts.kind || "", opts.small ? "small" : ""].join(" ").trim();
  const b = h("button", { class: cls, type: "button", disabled: opts.disabled || null, title: opts.title || null }, label);
  if (opts.onclick) b.addEventListener("click", opts.onclick);
  return b;
}

export function table(headers, rows) {
  const thead = h("thead", {}, h("tr", {}, headers.map((c) => h("th", {}, c))));
  return h("div", { class: "tablewrap" }, h("table", {}, thead, h("tbody", {}, rows)));
}

export function kv(pairs) {
  return h("dl", { class: "kv" }, pairs.filter(Boolean).map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
}

export function readout(k, v, sub) {
  return h("div", { class: "readout" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v),
    sub ? h("div", { class: "s" }, sub) : null);
}

// a line of text in a code box with a Copy button (clipboard only; nothing leaves the page)
// A line with a Copy button. copyText, when given, is what Copy takes: a command reads with plain names on screen
// and copies as the exact line.
export function copyline(text, label, copyText = null) {
  const b = button("Copy", { small: true });
  b.addEventListener("click", () => {
    if (navigator.clipboard) navigator.clipboard.writeText(copyText || text).catch(() => {});
    b.textContent = "Copied";
    setTimeout(() => { b.textContent = "Copy"; }, 1200);
  });
  return h("div", { class: "copyline" }, h("code", { title: label || null }, text), b);
}

export function bar(percent, live = false) {
  const p = Math.max(0, Math.min(100, percent || 0));
  // Widths go through the style object: the page's policy refuses style attributes.
  const fillEl = h("i", {});
  fillEl.style.width = `${p.toFixed(1)}%`;
  return h("div", { class: `meter${live ? " live" : ""}`, role: "progressbar", "aria-valuenow": String(p), "aria-valuemin": "0",
    "aria-valuemax": "100" }, fillEl);
}

// "{name}" fields of a copy line filled from values
export function fill(text, values) {
  return String(text || "").replace(/\{([a-z_]+)\}/g, (m, key) => (key in values ? String(values[key]) : m));
}

// a fold: a summary line that opens a body ("More settings", "Show folder")
export function fold(summary, ...kids) {
  return h("details", { class: "fold" }, h("summary", {}, summary), h("div", { class: "body" }, kids));
}

// "0.75" -> "750 m", "3" -> "3 km"
// A nested run's grids and the one whose steps take the most wall: "12 / 3 / 1 km, 21x real time, the 1 km grid
// sets the pace". words holds pace, pace_no_speed, grids_speed and grids_plain; "" for a one-grid run.
export function paceLine(st, words) {
  const g = st && st.grids;
  if (!g || !g.grids_km || g.grids_km.length < 2) return "";
  const speed = st.speed_x ? `${Math.round(st.speed_x)}x` : null;
  const text = g.pace_label ? (speed ? words.pace : words.pace_no_speed) : (speed ? words.grids_speed : words.grids_plain);
  return fill(text, { grids: g.grids_label, speed: speed || "", pace: g.pace_label || "" });
}

// "1.2 GB", "340 MB", "12 kB": bytes as a download line says them
export function bytesWords(n) {
  const v = Number(n) || 0;
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)} GB`;
  if (v >= 1e6) return `${Math.round(v / 1e6)} MB`;
  return `${Math.max(1, Math.round(v / 1e3))} kB`;
}

// "40 s", "3 min 20 s", "1 h 05 min": how long a preparation step has been going, to the second below an hour so
// the line visibly moves while a step that says nothing finer is still working
function sinceWords(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  if (s >= 3600) return `${Math.floor(s / 3600)} h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")} min`;
  if (s >= 60) return `${Math.floor(s / 60)} min ${s % 60} s`;
  return `${s} s`;
}

// Where a running forecast's preparation is, as the phrases of its live line: the step and the grid it is on, the
// download's files, times and bytes, GPU kernels compiling, the boundary times a chained forecast has ready, and
// how long the step it names (or, with no step named, the stage) has been going. st.preparation is the page server's reading of the engine's own records
// (gui/runs.py preparation()); words is the copy's "preparation" block. [] when the forecast has nothing left to
// prepare.
export function prepLine(st, words) {
  const p = st && st.preparation;
  if (!p || !words) return [];
  const out = [];
  const known = (v) => v !== null && v !== undefined;
  const f = p.fetch;
  if (p.stage === "fetch" && f) {
    if (f.times_total) out.push(fill(words.times, { done: f.times_done || 0, total: f.times_total }));
    else if (f.files_total) out.push(fill(words.files, { done: f.files_done || 0, total: f.files_total }));
    else if (known(f.files_done)) out.push(fill(words.files_open, { done: f.files_done }));
    if (f.bytes) out.push(f.bytes_total ? fill(words.bytes, { moved: bytesWords(f.bytes), total: bytesWords(f.bytes_total) })
      : fill(words.bytes_open, { moved: bytesWords(f.bytes) }));
  }
  if (p.stage === "prepare" || p.stage === "initialize") {
    const step = p.step;
    const name = step ? (words.steps[step.key] || step.label || "") : "";
    if (step && known(step.done) && step.count) out.push(fill(words.counted, { step: name, done: step.done, count: step.count }));
    else if (step && step.key === "domain_initialize" && known(step.index) && step.count > 1) {
      out.push(fill(step.grid_km ? words.grid : words.grid_plain,
        { step: name, n: step.index, count: step.count, km: spacing(step.grid_km) }));
    } else if (step && step.phase_number && step.phases) out.push(fill(words.numbered, { step: name, n: step.phase_number, count: step.phases }));
    else if (name) out.push(name);
    else if (p.phase && words.phases[p.phase]) out.push(words.phases[p.phase]);
    // A step's own time right after the step, before "compiling GPU kernels", which it does not time.
    if (step && known(p.step_seconds)) out.push(fill(words.for, { time: sinceWords(p.step_seconds) }));
    if (p.compiling) out.push(words.compiling);
  }
  if (p.stage === "forecast" && p.compiling) out.push(words.compiling);
  if (p.chained && p.boundaries && p.stage === "forecast") out.push(fill(words.boundaries, p.boundaries));
  // A time follows what it times: a step's own after that step (above), and the stage's after the stage's own
  // figures (a download's) or, when the line names a step or phase inside the stage, right after the stage's name
  // (first here), so a grid begun a second ago never reads as building for the stage's minutes.
  const inside = (p.stage === "prepare" || p.stage === "initialize") && (p.step || p.phase);
  const stepTimed = inside && p.step && known(p.step_seconds);
  if (!stepTimed && p.stage !== "forecast" && known(p.stage_seconds)) {
    const said = fill(words.for, { time: sinceWords(p.stage_seconds) });
    if (inside && out.length) out.unshift(said); else out.push(said);
  }
  return out;
}

export function spacing(km) {
  if (km === null || km === undefined || Number.isNaN(Number(km))) return "";
  const v = Number(km);
  return v < 1 ? `${Math.round(v * 1000)} m` : `${+v.toFixed(2)} km`;
}
