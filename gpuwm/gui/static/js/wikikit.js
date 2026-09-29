// The wiki's parts every page shares: footnotes that open the record behind a fact, the source drawer, the fact
// table with its source column, the sources list, prose assembled from facts, categories, links and the search
// route. Nothing here states a fact of its own: every
// fact comes from the page store with the ids of its sources, and a footnote is drawn for each.

import { h, append } from "./core.js";
import * as api from "./api.js";
import { go } from "./router.js";

// ---------------------------------------------------------------- links

export const eventHref = (id) => `#/event/${encodeURIComponent(id)}`;
export const kindHref = (id) => `#/kind/${encodeURIComponent(id)}`;
export const placeHref = (id) => `#/place/${encodeURIComponent(id)}`;
export const runHref = (id) => `#/run/${encodeURIComponent(id)}`;

// #/search/WORDS?type=..&region=..: the search route carries its words and filters, so a result list is a link
export function searchHref(params = {}) {
  const q = params.q || "";
  const rest = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (k !== "q" && v !== "" && v !== null && v !== undefined) rest.set(k, v);
  const tail = rest.toString();
  return `#/search/${encodeURIComponent(q)}${tail ? `?${tail}` : ""}`;
}

// args: the route's decoded segments (the words); query: the text after its "?", still encoded (the filters).
export function parseSearch(args, query = "") {
  const params = Object.fromEntries(new URLSearchParams(query || ""));
  return { q: (args || []).join("/"), ...params };
}

export function link(href, text, cls) {
  return h("a", { href, class: cls || null }, text);
}

export function hrefFor(row) {
  if (row.kind === "event") return eventHref(row.id);
  if (row.kind === "phenomenon") return kindHref(row.id);
  if (row.kind === "place") return placeHref(row.id);
  if (row.kind === "run") return runHref(row.id);
  return "#/library";
}

// "2005-08-28T18:00Z" -> "28 Aug 2005"
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export function day(text) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(text || ""));
  return m ? `${Number(m[3])} ${MON[Number(m[2]) - 1]} ${m[1]}` : String(text || "");
}

// ---------------------------------------------------------------- citations

// One numbering per page: a source gets its number the first time a fact cites it, and every later fact that
// cites it shows the same number. Clicking a number opens the source in the drawer.
export class Cites {
  constructor(sources, words) {
    this.sources = sources || {};
    this.words = words;
    this.order = [];
  }

  number(id) {
    let i = this.order.indexOf(id);
    if (i < 0) { this.order.push(id); i = this.order.length - 1; }
    return i + 1;
  }

  mark(ids) {
    const list = [].concat(ids || []).filter(Boolean);
    if (!list.length) return null;
    return h("sup", { class: "cite" }, list.map((id) => {
      const n = this.number(id);
      const src = this.sources[id] || {};
      const b = h("button", { type: "button", title: src.title || id, "aria-label": `Source ${n}: ${src.title || id}`,
        class: src.missing ? "missing" : null }, `[${n}]`);
      b.addEventListener("click", (ev) => { ev.preventDefault(); openSource(id, this); });
      return b;
    }));
  }

  // The numbers of a fact's records, as boxed buttons for a fact table's source column.
  cell(ids) {
    const list = [].concat(ids || []).filter(Boolean);
    return list.map((id) => {
      const n = this.number(id);
      const src = this.sources[id] || {};
      const b = h("button", { type: "button", class: `srcn${src.missing ? " missing" : ""}`, title: src.title || id,
        "aria-label": `Source ${n}: ${src.title || id}` }, String(n));
      b.addEventListener("click", () => openSource(id, this));
      return b;
    });
  }

  // The numbered list of a page's sources: every record a fact on it cited, in the order first cited.
  references(full = false) {
    return h("ol", { class: `srcs${full ? " full" : ""}` }, this.order.map((id) => {
      const src = this.sources[id] || { id, missing: true };
      const b = h("button", { type: "button", class: "link", title: src.publisher || "" }, src.title || id);
      b.addEventListener("click", () => openSource(id, this));
      return h("li", { id: `ref-${this.number(id)}` }, h("span", {}, String(this.number(id))), b);
    }));
  }

  // The sources as a side panel: a head with the count, the numbered list under it.
  panel() {
    const w = this.words.wiki.article;
    const n = this.order.length;
    return h("section", { class: "panel" }, h("div", { class: "phd" }, h("b", {}, w.sources),
      h("span", {}, n === 1 ? w.records_one : `${n} ${w.records}`)), this.references());
  }
}

// ---------------------------------------------------------------- the source drawer

let drawer = null;

function closeDrawer() {
  if (drawer) { drawer.remove(); drawer = null; }
  document.removeEventListener("keydown", onEscape);
}

function onEscape(ev) { if (ev.key === "Escape") closeDrawer(); }

window.addEventListener("hashchange", closeDrawer);

function rows(obj) {
  const entries = Object.entries(obj || {});
  if (!entries.length) return null;
  return h("div", { class: "tablewrap" }, h("table", { class: "rowtable" }, h("tbody", {},
    entries.map(([k, v]) => h("tr", {}, h("td", { class: "mono" }, k),
      h("td", { class: "mono" }, typeof v === "object" ? JSON.stringify(v) : String(v)))))));
}

export function openSource(id, cites) {
  const w = cites.words.wiki.source;
  const src = cites.sources[id] || { id, missing: true };
  const kind = src.missing ? "missing" : src.kind || "record";
  const body = h("div", { class: "drawer-body" });
  const close = h("button", { type: "button", class: "btn small", "aria-label": w.close }, w.close);
  close.addEventListener("click", closeDrawer);
  const facts = [];
  const pair = (k, v) => { if (v) facts.push(h("dt", {}, k), h("dd", {}, v)); };
  pair(w.publisher, src.publisher);
  pair(w.licence, src.licence);
  pair(w.retrieved, src.retrieved);
  pair(w.citation, src.citation);
  if (src.code) pair(w.code, h("code", {}, src.code));
  if (src.path) pair(w.file, h("code", {}, `${src.run}/${src.path}`));
  const inputs = (src.inputs || []).map((i) => {
    const b = h("button", { type: "button", class: "link" }, (cites.sources[i] || {}).title || i);
    b.addEventListener("click", () => openSource(i, cites));
    return h("li", {}, b);
  });
  const dataset = src.dataset && cites.sources[src.dataset] ? (() => {
    const b = h("button", { type: "button", class: "link" }, cites.sources[src.dataset].title);
    b.addEventListener("click", () => openSource(src.dataset, cites));
    return b;
  })() : null;
  let open = null;
  if (src.kind === "run-file") {
    open = h("a", { class: "btn small", href: api.filePath(src.run, src.path), target: "_blank", rel: "noopener" }, w.open_file);
  } else if (src.url) {
    open = h("a", { class: "btn small", href: src.url, target: "_blank", rel: "noopener noreferrer" }, w.open);
  }
  // core.append flattens the section arrays and skips the empty ones; the DOM's own append would print them.
  append(body, [
    h("div", { class: "tags" }, h("span", { class: "tag a" }, w.kinds[kind] || kind)),
    h("h3", {}, src.title || id),
    src.missing ? h("p", { class: "notice warn" }, w.missing) : null,
    facts.length ? h("dl", { class: "kv" }, facts) : null,
    dataset ? h("p", { class: "note" }, `${w.inputs}: `, dataset) : null,
    src.method ? [h("div", { class: "cap sub" }, w.method), h("p", {}, src.method)] : null,
    inputs.length ? [h("div", { class: "cap sub" }, w.inputs), h("ul", { class: "plain" }, inputs)] : null,
    src.result ? [h("div", { class: "cap sub" }, w.result), rows(src.result)] : null,
    src.row ? [h("div", { class: "cap sub" }, w.row), rows(src.row)] : null,
    src.folders ? [h("div", { class: "cap sub" }, w.folders), h("ul", { class: "plain mono" }, src.folders.map((f) => h("li", {}, f)))] : null,
    src.url ? h("p", { class: "note mono url" }, src.url) : null,
    open ? h("div", { class: "btns" }, open) : null,
  ]);
  closeDrawer();
  drawer = h("aside", { class: "drawer", role: "dialog", "aria-label": w.title },
    h("div", { class: "drawer-head" }, h("b", {}, `${w.title} ${cites.number(id)}`), close), body);
  document.body.append(drawer);
  document.addEventListener("keydown", onEscape);
  close.focus();
}

// ---------------------------------------------------------------- facts and prose

export function factIndex(facts) {
  const out = new Map();
  for (const f of facts || []) out.set(f.id, f);
  return out;
}

// Prose from a page's parts: {text} is the code's joining words, {fact} is a fact of the page, drawn with its
// footnote. A part that names a fact the page lacks is dropped, so no sentence claims more than the facts hold.
export function prose(parts, facts, cites, cls = "lede") {
  const p = h("p", { class: cls });
  for (const part of parts || []) {
    if (part.fact) {
      const f = facts.get(part.fact);
      if (!f) continue;
      p.append(h("span", { class: "fact" }, f.text), cites.mark(f.cite));
    } else if (part.text) {
      p.append(part.text);
    }
  }
  return p;
}

export function generatedMark(words) {
  return h("span", { class: "gen", title: words.wiki.article.generated_help, tabindex: "0" }, words.wiki.article.generated);
}

// A fact table: the name, the value, and the numbers of the records it comes from.
export function factTable(facts, cites) {
  const rows = (facts || []).filter((f) => f.infobox !== false).map((f) =>
    h("tr", {}, h("td", { class: "k" }, f.label), h("td", {}, f.text), h("td", { class: "src" }, cites.cell(f.cite))));
  return h("div", { class: "panel" }, h("table", { class: "facts" }, h("tbody", {}, rows)));
}

// A section of an article: a heading (with a quiet note on its right), then its parts.
export function section(title, ...kids) {
  let note = null;
  if (Array.isArray(title)) [title, note] = title;
  return h("section", { class: "wsec" }, h("h2", { class: "sec" }, title, note ? h("span", {}, note) : null), kids);
}

// A map in a panel: the canvas, then a line with the legend on the left and a note on the right.
export function mapPanel(canvas, legend, note) {
  return h("section", { class: "panel" }, canvas, h("div", { class: "pft" }, h("div", { class: "mleg" }, legend), note ? h("b", {}, note) : null));
}

export function categories(cats, words) {
  if (!cats || !cats.length) return null;
  return h("div", { class: "catbar" }, h("b", {}, words.wiki.article.categories),
    cats.map((c) => {
      const { label, ...filters } = c;
      return h("a", { class: "tag", href: searchHref(filters) }, label);
    }));
}

// The share of its place's events at least as strong, so "top 0.2%" outranks "top 33%" whatever the counts.
export function rarityText(pct) {
  if (pct === null || pct === undefined) return null;
  if (pct < 0.1) return "top 0.1%";
  if (pct < 10) return `top ${Number(pct.toFixed(1))}%`;
  return `top ${Math.round(pct)}%`;
}

export function rarityChip(row) {
  const pct = row.rarity_pct;
  if (pct === null || pct === undefined) return null;
  const cls = pct <= 1 ? "tag rare hi" : "tag rare";
  const of = Number(row.rarity_of || 0).toLocaleString("en-US");
  return h("span", { class: cls, title: `${row.rarity} of ${of}. ${row.rarity_text || ""}`.trim() }, rarityText(pct));
}

// A picture the Rust renderer wrote, shown as it is.
export function picture(runId, rel, alt) {
  return h("img", { src: api.filePath(runId, rel), alt: alt || "", loading: "lazy" });
}

export function searchBox(words, value = "") {
  const w = words.wiki.search;
  const input = h("input", { class: "input", type: "search", placeholder: w.placeholder, value,
    "aria-label": w.title, spellcheck: "false" });
  const form = h("form", { class: "bigsearch", role: "search" }, input,
    h("button", { class: "btn primary big", type: "submit" }, w.button));
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    go(searchHref({ q: input.value.trim() }).slice(2));
  });
  form.input = input;
  return form;
}
