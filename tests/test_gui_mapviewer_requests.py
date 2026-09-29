"""The map viewer draws the map that is selected, never a late picture of another one.

A picture of one map (temperature) can still be loading when the user picks another (radar)
at the same valid time.  The radar picture lands first; the temperature one lands later.  The
map must keep radar: a request belongs to the map and the showing that asked for it, not only
to the valid time, and a late answer to an old request is dropped.

The real mapviewer.js runs in Node with its imports replaced by small stand-ins that record
what the map draws.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

JS = Path(__file__).resolve().parents[1] / "gpuwm" / "gui" / "static" / "js"

DOM = r"""
const made = [];
class El {
  constructor(tag) {
    this.tag = tag; this.kids = []; this.on = {}; this.hidden = false; this.style = { setProperty() {} };
    this.classList = { add() {}, remove() {}, toggle() {}, contains() { return false; } };
    this.dataset = {}; this.textContent = ""; this.parentElement = null; this.scrollWidth = 0; this.clientWidth = 0;
  }
  get children() { return this.kids; }
  append(...k) { for (const x of k) if (x && typeof x === "object") { x.parentElement = this; this.kids.push(x); } }
  replaceChildren(...k) { this.kids = []; this.append(...k); }
  addEventListener(ev, fn) { (this.on[ev] = this.on[ev] || []).push(fn); }
  removeEventListener() {}
  click() { for (const fn of this.on.click || []) fn({ stopPropagation() {}, target: this }); }
  getBoundingClientRect() { return { left: 0, right: 0, top: 0, bottom: 0, width: 0, height: 0 }; }
  querySelector() { return null; }
  contains() { return false; }
  focus() {}
  remove() {}
  setPointerCapture() {}
  scrollIntoView() {}
}
globalThis.__made = made;
globalThis.__El = El;
globalThis.document = { body: new El("body"), documentElement: new El("html"), addEventListener() {}, removeEventListener() {},
  createElement: (t) => new El(t) };
globalThis.window = { addEventListener() {}, removeEventListener() {}, confirm: () => true };
globalThis.history = { replaceState() {} };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
"""

STUBS = {
    "core.js": r"""
export function h(tag, attrs, ...kids) {
  const el = new globalThis.__El(tag);
  if (attrs && attrs.dataset) el.dataset = attrs.dataset;
  if (attrs && attrs.hidden) el.hidden = true;
  el.textContent = kids.filter((k) => typeof k === "string").join("");
  el.append(...kids.filter((k) => k && typeof k === "object"));
  globalThis.__made.push(el);
  return el;
}
export function append(el, kids) { el.append(...kids.filter(Boolean)); }
export function bar() { return h("div"); }
export function fill(text) { return String(text); }
export function paceLine() { return ""; }
export function prepLine() { return []; }
""",
    "api.js": r"""
export const runPath = (id) => `/api/runs/${id}`;
export const filePath = (id, rel) => `${runPath(id)}/files/${rel}`;
export async function get(path) {
  const FIX = globalThis.__fixture;
  if (path.endsWith("/map")) return { projection: {}, domains: [], moves: [] };
  if (path.endsWith("/pictures")) return FIX.index;
  const m = /pictures\/list\?product=(.*)$/.exec(path);
  if (m) return FIX.lists[decodeURIComponent(m[1])];
  return { name: "run", status: { state: "complete" } };
}
export async function post() { return {}; }
export function follow() { return () => {}; }
""",
    "router.js": "export function notice() {}\nexport function errorText(e) { return String(e); }\n",
    "geo.js": r"""
export function grids() { return new Map(); }
export function gridOutline() { return []; }
export function domainNumber(token) { const m = /^d(\d+)/.exec(token || ""); return m ? Number(m[1]) : null; }
export function placesAt() { return new Map(); }
export function gridForPicture(g) { return g; }
export function borrowGeoref() { return null; }
export function unwrapLon(lon) { return lon; }
export function lonLatBox() { return null; }
export function boundsBox() { return null; }
export function screenRings() { return []; }
export function wrapLon(lon) { return lon; }
""",
    "field.js": r"""
// a picture's load takes the time the fixture gives its path
export function placePicture(url) {
  const ms = globalThis.__fixture.delay[url.split("/files/")[1]] || 5;
  return new Promise((ok) => setTimeout(() => ok({ bent: true, legend: null, url }), ms));
}
export function drawPlaced(ctx, map, placed) { globalThis.__drawn.push(placed.url.split("/files/")[1]); }
""",
    "looks.js": "export function productName(p) { return p; }\nexport function groupProducts() { return []; }\n",
    "time.js": r"""
export function parseTime(t) { const d = t ? new Date(t) : null; return d && !isNaN(d) ? d : null; }
export function longTime(d) { return String(d); }
export function leftWords() { return ""; }
export function hourWords() { return ""; }
""",
    "command.js": "export function actionButton() { return { button: {} }; }\n",
    # The viewer offers Downscale from a finished run's map; this test never opens it.
    "downscale.js": "export async function openDownscale() { return { close() {} }; }\n",
}

SCRIPT = r"""
import "./dom.mjs";
const VALID = "2026-09-24T18:00:00Z";
globalThis.__drawn = [];
globalThis.__fixture = {
  index: { domains: ["d01"], products: ["T2", "REFL"], favourites: ["T2", "REFL"], by_domain: {}, count: 2 },
  lists: {
    T2: { georefs: [], pictures: [{ domain: "d01", valid: VALID, path: "T2/d01.png", geo: null }] },
    REFL: { georefs: [], pictures: [{ domain: "d01", valid: VALID, path: "REFL/d01.png", geo: null }] },
  },
  // the temperature picture answers three tenths of a second late
  delay: { "T2/d01.png": 300, "REFL/d01.png": 5 },
};
const layers = [];
const map = {
  width: 800, height: 600, inset: {},
  add(layer) { layers.push(layer); return () => {}; },
  on() { return () => {}; },
  redraw() { for (const l of layers) if (!l.above) l.draw({}); globalThis.__drawn.push("|"); },
  fit() {}, zoomAt() {}, screen() { return [0, 0]; },
};
const words = new Proxy({}, { get: (_, k) => (k === "states" || k === "stages" ? {} : new Proxy({}, { get: (_, n) => String(n) })) });
const { openViewer } = await import("./mapviewer.mjs");
const opened = openViewer({ map, w: words, stage: new globalThis.__El("div"), looks: {}, links: null }, "run1");
await new Promise((ok) => setTimeout(ok, 50));
// the user picks radar while temperature is still loading
const chip = globalThis.__made.find((el) => el.tag === "button" && el.textContent === "REFL");
chip.click();
await opened;
await new Promise((ok) => setTimeout(ok, 600));
// a later redraw (a pan, a hover) shows whatever the map holds now
map.redraw();
// the frames the map drew, each ended by "|": the last one is what the user is looking at
const frames = globalThis.__drawn.join(",").split("|").map((f) => f.split(",").filter(Boolean));
const last = [...frames].reverse().find((f) => f.length) || [];
console.log(JSON.stringify({ last, frames }));
"""


@pytest.fixture(scope="module")
def viewed(tmp_path_factory):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    folder = tmp_path_factory.mktemp("mapviewer")
    source = (JS / "mapviewer.js").read_text(encoding="utf-8")
    for name in STUBS:
        source = source.replace(f'from "./{name}"', f'from "./{name[:-3]}.mjs"')
        (folder / f"{name[:-3]}.mjs").write_text(STUBS[name], encoding="utf-8")
    (folder / "mapviewer.mjs").write_text(source, encoding="utf-8")
    (folder / "dom.mjs").write_text(DOM, encoding="utf-8")
    (folder / "t.mjs").write_text(SCRIPT, encoding="utf-8")
    done = subprocess.run([node, "t.mjs"], cwd=folder, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_late_picture_of_another_map_never_replaces_the_selected_one(viewed):
    assert viewed["last"] == ["REFL/d01.png"], viewed["frames"]


def test_the_selected_map_is_drawn(viewed):
    assert any(f == ["REFL/d01.png"] for f in viewed["frames"]), viewed["frames"]
