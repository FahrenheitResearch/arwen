// A forecast's map: the renderer's pictures placed at their true place on the basemap, the time bar across the
// bottom with Play and the arrow keys, the map picker and see-through floating over it. The same page watches a
// forecast live while it runs: new frames arrive as the renderer writes them and the time bar grows.

import { h } from "./core.js";
import * as api from "./api.js";
import { register, runContext, runCrumbs, runRoute, notice } from "./router.js";
import { MapView, loadBasemap } from "./geomap.js";
import { openViewer } from "./mapviewer.js";

async function render(body, args, page, route) {
  const words = page.words;
  const w = words.screens;
  const runId = args[0];
  if (!runId) { location.hash = "#/runs"; return null; }
  const detail = await api.get(api.runPath(runId));
  const title = detail.title || detail.name || runId;
  page.setCrumbs(runCrumbs(words, runId, title, w.shell.map));
  page.setContext(runContext(words, runId, title, route));
  document.title = `${title} · ${w.app.name}`;

  const mapEl = h("div", { class: "mapel" });
  const stage = h("div", { class: "mv" }, mapEl);
  body.append(stage);
  const map = new MapView(mapEl);
  map.inset = { left: 0, right: 0, top: 56, bottom: 72 };
  loadBasemap("/static/map/basemap.bin")
    .then((layers) => { map.basemap = layers; map.redraw(); })
    .catch(() => notice(w.app.map_failed, "warn"));
  const links = h("div", { class: "mvlinks" },
    h("a", { href: `#/${runRoute("run", runId)}` }, w.mapviewer.article),
    h("a", { href: `#/${runRoute("explore", runId)}` }, w.mapviewer.files));
  const app = { map, w, stage, looks: words.looks || {}, links, onchange: () => page.refreshSide() };
  const close = await openViewer(app, runId, args[1] || null, route);
  return () => { close(); map.close(); };
}

register("results", (body, args, page) => render(body, args, page, "results"), { bleed: true });
register("watch", (body, args, page) => render(body, args, page, "watch"), { bleed: true });
