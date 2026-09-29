"""The run map in a real browser: a picture that lands late never replaces the map picked since.

The defect: the map viewer kept pending pictures by valid time only, so a
temperature picture that finished loading after radar was picked was drawn
over radar while the picker said radar. Here the shipped viewer modules run
in headless Chromium against a fixture server holding the temperature file
back; every picture drawn after the switch must be radar's.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse, parse_qs

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")

GUI = Path(__file__).resolve().parents[1] / "gpuwm" / "gui"
#: The fixture server binds in this range: the other ports on the test machines belong to other services.
PORTS = range(8770, 8800)
HELD_S = 1.0
#: Set when the page has asked for the temperature picture, which the server then holds back.
ASKED = threading.Event()
GEOREF = {"projection": {"kind": "geographic", "central_meridian_deg": 0},
          "plot_rect_px": {"x": 0, "y": 0, "width": 200, "height": 200},
          "extent": {"x_min": -100, "x_max": -80, "y_min": 30, "y_max": 50},
          "geographic_bounds": [-100, -80, 30, 50]}
PAGE = """<!doctype html><html><head><link rel="stylesheet" href="/static/css/app.css"></head>
<body><div id="notices"></div><main id="stage" class="mv" style="height:800px;position:relative"></main>
<script type="module">
import { setNoticeBox } from '/static/js/router.js';
import { MapView } from '/static/js/geomap.js';
import { openViewer } from '/static/js/mapviewer.js';
const words = await (await fetch('/words')).json();
setNoticeBox(document.getElementById('notices'));
const stage = document.getElementById('stage');
const map = new MapView(stage);
window.viewerMap = map; window.fieldDraws = [];
const draw = map.ctx.drawImage.bind(map.ctx);
map.ctx.drawImage = (image, ...rest) => {
  if (image instanceof HTMLCanvasElement) {
    const px = image.getContext('2d').getImageData(Math.floor(image.width / 2), Math.floor(image.height / 2), 1, 1).data;
    window.fieldDraws.push([px[0], px[1], px[2]]);
  }
  return draw(image, ...rest);
};
// Not awaited: the viewer draws its first map while the test picks another.
window.opened = openViewer({ map, w: words.screens, looks: words.looks, stage }, 'late');
window.ready = true;
</script></body></html>"""


def _svg(colour: str) -> bytes:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">'
            f'<rect width="200" height="200" fill="{colour}"/></svg>').encode()


class Fixture(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401 - quiet
        pass

    def _send(self, body: bytes, kind: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, document) -> None:
        self._send(json.dumps(document).encode(), "application/json")

    def do_GET(self):  # noqa: N802 - the handler's name
        url = urlparse(self.path)
        path = url.path
        if path == "/words":
            return self._json({name: json.loads((GUI / "copy" / f"{name}.json").read_text(encoding="utf-8"))
                               for name in ("screens", "looks", "wiki", "assistant")})
        if path == "/api/session":
            return self._json({"token": "fixture", "token_header": "X-ArWen-Token"})
        if path == "/api/runs/late":
            return self._json({"id": "late", "name": "late", "status": {"state": "finished"}})
        if path == "/api/runs/late/map":
            return self._json({"domains": [], "region": [[-100, 30], [-80, 30], [-80, 50], [-100, 50], [-100, 30]]})
        if path == "/api/runs/late/pictures":
            return self._json({"count": 2, "domains": ["d01"],
                               "products": ["2m_temperature", "composite_reflectivity"],
                               "favourites": ["2m_temperature", "composite_reflectivity"]})
        if path == "/api/runs/late/pictures/list":
            product = parse_qs(url.query).get("product", [""])[0]
            return self._json({"georefs": [GEOREF], "pictures": [
                {"name": "field", "path": f"{product}.svg", "domain": "d01",
                 "valid": "2026-09-24T12:00:00Z", "geo": 0}]})
        if path.startswith("/api/runs/late/files/"):
            temperature = "2m_temperature" in path
            if temperature:
                ASKED.set()
                time.sleep(HELD_S)
            return self._send(_svg("#ff0000" if temperature else "#00ff00"), "image/svg+xml")
        if path.startswith("/static/"):
            file = GUI / path.lstrip("/")
            if file.is_file():
                kind = "text/javascript" if file.suffix == ".js" else "text/css" if file.suffix == ".css" else \
                    "application/octet-stream"
                return self._send(file.read_bytes(), kind)
            self.send_response(404)
            self.end_headers()
            return None
        return self._send(PAGE.encode(), "text/html")


@pytest.fixture(scope="module")
def fixture_server():
    server = None
    for port in PORTS:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Fixture)
            break
        except OSError:
            continue
    if server is None:
        pytest.skip("No free port in 8770-8799")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_a_late_picture_of_another_map_is_never_drawn_over_the_map_picked(fixture_server):
    with playwright_api.sync_playwright() as p:
        browser = None
        for channel in ("chrome", "msedge", None):
            try:
                browser = p.chromium.launch(channel=channel, headless=True) if channel else p.chromium.launch(headless=True)
                break
            except Exception:  # noqa: BLE001 - try the next browser this computer has
                continue
        if browser is None:
            pytest.skip("No Chromium-family browser is installed")
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda err: errors.append(str(err)))
            page.goto(fixture_server + "/")
            page.wait_for_function("() => window.ready === true", timeout=20000)
            # The first map (temperature) is still loading, held back by the server; radar is picked meanwhile.
            assert ASKED.wait(10), "the page never asked for the first map's picture"
            # The chip's own click handler, as a tap on it runs it: the top bar may be laid out with the chip
            # folded away until the first map has drawn, and the pick must land while that map is still held.
            page.locator(".quick button").nth(1).dispatch_event("click")
            assert page.evaluate("window.fieldDraws.length") == 0, "the held picture was drawn before the pick"
            page.wait_for_timeout(int((HELD_S + 0.8) * 1000))
            draws = page.evaluate("window.fieldDraws")
            picked = page.locator(".picker-btn").inner_text()
            centre = page.evaluate("""() => { const m = window.viewerMap;
                return [...m.ctx.getImageData(Math.round(m.width * m.dpr / 2), Math.round(m.height * m.dpr / 2), 1, 1).data]; }""")
        finally:
            browser.close()
    assert not errors, errors
    assert draws, "the viewer drew no picture"
    assert all(g > 200 and r < 80 for r, g, _b in draws), draws  # radar's green only, never temperature's red
    assert "eflectivity" in picked or "adar" in picked, picked
    assert centre[1] > 150 and centre[0] < 120, centre
