# SPDX-License-Identifier: Apache-2.0
"""One whole browser journey, four screenshots, always stop the owned server."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import threading
import urllib.request

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path, required=True)
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    if args.port == 8080:
        parser.error("Port 8080 is excluded")
    args.screenshots.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((ROOT / "catalog.json").read_text())
    # Refuse to borrow an existing server, including one owned by another lane.
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    env = dict(os.environ, GRADIO_SERVER_NAME="127.0.0.1", GRADIO_SERVER_PORT=str(args.port),
               GRADIO_ANALYTICS_ENABLED="False", HF_HUB_OFFLINE="1")
    output = args.screenshots / "server.log"
    errors = []
    started = time.perf_counter()
    pictures = []
    with output.open("w") as log:
        server = subprocess.Popen([sys.executable, str(ROOT / "app.py")], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        stopped = threading.Event()
        def monitor():
            with (args.screenshots / "cpu.log").open("w") as cpu:
                while not stopped.wait(0.5):
                    subprocess.run(["ps", "-o", "pid,ppid,pcpu,nlwp,args", "-p", str(server.pid)], stdout=cpu)
                    cpu.flush()
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + 45
            while True:
                if server.poll() is not None:
                    raise RuntimeError(output.read_text())
                try:
                    with urllib.request.urlopen(url, timeout=1) as response:
                        assert response.status == 200
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError("Local server did not become ready")
                    time.sleep(0.2)
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
                page = browser.new_page(viewport={"width": 1460, "height": 1120}, device_scale_factor=1)
                page.on("pageerror", lambda error: errors.append(str(error)))
                external = []
                page.route("**/*", lambda route: route.continue_() if route.request.url.startswith(url) or route.request.url.startswith(("data:", "blob:")) else (external.append(route.request.url), route.abort())[-1])
                page.goto(url, wait_until="networkidle")

                def ready_image():
                    images = page.locator('img').filter(visible=True)
                    expect(images.last).to_be_visible()
                    page.wait_for_function("""() => [...document.images].filter(i => i.getBoundingClientRect().width > 200).some(i => i.complete && i.naturalWidth >= 1000)""")
                    return images.last

                def snap(name):
                    ready_image() if name != "run-your-own" else None
                    page.screenshot(path=str(args.screenshots / f"{name}.png"), full_page=True, animations="disabled")
                    pictures.append(name)

                def map_caption(sample, row):
                    text = f'{sample["label"]} · {row["variable"]} · {row["lead"]} · Valid {row["valid"]}'
                    expect(page.get_by_text(text, exact=True)).to_be_visible()

                for index, sample in enumerate(catalog["samples"]):
                    page.get_by_role("radio", name=sample["label"], exact=True).check()
                    map_caption(sample, sample["maps"][0])
                    # Exercise both variable and lead selection; the screenshot shows temperature at the last lead.
                    variables = list(dict.fromkeys(row["variable"] for row in sample["maps"]))
                    page.get_by_role("radio", name=variables[-1], exact=True).check()
                    expected = [row for row in sample["maps"] if row["variable"] == variables[-1]][0]
                    map_caption(sample, expected)
                    last_lead = sample["maps"][-1]["lead"]
                    page.get_by_role("radio", name=last_lead, exact=True).check()
                    map_caption(sample, sample["maps"][-1])
                    page.get_by_role("radio", name=variables[0], exact=True).check()
                    # Variable changes reset lead. Pick the final lead again.
                    map_caption(sample, sample["maps"][0])
                    page.get_by_role("radio", name=last_lead, exact=True).check()
                    image = ready_image()
                    expected_row = next(row for row in sample["maps"] if row["variable"] == variables[0] and row["lead"] == last_lead)
                    map_caption(sample, expected_row)
                    expect(image).to_have_attribute("src", re.compile(re.escape(Path(expected_row["image"]).name)))
                    page.wait_for_function("src => [...document.images].some(i => i.src === src && i.complete && i.naturalWidth >= 1000)", arg=image.get_attribute("src"))
                    response = page.request.get(image.get_attribute("src"))
                    assert response.ok
                    actual_hash = hashlib.sha256(response.body()).hexdigest()
                    expected_hash = hashlib.sha256((ROOT / expected_row["image"]).read_bytes()).hexdigest()
                    assert actual_hash == expected_hash, "Displayed map bytes differ from catalog image"
                    snap("hex-map" if index == 0 else "arwen-map")
                page.get_by_role("tab", name="Soundings", exact=True).click()
                points = list(dict.fromkeys(row["point"] for row in catalog["soundings"]))
                page.get_by_role("radio", name=points[-1], exact=True).check()
                last_sound = [row for row in catalog["soundings"] if row["point"] == points[-1]][-1]
                expect(page.get_by_text(points[-1], exact=False).last).to_be_visible()
                # Map and sounding lead labels may match, so scope to the active tab panel.
                panel = page.get_by_role("tabpanel").filter(visible=True)
                first_sound = next(row for row in catalog["soundings"] if row["point"] == points[-1])
                expect(panel.get_by_text(f'{first_sound["sample_label"]} · {first_sound["point"]} · {first_sound["lead"]} · Valid {first_sound["valid"]}', exact=True)).to_be_visible()
                panel.get_by_role("radio", name=last_sound["lead"], exact=True).check()
                expect(panel.get_by_text(f'Valid {last_sound["valid"]}', exact=False)).to_be_visible()
                sound_image = ready_image()
                expect(sound_image).to_have_attribute("src", re.compile(re.escape(Path(last_sound["image"]).name)))
                page.wait_for_function("src => [...document.images].some(i => i.src === src && i.complete && i.naturalWidth >= 1000)", arg=sound_image.get_attribute("src"))
                snap("sounding")
                page.get_by_role("tab", name="Run your own", exact=True).click()
                expect(page.get_by_text("gpuwm go configs/myarea.toml", exact=True)).to_be_visible()
                snap("run-your-own")
                assert not errors, errors
                browser.close()
                result = {"screenshots": len(pictures), "map_samples": len(catalog["samples"]),
                          "variables_per_sample": len(variables), "leads_per_sample": len(set(row["lead"] for row in catalog["samples"][0]["maps"])),
                          "sounding_points": len(points), "page_errors": len(errors),
                          "external_requests_blocked": len(external), "port": args.port,
                          "bind": "127.0.0.1", "seconds": round(time.perf_counter() - started, 3)}
                (args.screenshots / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result))
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            stopped.set()
            watcher.join(timeout=2)
            print(f"owned_server_stopped pid={server.pid} returncode={server.returncode}")


if __name__ == "__main__":
    main()
