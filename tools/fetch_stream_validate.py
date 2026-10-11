#!/usr/bin/env python3
"""49-object HTTP transport check. This is a replay, never a C3 forecast bench.

The HTTP origin emits valid envelopes with zero bodies, not weather fields.
Every object goes through the real rw_fetch object CLI. Independent hashlib
reads are verification after fetch end and are excluded from fetch wall.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from gpuwm import fetch, fetch_pool, rustwx_fetch
from gpuwm.timeline.ledger import emit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--mib", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    length = args.mib * 1024 * 1024 + 37
    data = bytearray(length)
    data[:4] = b"GRIB"; data[7] = 2
    data[8:16] = length.to_bytes(8, "big"); data[-4:] = b"7777"
    payload = bytes(data)
    landed = {}
    lock = threading.Lock()

    class Origin(BaseHTTPRequestHandler):
        def log_message(self, *ignored):
            pass
        def do_GET(self):
            raw = self.headers.get("Range")
            start, end = 0, len(payload)-1
            if raw:
                first, last = raw.removeprefix("bytes=").split("-")
                start, end = int(first), min(int(last) if last else end, end)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(end-start+1))
            self.end_headers()
            # Keep the last object late enough to expose overlap of bindings.
            if self.path.endswith("048") and end > 0:
                time.sleep(0.05)
            self.wfile.write(payload[start:end+1])
            if end == len(payload)-1:
                with lock:
                    landed[self.path] = time.time()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    results = []
    try:
        for row in ("constrained", "balanced", "wide"):
            os.environ["GPUWM_FETCH_ROW"] = row
            folder = args.out / row
            folder.mkdir()
            os.environ["GPUWM_TIME_LEDGER"] = str(folder / "ledger.jsonl")
            started = time.time()
            landed.clear()
            bound_at = None
            entries = []
            def complete(index, entry):
                nonlocal bound_at
                if bound_at is None:
                    # Bind planned inputs when the first verified file lands.
                    fetch._atomic_write_text(folder / "inputs.txt", "".join(
                        f"{folder / f'input-{i:03d}.grib2'}\n" for i in range(49)))
                    fetch._atomic_write_text(folder / "supplements.txt",
                        f"surface={folder / 'input-000.grib2'}\n")
                    bound_at = time.time()
            emit("worker.claim", attrs={"basis": "PROXY transport validation"})
            emit("fetch.request", "begin", attrs={"worker_row": row})
            streams = fetch_pool.chunk_streams_per_file(None, files=49)
            def transfer(index):
                path = folder / f"input-{index:03d}.grib2"
                url = f"http://127.0.0.1:{server.server_port}/input-{index:03d}"
                env = dict(os.environ, RUSTWX_DOWNLOAD_STREAMS=str(streams))
                entry = rustwx_fetch._run([str(args.binary), "object", "--url", url,
                    "--output", str(path)], what="object", out=folder, env=env)
                fetch._remember_stream_facts(path, entry)
                return {"name": path.name, **entry}
            with fetch_pool.completion_hook(complete):
                entries, receipt = fetch_pool.run_transfers([
                    fetch_pool.TransferJob(f"input-{i:03d}",
                        f"http://127.0.0.1:{server.server_port}",
                        lambda i=i: transfer(i)) for i in range(49)], workers=None)
            fetch._atomic_write_text(folder / "fetch-manifest.json", json.dumps(
                {"files": entries, "concurrency": receipt}, indent=2))
            emit("fetch.request", "end", attrs={"worker_row": row})
            ended = time.time()
            emit("finish", attrs={"basis": "PROXY transport validation"})
            rehash = {entry["name"]: fetch.sha256_file(folder / entry["name"]) for entry in entries}
            identity = all(rehash[e["name"]] == e["sha256"] for e in entries)
            network_end = max(landed.values())
            last_receipt = max(e["last_byte_unix"] for e in entries)
            deleted = []
            for entry in entries:
                path = folder / entry["name"]
                deleted.append({"path": str(path), "bytes": path.stat().st_size})
                path.unlink()
            result = {"scope": "49-object local HTTP transport validation, not C3",
                "worker_row": row, "workers": fetch_pool.resolve_file_workers(None),
                "stream_budget": fetch_pool.worker_profile()["stream_budget"],
                "files": len(entries), "bytes": sum(e["bytes"] for e in entries),
                "fetch_wall_seconds": ended-started,
                "origin_last_write_to_end_seconds": ended-network_end,
                "range_complete_to_end_seconds": ended-last_receipt,
                "binding_before_last_write_seconds": network_end-bound_at,
                "manifest_identity": "PASS" if identity else "FAIL",
                "independent_sha256": rehash, "deleted": deleted}
            (folder / "receipt.json").write_text(json.dumps(result, indent=2)+"\n")
            timeline = subprocess.run([sys.executable, "-m", "gpuwm.cli", "timeline",
                str(folder / "ledger.jsonl"), "--output", str(folder / "timeline.json")],
                text=True, capture_output=True)
            (folder / "timeline.log").write_text(timeline.stdout + timeline.stderr)
            result["timeline_exit"] = timeline.returncode
            results.append(result)
        (args.out / "summary.json").write_text(json.dumps(results, indent=2)+"\n")
        print(json.dumps([{key: value for key, value in row.items()
            if key not in ("independent_sha256", "deleted")} for row in results], indent=2))
    finally:
        server.shutdown(); thread.join(); server.server_close()


if __name__ == "__main__":
    main()
