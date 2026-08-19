"""Fetch a URL list through the engine's transfer pool, and receipt it.

The generic many-file acquisition driver: one line per object, the
bounded pool from :mod:`gpuwm.fetch_pool` moving the bytes, and the same
fail-closed per-file discipline as every ``gpuwm fetch`` route -- a
unique staged name promoted only after verification, one failed file
refusing by name, and a receipt stating files, bytes, workers, wall and
the effective speedup against the serial model.

Two jobs today:

* the MEASURED before/after instrument for pool sizing (``--workers 1``
  is the serial baseline; the default is the engine default), against
  any real endpoint that publishes many objects per cycle;
* the seam a future mapped-source acquisition route drops onto: sources
  that publish one object per field per lead need exactly this shape.

URL list format, one object per line (blank lines and ``#`` comments
ignored)::

    URL [<TAB> expected_sha256 [<TAB> expected_bytes]]

When an expectation is given it is verified fail-closed; when it is
not, the driver still refuses empty payloads and records the observed
sha256 and size, so a later run of the same list CAN be held to them.

Politeness is the pool's: per-host caps (NOMADS harder than everyone),
the node-wide NOMADS spacing governor under every request, and
``Retry-After`` honored between attempts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request

from gpuwm import fetch_pool
from gpuwm.nomads_governor import paced_urlopen, retry_after_seconds

RECEIPT_SCHEMA = "gpuwm-fetch-concurrency-measurement-v1"

#: Attempts per file: one retry, then the refusal names the file.
ATTEMPTS = 2

_USER_AGENT = "gpuwm-fetch-pool/1"


def parse_url_list(text: str) -> list[dict]:
    """``[{url, name, sha256|None, bytes|None}, ...]``, refusing junk."""

    jobs: list[dict] = []
    names: set[str] = set()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        url = fields[0].strip()
        if not url.startswith(("http://", "https://", "file:")):
            raise ValueError(
                f"line {lineno}: {url!r} is not a fetchable URL")
        name = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
        if not name:
            raise ValueError(f"line {lineno}: {url!r} names no object")
        if name in names:
            raise ValueError(
                f"line {lineno}: duplicate object name {name!r}; two list "
                "entries would overwrite each other's bytes")
        names.add(name)
        expected_sha = fields[1].strip().lower() if len(fields) > 1 else None
        expected_bytes = int(fields[2]) if len(fields) > 2 else None
        if expected_sha is not None and len(expected_sha) != 64:
            raise ValueError(
                f"line {lineno}: {expected_sha!r} is not a sha256 hex digest")
        jobs.append({"url": url, "name": name, "sha256": expected_sha or None,
                     "bytes": expected_bytes})
    if not jobs:
        raise ValueError("the URL list names no objects")
    return jobs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(spec: dict, out: Path, *, timeout: float = 300.0,
                 opener=paced_urlopen, sleeper=time.sleep) -> dict:
    """Transfer and verify one object; return its receipt entry.

    Staged under a per-process unique name and promoted only after the
    bars pass, exactly as the fetch routes stage; a failed attempt
    removes its own staging.  The wait before the retry honors the
    server's ``Retry-After`` when it sent one.
    """

    destination = out / spec["name"]
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        partial = destination.with_suffix(
            destination.suffix + f".{os.getpid()}-{time.time_ns()}.part")
        try:
            request = Request(spec["url"],
                              headers={"User-Agent": _USER_AGENT})
            with opener(request, timeout=timeout) as response, \
                    partial.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            size = partial.stat().st_size
            if size == 0:
                raise ValueError("the server returned an empty payload")
            if spec["bytes"] is not None and size != spec["bytes"]:
                raise ValueError(
                    f"{size:,} B landed where the list expects "
                    f"{spec['bytes']:,} B")
            digest = _sha256(partial)
            if spec["sha256"] is not None and digest != spec["sha256"]:
                raise ValueError(
                    "sha256 mismatch against the list's expectation")
            os.replace(partial, destination)
            return {"name": spec["name"], "url": spec["url"],
                    "bytes": size, "sha256": digest,
                    "seconds": round(time.perf_counter() - started, 6),
                    "attempts": attempt}
        except Exception as error:
            partial.unlink(missing_ok=True)
            last_error = error
            if attempt == ATTEMPTS:
                break
            asked = retry_after_seconds(error)
            sleeper(max(2.0 * attempt, asked if asked is not None else 0.0))
    detail = last_error
    if isinstance(detail, HTTPError):
        detail = f"HTTP {detail.code} {detail.reason}"
    raise RuntimeError(
        f"{spec['name']}: refused after {ATTEMPTS} attempts ({detail}); "
        f"url {spec['url']}") from last_error


def run(url_list: Path, out: Path, *, workers: int | None,
        label: str | None = None, opener=paced_urlopen) -> dict:
    specs = parse_url_list(url_list.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        fetch_pool.TransferJob(
            name=spec["name"], url=spec["url"],
            action=(lambda spec=spec: download_one(spec, out,
                                                   opener=opener)))
        for spec in specs]
    entries, receipt = fetch_pool.run_transfers(jobs, workers=workers)
    return {
        "schema": RECEIPT_SCHEMA,
        "label": label,
        "url_list": str(url_list),
        "out": str(out),
        "concurrency": receipt,
        "files": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url-list", type=Path, required=True,
                        help="one object per line: URL [TAB sha256 "
                             "[TAB bytes]]")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory (created)")
    parser.add_argument("--workers", type=int, default=None,
                        help="files in flight at once (default "
                             f"{fetch_pool.DEFAULT_FILE_WORKERS}; 1 is the "
                             "serial baseline)")
    parser.add_argument("--label", default=None,
                        help="free-text tag recorded in the receipt")
    args = parser.parse_args(argv)
    receipt = run(args.url_list, args.out, workers=args.workers,
                  label=args.label)
    json.dump(receipt, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
