"""Keep prepared resources alive while waiting for a table-selected analysis.

This module only orchestrates. Source selection belongs to the source tables;
GRIB decoding and initialization belong to the existing Rust/GPU preparation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SCHEMA = "gpuwm-standby-v1"


def wall_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class StandbyReceipt:
    """Durable stage boundaries; no device synchronization or array copies."""

    def __init__(self, path, *, background=False, **metadata):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.document = dict(schema=SCHEMA, status="PRESTAGING", stages=[],
                             started_wall=wall_now(), **metadata)
        self._seen_mono = None
        self._first_step_mono = None
        self._prep_stages = {}
        self._condition = threading.Condition(self.lock)
        self._pending = None
        self._version = self._written = 0
        self._write_error = None
        self._stopping = False
        self._writer = None
        if background:
            self._writer = threading.Thread(target=self._write_loop,
                                            name="standby-receipt", daemon=True)
            self._writer.start()
        self.save()
        self.flush()

    def save(self):
        with self.lock:
            if self._write_error is not None:
                raise RuntimeError("standby receipt writer failed") from self._write_error
            payload = json.dumps(self.document, indent=2, allow_nan=False, default=str) + "\n"
            if self._writer is None:
                self._write(payload)
            else:
                self._version += 1
                self._pending = (self._version, payload)
                self._condition.notify_all()

    def _write(self, payload):
        temporary = self.path.with_suffix(".json.part")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, self.path)

    def _write_loop(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._stopping)
                if self._pending is None:
                    return
                version, payload = self._pending
                self._pending = None
            try:
                self._write(payload)
            except BaseException as error:
                with self._condition:
                    self._write_error = error
                    self._condition.notify_all()
                return
            with self._condition:
                self._written = version
                self._condition.notify_all()

    def flush(self):
        if self._writer is not None:
            with self._condition:
                self._condition.wait_for(lambda: self._written == self._version
                                         or self._write_error is not None)
                if self._write_error is not None:
                    raise RuntimeError("standby receipt writer failed") from self._write_error

    def close(self):
        try:
            self.flush()
        finally:
            if self._writer is not None:
                with self._condition:
                    self._stopping = True
                    self._condition.notify_all()
                self._writer.join()

    def update(self, **fields):
        with self.lock:
            self.document.update(fields)
            self.save()

    @contextmanager
    def stage(self, name, *, parallelism):
        row = dict(name=name, start_wall=wall_now(), end_wall=None,
                   parallelism=parallelism, status="RUNNING")
        started = time.monotonic()
        with self.lock:
            self.document["stages"].append(row)
            self.save()
        try:
            yield row
        except BaseException as error:
            row.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        else:
            row["status"] = "DONE"
        finally:
            with self.lock:
                row.update(end_wall=wall_now(), wall_seconds=time.monotonic() - started)
                self.save()

    def seen(self, obj):
        with self.lock:
            observations = self.document.setdefault("object_observations", {})
            if obj.url not in observations:
                observations[obj.url] = {**asdict(obj), "first_seen_wall": wall_now()}
                self.save()
            if self._seen_mono is None:
                self._seen_mono = time.monotonic()
                self.update(object={**asdict(obj), "first_seen_wall": wall_now()})

    def preparation_event(self, event, **fields):
        """Retain native decode/static/state events without adding device syncs."""
        if event != "warning" or fields.get("code") != "preparation_progress":
            return
        record = fields["preparation"]
        key = (threading.get_ident(), record["stage"], record.get("index"))
        with self.lock:
            if record["event"] == "started":
                row = dict(name=record["stage"], start_wall=wall_now(), end_wall=None,
                           status="RUNNING", parallelism=record.get("backend") or
                           "native preparation worker policy", native_event=record)
                self._prep_stages[key] = row
                self.document["stages"].append(row)
            elif record["event"] in ("finished", "failed"):
                row = self._prep_stages.pop(key, None)
                if row is not None:
                    row.update(end_wall=wall_now(), wall_seconds=record["elapsed_seconds"],
                               status="FAILED" if record["event"] == "failed" else "DONE",
                               native_event=record)
            self.save()

    def first_step(self, **fields):
        """Called on the runner's first completed step, never at launch."""
        with self.lock:
            if self._first_step_mono is None:
                self._first_step_mono = time.monotonic()
                if self._seen_mono is None:
                    raise RuntimeError("forecast reported a step before seeing its analysis")
                self.update(first_step_wall=wall_now(), first_step=fields,
                            object_seen_to_first_step_seconds=(
                                self._first_step_mono - self._seen_mono),
                            status="FORECASTING")
                # A late-starting door must not look like a box that was warm
                # before publication merely because its first HEAD was late.
                try:
                    published = min(parsedate_to_datetime(row["last_modified"])
                                    for row in self.document["object_observations"].values()
                                    if row.get("last_modified"))
                    warm = datetime.fromisoformat(self.document["prestage_complete_wall"])
                    step = datetime.fromisoformat(self.document["first_step_wall"])
                    self.update(prestage_before_object_publication=warm <= published,
                                object_last_modified_to_first_step_seconds=(step-published).total_seconds())
                except (KeyError, TypeError, ValueError):
                    self.update(prestage_before_object_publication=None)


@dataclass(frozen=True)
class PostedObject:
    url: str
    size: int
    etag: str | None = None
    last_modified: str | None = None


class ObjectChanged(RuntimeError):
    """A posted object changed while its conditional ranges were read."""


class HTTPObjectStore:
    """Anonymous HEAD and version-bound parallel ranged GETs, for any route."""

    def __init__(self, *, workers=8, timeout=15.0):
        if workers < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("object fetch needs positive workers and timeout")
        self.workers, self.timeout = workers, timeout

    def head(self, url):
        try:
            with urlopen(Request(url, method="HEAD"), timeout=self.timeout) as response:
                size = int(response.headers.get("Content-Length", 0))
                return PostedObject(url, size, response.headers.get("ETag"),
                                    response.headers.get("Last-Modified"))
        except HTTPError as error:
            if error.code == 404:
                return None
            raise

    def _range(self, obj, start, end):
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if obj.etag:
            headers["If-Match"] = obj.etag
        elif obj.last_modified:
            headers["If-Unmodified-Since"] = obj.last_modified
        try:
            response = urlopen(Request(obj.url, headers=headers), timeout=self.timeout)
        except HTTPError as error:
            if error.code in (404, 412, 416):
                raise ObjectChanged(f"analysis changed during fetch: {obj.url}") from error
            raise
        expected = f"bytes {start}-{end}/{obj.size}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            response.close()
            raise RuntimeError(f"analysis server did not honor range {expected}: {obj.url}")
        if obj.etag and response.headers.get("ETag", obj.etag) != obj.etag:
            response.close()
            raise ObjectChanged(f"analysis ETag changed: {obj.url}")
        return response

    def complete(self, obj):
        # An object can exist while a fallback server is still writing it.
        # Version-bound tail reads allow launching on the first complete poll,
        # without imposing an extra size-stability interval on atomic stores.
        if obj.size < 20:
            return False
        with self._range(obj, obj.size - 4, obj.size - 1) as response:
            return response.read(5) == b"7777"

    def fetch(self, obj, destination):
        destination = Path(destination)
        partial = destination.with_suffix(destination.suffix + ".part")
        # The output belongs to this run. Each range writes disjoint offsets;
        # the coordinator never buffers the whole GRIB or joins byte strings.
        count = min(self.workers, max(1, math.ceil(obj.size / (8 * 1024 * 1024))))
        width = math.ceil(obj.size / count)
        with partial.open("wb") as output:
            output.truncate(obj.size)

        def copy_range(start):
            end = min(start + width, obj.size) - 1
            # A server that closes a range early (NOMADS did, V61 2026-10-22
            # 00:12Z: "analysis range ended 82625800 bytes early") is asked
            # again for the rest of that range, from the first missing byte.
            offset, attempts = start, 0
            with partial.open("r+b") as output:
                while offset <= end:
                    try:
                        with self._range(obj, offset, end) as response:
                            output.seek(offset)
                            while offset <= end:
                                block = response.read(min(1024 * 1024, end - offset + 1))
                                if not block:
                                    break
                                output.write(block)
                                offset += len(block)
                            if offset > end and response.read(1):
                                raise IOError("analysis range exceeded its declared length")
                    except (OSError, TimeoutError) as error:
                        if isinstance(error, ObjectChanged):
                            raise
                        attempts += 1
                        if attempts > 8:
                            raise
                        continue
                    if offset <= end:
                        attempts += 1
                        if attempts > 8:
                            raise IOError(f"analysis range ended {end - offset + 1} bytes early after {attempts} attempts")

        try:
            with ThreadPoolExecutor(count, thread_name_prefix="analysis-fetch") as pool:
                list(pool.map(copy_range, range(0, obj.size, width)))
            with partial.open("rb") as source:
                if source.read(4) != b"GRIB":
                    raise IOError("analysis object has no GRIB header")
                source.seek(-4, os.SEEK_END)
                if source.read() != b"7777":
                    raise IOError("analysis object has no complete GRIB end marker")
            os.replace(partial, destination)
            return dict(path=str(destination), bytes=obj.size, workers=count,
                        etag=obj.etag)
        finally:
            partial.unlink(missing_ok=True)


def wait_for_object(urls, store, receipt, *, poll_s=3.0, timeout_s=None,
                    headline=True, stage_name="poll"):
    if not urls or not math.isfinite(poll_s) or poll_s <= 0:
        raise ValueError("standby needs object routes and a positive finite poll interval")
    if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("standby timeout must be positive and finite")
    started = time.monotonic()
    if headline:
        receipt.update(status="WAITING", poll_start_wall=wall_now(), poll_s=poll_s)

    def probe(url):
        try:
            obj = store.head(url)
            if obj is None:
                return None, None
            if headline:
                receipt.seen(obj)
            return (obj if store.complete(obj) else None), None
        except (HTTPError, URLError, TimeoutError, OSError, ObjectChanged) as error:
            return None, f"{url}: {type(error).__name__}: {error}"

    with receipt.stage(stage_name, parallelism=f"{len(urls)} HTTP workers"), \
            ThreadPoolExecutor(len(urls), thread_name_prefix="analysis-poll") as pool:
        while True:
            tick = time.monotonic()
            # Probe fallback beside primary so an unavailable primary cannot
            # hold a complete fallback behind its connection timeout.
            results = list(pool.map(probe, urls))
            errors = [error for _, error in results if error]
            if errors:
                receipt.update(last_probe_errors=errors)
            for obj, _ in results:
                if obj is not None:
                    if headline:
                        receipt.update(complete_object={**asdict(obj), "complete_wall": wall_now()})
                    return obj
            if timeout_s is not None and time.monotonic() - started >= timeout_s:
                raise TimeoutError("analysis did not become complete before standby timeout; "
                                   + "; ".join(errors or urls))
            delay = max(0.0, poll_s - (time.monotonic() - tick))
            if timeout_s is not None:
                delay = min(delay, max(0.0, timeout_s - (time.monotonic() - started)))
            time.sleep(delay)


def run_standby(backend, *, urls, outdir, metadata=None, store=None,
                poll_s=3.0, timeout_s=None):
    """Prestage once, then launch with those same live objects in this process.

    A backend owns ``prestage(receipt)``, ``launch(path, receipt)`` and
    ``close()``. It calls receipt.first_step() only from the actual completed
    step observer, and records decode/state-build through receipt.stage().
    """
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "standby-receipt.json"
    # A second attempt must never erase evidence of a failed warm launch.
    with receipt_path.open("x", encoding="utf-8"):
        pass
    receipt = StandbyReceipt(receipt_path, background=True, object_routes=list(urls),
                             **(metadata or {}))
    store = store or HTTPObjectStore(timeout=min(15.0, poll_s))
    try:
        backend.prestage(receipt)
        receipt.update(prestage_complete_wall=wall_now())
        obj = wait_for_object(urls, store, receipt, poll_s=poll_s, timeout_s=timeout_s)
        path = root / "analysis.grib2"
        with receipt.stage("fetch", parallelism=f"{store.workers} HTTP workers") as stage:
            stage["result"] = store.fetch(obj, path)
        receipt.update(status="LAUNCHING", launch_wall=wall_now())
        result = backend.launch(path, receipt)
        if receipt._first_step_mono is None:
            raise RuntimeError("warmed standby returned without a completed forecast step")
        receipt.update(status="DONE", forecast_receipt=result, finished_wall=wall_now())
    except BaseException as error:
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                       finished_wall=wall_now())
        raise
    finally:
        try:
            backend.close()
        except BaseException as error:
            receipt.update(status="FAILED", cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            receipt.close()
    return receipt.document
