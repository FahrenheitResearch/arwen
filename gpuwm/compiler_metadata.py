"""Compiler metadata cache usable by a bare device-probe interpreter.

Only standard-library imports at module scope. Values are measured NVRTC
output, separated by the digest of the loaded compiler, with atomic writes
and corruption fallback. No CUDA context or model import is required.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import warnings

def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".warm-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CompilerMetadata:
    """Persist measured empty-program output, never an inferred build string.

    CuPy asks NVRTC for this text before looking up an already cached image.
    Exact options and target stay in the key, and the loaded compiler digest
    separates builds with the same reported major/minor version.
    """

    def __init__(self, root, identity):
        self.binding = dict(schema="gpuwm.compiler-metadata/v1", compiler=identity)
        self.root = Path(root) / "compiler-metadata" / _digest(self.binding)
        self.lock = threading.Lock()
        self.flights = {}
        self.counts = dict(hits=0, measured=0, invalid=0, errors=0)

    def resolve(self, arguments, measure):
        key = _digest(arguments)
        with self.lock:
            future = self.flights.get(key)
            owner = future is None
            if owner:
                future = concurrent.futures.Future()
                self.flights[key] = future
        if not owner:
            return future.result()
        path = self.root / (key + ".json")
        try:
            value = None
            try:
                row = json.loads(path.read_text())
                if (row["binding"] != self.binding or row["arguments"] != json.loads(json.dumps(arguments))
                        or not isinstance(row["value"], str) or not row["value"]
                        or row["sha256"] != _digest(dict(binding=self.binding, arguments=arguments, value=row["value"]))):
                    raise ValueError("compiler metadata digest or binding mismatch")
                value = row["value"]
                with self.lock:
                    self.counts["hits"] += 1
            except FileNotFoundError:
                pass
            except (OSError, ValueError, KeyError, TypeError):
                with self.lock:
                    self.counts["invalid"] += 1
                warnings.warn("warm compiler metadata is damaged; measuring it again", RuntimeWarning)
            if value is None:
                value = measure()
                with self.lock:
                    self.counts["measured"] += 1
                if isinstance(value, str) and value:
                    row = dict(binding=self.binding, arguments=arguments, value=value)
                    row["sha256"] = _digest(row)
                    try:
                        _atomic(path, (json.dumps(row, sort_keys=True) + "\n").encode())
                    except OSError as error:
                        with self.lock:
                            self.counts["errors"] += 1
                        warnings.warn(f"warm compiler metadata write failed: {error}", RuntimeWarning)
            future.set_result(value)
            if not value:
                with self.lock:
                    self.flights.pop(key, None)
            return value
        except BaseException as error:
            future.set_exception(error)
            with self.lock:
                self.flights.pop(key, None)
            raise

    def receipt(self):
        with self.lock:
            return dict(path=str(self.root), **self.counts)


def nvrtc_banner(nvrtc, library, root=None):
    """Return the actual banner, persisting it only against resolved library bytes."""
    def measure():
        # Through the keyed helper (A160): a fixed program name compiled
        # directly can be answered from NVRTC's own cache under the other
        # flush mode.  The banner is the PTX header, which the key comment
        # in the source does not touch.
        from gpuwm.nvrtc_cache_key import compile_program
        try:
            ptx = compile_program("", (), name="compile_platform_probe.cu", nvrtc=nvrtc)
        except Exception:
            return ""
        if isinstance(ptx, bytes):
            ptx = ptx.decode("ascii", "replace")
        return "\n".join(line for line in ptx.splitlines()[:12] if line.startswith("//"))

    if library.get("status") != "resolved":
        return measure()
    directory = root or os.environ.get("GPUWM_KERNEL_PLAN_DIR") or Path.home() / ".cache" / "gpuwm" / "kernel-plans"
    cache = CompilerMetadata(Path(directory).expanduser(), {"nvrtc_library_sha256": library["sha256"]})
    return cache.resolve({"kind": "nvrtc-banner"}, measure)
