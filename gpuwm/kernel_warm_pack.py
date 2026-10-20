"""Exact-request cubins retained automatically by the model's compile plan.

The index belongs to one shape/options/numerics/compiler/SM tuple. Cubins are
immutable and content addressed. Only actual successful compiler results enter
it; a missing or damaged image takes the ordinary parallel compiler path.
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

SCHEMA = "gpuwm.warm-images/v1"


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


class WarmImages:
    def __init__(self, root, identity, tag, arch):
        self.binding = dict(schema=SCHEMA, compiler=identity, tag=tag, arch=str(arch))
        self.root = Path(root) / "images" / _digest(self.binding)
        self.index = self.root / "index.json"
        self.rows = self._read()
        self.memory = {}
        self.pending = {}
        self.lock = threading.RLock()
        self.writer = None
        self.unbound = set()
        self.jobs = []
        self.counts = dict(index_entries=len(self.rows), hits=0, misses=0, stored=0, invalid=0, errors=0, unbound=0)

    def _read(self):
        try:
            document = json.loads(self.index.read_text())
            if document["binding"] == self.binding and isinstance(document["entries"], dict):
                return document["entries"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return {}

    @staticmethod
    def key(arguments):
        # Preserve filename, options in order, aliases and backend selectors.
        from gpuwm.kernel_dependencies import header_dependencies
        dependencies = header_dependencies(arguments.get("source", ""), arguments.get("options", ()), arguments.get("filename", "kern.cu"))
        return _digest(dict(request=arguments, headers=dependencies)) if dependencies else _digest(arguments)

    def _request_key(self, arguments):
        try:
            return self.key(arguments)
        except (OSError, ValueError) as error:
            with self.lock:
                self.counts["unbound"] += 1
                detail = str(error)
                if detail not in self.unbound:
                    self.unbound.add(detail)
                    warnings.warn(f"warm image dependency is unbound; using ordinary compiler/cache: {detail}", RuntimeWarning)
            return None

    def get(self, arguments):
        key = self._request_key(arguments)
        if key is None:
            return None
        with self.lock:
            if key in self.memory:
                image, mapping = self.memory[key]
                self.counts["hits"] += 1
                return image, dict(mapping) if isinstance(mapping, dict) else mapping
            row = self.rows.get(key)
            if row is None:
                self.counts["misses"] += 1
                return None
            try:
                if row["sha256"] != _digest(dict(key=key, image=row["image"], mapping=row["mapping"])):
                    raise ValueError("index entry digest mismatch")
                digest = row["image"]
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("invalid image digest")
                image = (self.root / (digest + ".cubin")).read_bytes()
                if not image.startswith(b"\x7fELF") or hashlib.sha256(image).hexdigest() != digest:
                    raise ValueError("image digest mismatch")
                mapping = row["mapping"]
                if mapping is not None and (not isinstance(mapping, dict) or
                        any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items())):
                    raise ValueError("invalid symbol mapping")
            except (OSError, ValueError, KeyError, TypeError):
                self.counts["invalid"] += 1
                self.rows.pop(key, None)
                warnings.warn("warm kernel image is missing or damaged; recompiling the exact request", RuntimeWarning)
                return None
            self.memory[key] = image, mapping
            self.counts["hits"] += 1
            return image, dict(mapping) if isinstance(mapping, dict) else mapping

    def put(self, arguments, result):
        image, mapping = result
        if not isinstance(image, bytes) or not image.startswith(b"\x7fELF"):
            return  # PTX/linking routes keep their existing cache and compiler.
        key = self._request_key(arguments)
        if key is None:
            return None
        with self.lock:
            if key in self.memory:
                return
            mapping = dict(mapping) if isinstance(mapping, dict) else mapping
            self.memory[key] = image, mapping
            if self.writer is None:
                self.writer = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="warm-image-write")
            self.jobs.append(self.writer.submit(self._write_image, key, image, mapping))

    def _write_image(self, key, image, mapping):
        try:
            digest = hashlib.sha256(image).hexdigest()
            _atomic(self.root / (digest + ".cubin"), image)
            with self.lock:
                row = dict(image=digest, mapping=mapping,
                           sha256=_digest(dict(key=key, image=digest, mapping=mapping)))
                self.rows[key] = row
                self.pending[key] = row
                self.counts["stored"] += 1
        except Exception as error:
            with self.lock:
                self.counts["errors"] += 1
            warnings.warn(f"warm image write failed: {error}", RuntimeWarning)

    def save(self):
        with self.lock:
            jobs = list(self.jobs)
        for job in jobs:
            job.result()
        with self.lock:
            fresh = dict(self.pending)
        if not fresh:
            return
        try:
            # Other members of an ensemble may publish this same shape. Merge
            # under a process lock; never discard another member's new requests.
            import fcntl
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / "index.lock").open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                rows = self._read()
                rows.update(fresh)
                _atomic(self.index, (json.dumps(dict(binding=self.binding, entries=rows), sort_keys=True) + "\n").encode())
            with self.lock:
                for key, row in fresh.items():
                    if self.pending.get(key) == row:
                        self.pending.pop(key, None)
        except Exception as error:
            with self.lock:
                self.counts["errors"] += 1
            warnings.warn(f"warm image index write failed: {error}", RuntimeWarning)

    def close(self):
        self.save()
        if self.writer is not None:
            self.writer.shutdown(wait=True)

    def receipt(self):
        with self.lock:
            return dict(path=str(self.index), **self.counts)



def install_metadata(compiler, root, identity):
    """Intercept only compiler metadata probes, keeping real compilation intact."""
    original = compiler._preprocess
    existing = getattr(original, "_gpuwm_metadata", None)
    if existing is not None:
        return existing
    from gpuwm.compiler_metadata import CompilerMetadata
    cache = CompilerMetadata(root, identity)

    def preprocess(source, options, arch, backend):
        if source != "" or backend != "nvrtc":
            return original(source, options, arch, backend)
        from gpuwm.kernel_dependencies import forced_include
        if forced_include(options):
            warnings.warn("forced-include metadata requires ordinary NVRTC preprocessing", RuntimeWarning)
            return original(source, options, arch, backend)
        arguments = dict(kind="empty-preprocess", options=list(options), arch=str(arch), backend=backend)
        return cache.resolve(arguments, lambda: original(source, options, arch, backend))

    preprocess.__wrapped__ = original
    preprocess._gpuwm_metadata = cache
    compiler._preprocess = preprocess
    return cache
