"""Capture the exact legacy radiation deferred checks for native replay.

Prototype adapter. Successful replay performs no Python verdict callback.
Source pins reject a changed or unknown verdict rather than freeze its result.
This does not provide whole-model rollback after a partially submitted failure.
"""
from contextlib import contextmanager
from pathlib import Path
import ctypes as C
import hashlib
import json
import numpy as np


class Rule(C.Structure):
    _fields_ = [("offset", C.c_size_t), ("count", C.c_size_t),
                ("kind", C.c_uint32), ("reserved", C.c_uint32),
                ("threshold", C.c_uint64), ("active_count", C.c_void_p)]


class Guard(C.Structure):
    _fields_ = [("words", C.c_void_p), ("nwords", C.c_size_t),
                ("rules", C.c_void_p), ("nrules", C.c_size_t),
                ("failed_rule", C.c_void_p), ("failed_word", C.c_void_p)]


class DeferredNativeChecks:
    def __init__(self, plan, helper, module, *, cupy=None, pins=None):
        self.plan, self.helper, self.module = plan, helper, module
        if cupy is None:
            import cupy
        self.cp = cupy
        if pins is None:
            pins = json.loads(Path(__file__).with_name("radiation_verdict_pins.json").read_text())
        self.pins = {(p["file"], p["name"], p["line"]): p for p in pins}
        self.groups = []
        self._sinks = []

    @contextmanager
    def install(self):
        """Patch only during single-threaded capture; restore even on failure."""
        old = self.module.deferred
        self.module.deferred = self.deferred
        try:
            yield self
        finally:
            self.module.deferred = old

    @contextmanager
    def active_chunk(self, count, threshold):
        """Associate captured chunk checks with its live compacted-column count."""
        if not isinstance(count, C.c_uint64) or not 0 <= threshold < 2**64:
            raise ValueError("active chunk requires uint64 count and threshold")
        if not self._sinks:
            raise RuntimeError("active chunk requires a deferred scope")
        sink, activity = self._sinks[-1]
        start = len(sink)
        try:
            yield
        finally:
            for index in range(start, len(sink)):
                if index in activity:
                    raise RuntimeError("overlapping active chunk scopes")
                activity[index] = (count, threshold)

    def _pin(self, value, verdict, hashes):
        code = verdict.__code__
        path = Path(code.co_filename)
        key = (path.name, code.co_name, code.co_firstlineno)
        pin = self.pins.get(key)
        if pin is None:
            raise RuntimeError(f"unrecognized radiation verdict {key!r}")
        if path not in hashes:
            hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[path] != pin["sha256"]:
            raise RuntimeError(f"changed radiation verdict source {path}")
        if np.dtype(value.dtype) != np.dtype(pin["dtype"]):
            raise RuntimeError(f"unexpected radiation verdict dtype {key!r}")
        size, shape = int(value.size), tuple(value.shape)
        if size < 1 or (pin["shape"] == "scalar" and shape != ()) or (pin["shape"] == "one" and size != 1):
            raise RuntimeError(f"unexpected radiation verdict shape {key!r}: {shape}")
        return pin

    @contextmanager
    def deferred(self):
        sink, activity = [], {}
        token = self.module._SINK.set(sink)
        self._sinks.append((sink, activity))
        try:
            yield
        finally:
            self._sinks.pop()
            self.module._SINK.reset(token)
        if not sink:
            return
        hashes = {}
        pins = [self._pin(v, f, hashes) for v, f in sink]
        # Large concatenate uploads an immutable pointer/stride descriptor;
        # small concatenate can issue kernels without an upload.
        with self.plan.constant_uploads(private_events=True, allow_empty=True):
            flat = self.cp.concatenate([self.cp.ravel(v).astype(self.cp.float64)
                                        for v, _ in sink])
        with self.plan.readback() as readback:
            host = self.cp.asnumpy(flat)
        address, nbytes = self.plan.host_buffer(readback[0])
        if nbytes != int(flat.size) * 8:
            raise RuntimeError("deferred readback size differs from packed float64")
        words = np.ctypeslib.as_array((C.c_double * int(flat.size)).from_address(address))
        rules = (Rule * len(sink))()
        offsets, offset, count_owners = [], 0, []
        for index, ((value, _), pin) in enumerate(zip(sink, pins)):
            count, threshold = activity.get(index, (None, 0))
            rules[index] = Rule(offset, int(value.size), pin["kind"], 0,
                                threshold, C.addressof(count) if count is not None else None)
            offsets.append(offset)
            offset += int(value.size)
            if count is not None:
                count_owners.append(count)
        failed_rule, failed_word = C.c_uint64(), C.c_uint64()
        arguments = Guard(address, int(flat.size), C.addressof(rules), len(sink),
                          C.addressof(failed_rule), C.addressof(failed_word))

        def original_error():
            index = failed_rule.value
            if index >= len(sink):
                return RuntimeError("native radiation guard returned invalid failure index")
            value, verdict = sink[index]
            start = offsets[index]
            part = words[start:start + int(value.size)].astype(value.dtype).reshape(value.shape)
            try:
                verdict(np.asarray(part))
            except Exception as error:
                return error
            return RuntimeError("native radiation refusal disagreed with original verdict")

        operation = self.plan.native_host(
            self.helper, "host_numeric_guard", arguments, sink, flat, rules,
            words, failed_rule, failed_word, *count_owners, error=original_error)
        # Preserve the ordinary capture-time verdicts and their exact casts.
        for index, (value, verdict) in enumerate(sink):
            count, threshold = activity.get(index, (None, 0))
            if count is not None and count.value <= threshold:
                continue
            start = offsets[index]
            verdict(np.asarray(host[start:start + int(value.size)].astype(value.dtype).reshape(value.shape)))
        self.groups.append({"readback": readback[0], "native": operation,
                            "checks": len(sink), "words": int(flat.size),
                            "sources": {str(k): v for k, v in hashes.items()}})
