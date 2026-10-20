"""Explicit CUDA recording and native submission, with typed parameter slots.

This low-level API does not infer host model clocks or control flow. An adapter
must bind those values and preserve allocation/module/event lifetimes before a
recording can replace a Python model step.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes as C
import os
from pathlib import Path
import threading


_ERROR_HOOK = C.CFUNCTYPE(None, C.c_int, C.c_size_t, C.c_char_p)


class RecordingError(RuntimeError):
    pass


class Plan:
    """Own a launch sequence and keep its device resources alive.

    Recording executes the original Python path. Replay never invokes it.
    Only an explicit parameter binding changes a recorded argument.
    """
    def __init__(self, library=None, *, diagnostic=False):
        path = Path(library or os.environ["GPUWM_NATIVE_STEP_LIBRARY"]).resolve()
        if str(path) not in os.environ.get("LD_PRELOAD", "").split(":"):
            raise RecordingError("start the child with this library in LD_PRELOAD before importing CuPy")
        self.library = C.CDLL(str(path))
        self._configure()
        self.handle = self.library.awstep_create()
        if not self.handle:
            raise MemoryError("native step plan")
        self.resources = []
        self._allocation_pool = None
        self._recording = False
        self._closed = False
        self._owner = threading.get_ident()
        self._predicates = (0, 0)
        self._ready = False
        self._guard_errors = {}
        self.failure_stack = []
        self.audit_events = []
        self._diagnostic_callback = None
        if diagnostic:
            self._check(self.library.awstep_audit(self.handle))
            self.trace_failures()

    def _configure(self):
        lib = self.library
        signatures = {
            "create": (C.c_void_p, []),
            "begin": (C.c_int, [C.c_void_p]),
            "audit": (C.c_int, [C.c_void_p]),
            "transfer_bytes": (C.c_size_t, [C.c_void_p, C.c_size_t]),
            "error_hook": (C.c_int, [C.c_void_p, _ERROR_HOOK]),
            "end": (C.c_int, [C.c_void_p]),
            "host_begin": (C.c_int, [C.c_void_p, C.c_uint, C.c_uint]),
            "host_end": (C.c_int, [C.c_void_p, C.POINTER(C.c_size_t)]),
            "guard_index": (C.c_size_t, [C.c_void_p]),
            "guard_value": (C.c_uint64, [C.c_void_p]),
            "predicate": (C.c_int, [C.c_void_p, C.c_uint64, C.c_uint64]),
            "bind": (C.c_int, [C.c_void_p] + [C.c_size_t] * 4),
            "execute": (C.c_int, [C.c_void_p, C.c_void_p, C.c_size_t, C.c_uint64]),
            "count": (C.c_size_t, [C.c_void_p]),
            "kernel_count": (C.c_size_t, [C.c_void_p]),
            "failed": (C.c_size_t, [C.c_void_p]),
            "error": (C.c_char_p, [C.c_void_p]),
            "op": (C.c_int, [C.c_void_p, C.c_size_t, C.POINTER(C.c_uint),
                             C.POINTER(C.c_size_t), C.c_char_p, C.c_size_t]),
            "api": (C.c_int, [C.c_void_p, C.c_size_t, C.c_char_p, C.c_size_t]),
            "arg": (C.c_size_t, [C.c_void_p, C.c_size_t, C.c_size_t, C.c_void_p, C.c_size_t]),
            "destroy": (None, [C.c_void_p]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(lib, "awstep_" + name)
            function.restype, function.argtypes = restype, argtypes
        for name in ("awstep_finite_begin", "awstep_masked_zero_begin"):
            if hasattr(lib, name):
                getattr(lib, name).restype = C.c_int
                getattr(lib, name).argtypes = [C.c_void_p, C.c_uint, C.c_uint64]
        if hasattr(lib, "awstep_value_begin"):
            lib.awstep_value_begin.restype = C.c_int
            lib.awstep_value_begin.argtypes = [C.c_void_p, C.c_uint, C.c_uint64, C.c_uint64]
        if hasattr(lib, "awstep_host_end_optional"):
            lib.awstep_host_end_optional.restype = C.c_int
            lib.awstep_host_end_optional.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
        if hasattr(lib, "awstep_numeric_begin"):
            lib.awstep_numeric_begin.restype = C.c_int
            lib.awstep_numeric_begin.argtypes = [C.c_void_p, C.POINTER(C.c_ubyte), C.c_size_t, C.c_double]
        if hasattr(lib, "awstep_event_owned"):
            lib.awstep_event_owned.restype = C.c_int
            lib.awstep_event_owned.argtypes = [C.c_void_p, C.c_size_t]

    def _check(self, rc):
        if rc:
            detail = self.library.awstep_error(self.handle).decode()
            index = self.library.awstep_failed(self.handle)
            if rc == -1007 and index in self._guard_errors:
                column = self.library.awstep_guard_index(self.handle)
                value = self.library.awstep_guard_value(self.handle)
                raise self._guard_errors[index](column, value)
            raise RecordingError(f"native step rc={rc}, operation={index}: {detail}")

    def _same_thread(self):
        if self._closed:
            raise RecordingError("plan is closed")
        if threading.get_ident() != self._owner:
            raise RecordingError("a plan belongs to its creating host thread")

    def trace_failures(self):
        """Attach diagnostic Python frames at the first capture failure."""
        self._same_thread()
        import traceback
        def failed(code, operation, message):
            self.failure_stack = [
                {"file": f.filename, "line": f.lineno, "function": f.name}
                for f in traceback.extract_stack()[:-1]]
            self.audit_events.append({"code": code, "operation": operation,
                                      "reason": message.decode(), "stack": self.failure_stack})
        self._diagnostic_callback = _ERROR_HOOK(failed)
        self._check(self.library.awstep_error_hook(self.handle, self._diagnostic_callback))

    @contextmanager
    def record(self, *resources):
        self._same_thread()
        import cupy as cp
        if self._ready or self._recording:
            raise RecordingError("a launch sequence is recorded only once")
        self.resources.extend(resources)
        allocator = cp.cuda.get_allocator()
        # A private pool permits the original temporary-address reuse during
        # recording and reserves its backing blocks for replay. Retaining
        # each temporary separately would materialize the whole step.
        self._allocation_pool = cp.cuda.MemoryPool(allocator)
        self.resources.append(self._allocation_pool)
        self._check(self.library.awstep_begin(self.handle))
        self._recording = True
        try:
            with cp.cuda.using_allocator(self._allocation_pool.malloc):
                yield self
        except BaseException:
            self.library.awstep_end(self.handle)
            raise
        else:
            self._check(self.library.awstep_end(self.handle))
            self._ready = True
        finally:
            self._recording = False

    @contextmanager
    def when(self, *, require=0, reject=0):
        """Group commands under explicit per-step cadence bits."""
        self._same_thread()
        old = self._predicates
        new = (old[0] | require, old[1] | reject)
        self._check(self.library.awstep_predicate(self.handle, *new))
        self._predicates = new
        try:
            yield
        finally:
            self._predicates = old
            self._check(self.library.awstep_predicate(self.handle, *old))

    @contextmanager
    def constant_uploads(self, *, private_events=False, allow_empty=False):
        """Retain immutable H2D source bytes under an explicit caller contract.

        allow_empty=True admits a device-only library path with zero uploads.
        It never changes the exact-one-readback requirement of guards.

        Use only for values invariant over the plan lifetime. A changing host
        input must not enter this scope. Recording executes the original copy;
        native replay reads the independently owned pinned snapshot.

        private_events=True additionally declares any event records in this
        scope to be internal upload bookkeeping, with no public observer,
        timing query or external waiter. Replay uses owned ordering events;
        ordinary public events keep their caller-owned lifetime contract.
        """
        self._same_thread()
        end_scope = (self.library.awstep_host_end_optional if allow_empty
                     else self.library.awstep_host_end)
        self._check(self.library.awstep_host_begin(self.handle, 3 if private_events else 1, 0))
        index = C.c_size_t()
        try:
            yield
        except BaseException:
            end_scope(self.handle, C.byref(index))
            raise
        else:
            self._check(end_scope(self.handle, C.byref(index)))

    @contextmanager
    def zero_guard(self, error, *, word_bytes=1, mask=None):
        """One D2H zero-status check, preserving the first failing word.

        ``error(index, value)`` returns the original model exception. It runs
        only on failure, after native replay stopped before later operations.
        A failure invalidates this partially submitted plan. Recording makes
        an additional read into owned pinned storage; replay makes one read.
        The scope must contain exactly one D2H transfer. It does not support
        dynamic subset cardinality or arbitrary host arithmetic. An optional
        uint64 mask restricts checks to selected words in a <=64-word result.
        """
        self._same_thread()
        if not callable(error):
            raise TypeError("error must construct the model exception")
        if mask is None:
            self._check(self.library.awstep_host_begin(self.handle, 2, word_bytes))
        else:
            if not isinstance(mask, int) or not 0 < mask < (1 << 64):
                raise ValueError("zero guard mask must be a nonzero uint64")
            self._check(self.library.awstep_masked_zero_begin(self.handle, word_bytes, mask))
        index = C.c_size_t()
        try:
            yield
        except BaseException:
            self.library.awstep_host_end(self.handle, C.byref(index))
            raise
        else:
            rc = self.library.awstep_host_end(self.handle, C.byref(index))
            if index.value != C.c_size_t(-1).value:
                self._guard_errors[index.value] = error
            self._check(rc)

    @contextmanager
    def finite_guard(self, error, *, word_bytes, mask):
        """Check selected IEEE float words from one D2H result inside replay.

        The nonzero bit mask selects up to 64 float32 or float64 words.
        NaNs and either infinity fail; signed zeros and subnormals pass.
        The exception factory receives the index and original raw word bits.
        This checks finiteness only, not derived arithmetic or thresholds.
        """
        self._same_thread()
        if not callable(error):
            raise TypeError("error must construct the model exception")
        if word_bytes not in (4, 8) or not isinstance(mask, int) or not 0 < mask < (1 << 64):
            raise ValueError("finite guard needs float32/float64 and a nonzero uint64 mask")
        self._check(self.library.awstep_finite_begin(self.handle, word_bytes, mask))
        index = C.c_size_t()
        try:
            yield
        except BaseException:
            self.library.awstep_host_end(self.handle, C.byref(index))
            raise
        else:
            rc = self.library.awstep_host_end(self.handle, C.byref(index))
            if index.value != C.c_size_t(-1).value:
                self._guard_errors[index.value] = error
            self._check(rc)

    @contextmanager
    def value_guard(self, error, *, word_bytes, mask, expected):
        """Require selected integer words to equal one exact expected value."""
        self._same_thread()
        if not callable(error):
            raise TypeError("error must construct the model exception")
        if word_bytes not in (1, 4, 8) or not isinstance(mask, int) or not 0 < mask < (1 << 64):
            raise ValueError("value guard needs width 1/4/8 and a nonzero uint64 mask")
        if not isinstance(expected, int) or not 0 <= expected < (1 << (8 * word_bytes)):
            raise ValueError("expected value does not fit word width")
        self._check(self.library.awstep_value_begin(self.handle, word_bytes, mask, expected))
        index = C.c_size_t()
        try:
            yield
        except BaseException:
            self.library.awstep_host_end(self.handle, C.byref(index))
            raise
        else:
            rc = self.library.awstep_host_end(self.handle, C.byref(index))
            if index.value != C.c_size_t(-1).value:
                self._guard_errors[index.value] = error
            self._check(rc)

    @contextmanager
    def numeric_guard(self, error, *, rules, threshold):
        """Apply ordered binary64 predicates to one D2H result in native code.

        One rule per word: 1 fails on numeric nonzero; 2 fails below the
        finite threshold except for negative infinity. Signed zero and NaN
        follow IEEE comparisons. The rule vector is copied, has no 64-word
        limit and must exactly cover the readback. Error receives raw bits.
        This is an explicit adapter contract, not arbitrary Python replay.
        """
        import math
        self._same_thread()
        if not callable(error):
            raise TypeError("error must construct the model exception")
        rules = tuple(rules)
        if not rules or any(type(rule) is not int or rule not in (1, 2) for rule in rules):
            raise ValueError("numeric rules must be a nonempty sequence of 1 or 2")
        if not math.isfinite(threshold):
            raise ValueError("numeric threshold must be finite")
        raw = (C.c_ubyte * len(rules))(*rules)
        self._check(self.library.awstep_numeric_begin(self.handle, raw, len(rules), threshold))
        index = C.c_size_t()
        try:
            yield
        except BaseException:
            self.library.awstep_host_end(self.handle, C.byref(index))
            raise
        else:
            rc = self.library.awstep_host_end(self.handle, C.byref(index))
            if index.value != C.c_size_t(-1).value:
                self._guard_errors[index.value] = error
            self._check(rc)

    @contextmanager
    def readback(self):
        """Capture one synchronized D2H payload for a later native consumer.

        The yielded list receives its operation index on normal scope exit.
        Borrow its owned storage with host_buffer before submitting. The native
        consumer must follow this operation under compatible predicates. This
        does not permit freezing a host branch or publish model output.
        """
        self._same_thread()
        self._check(self.library.awstep_host_begin(self.handle, 4, 0))
        index = C.c_size_t()
        operation = []
        try:
            yield operation
        except BaseException:
            self.library.awstep_host_end(self.handle, C.byref(index))
            raise
        else:
            self._check(self.library.awstep_host_end(self.handle, C.byref(index)))
            operation.append(index.value)

    @contextmanager
    def native_upload(self):
        """Retain one mutable H2D source for a preceding native producer.

        The yielded list receives the upload operation index. Borrow that
        buffer with host_buffer and bind the earlier native producer's output
        pointer to it through its frame. Capture uses the ordinary source;
        replay uses owned storage written by that producer on every selected
        replay. Any copy-bookkeeping events are private and retained. The upload
        completes on its stream before a later replay can overwrite its buffer.
        The producer must run before the upload under compatible predicates.
        """
        self._same_thread()
        self._check(self.library.awstep_host_begin(self.handle, 5, 0))
        operation, index = [], C.c_size_t()
        try:
            yield operation
        except BaseException:
            self.library.awstep_host_end(self.handle, C.byref(index))
            raise
        else:
            self._check(self.library.awstep_host_end(self.handle, C.byref(index)))
            operation.append(index.value)

    def count_predicate(self, count, *, greater_than, bit):
        """Select later groups from a live native-produced uint64 count.

        Retains count storage until close. Record after its synchronized
        producer. Only the named flag bit changes; other flags survive.
        Capture executes the ordinary path, so an adapter recording padded
        groups must guard all reads and publication using current counts.
        """
        self._same_thread()
        if not isinstance(count, C.c_uint64):
            raise TypeError("count requires a retained ctypes uint64")
        if type(greater_than) is not int or not 0 <= greater_than < 2**64:
            raise ValueError("threshold must fit uint64")
        if type(bit) is not int or not 0 <= bit < 64:
            raise ValueError("predicate bit must be in 0..63")
        function = self.library.awstep_count_predicate
        function.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.c_uint64, C.c_uint]
        function.restype = C.c_int
        self.resources.append(count)
        self._check(function(self.handle, C.byref(count), greater_than, bit))

    def host_buffer(self, operation):
        """Borrow synchronized readback or native upload storage until close.

        Readback addresses feed a later compiled native consumer. Upload
        addresses receive output from a preceding compiled native producer.
        Do not free or use concurrently, and preserve compatible predicates.
        Ordinary immutable upload buffers are not exposed by this interface.
        """
        self._same_thread()
        address, size = C.c_void_p(), C.c_size_t()
        function = self.library.awstep_host_buffer
        function.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p),
                             C.POINTER(C.c_size_t)]
        function.restype = C.c_int
        self._check(function(self.handle, operation, C.byref(address), C.byref(size)))
        return address.value, size.value

    def native_host(self, library, symbol, arguments, *owners, error=None):
        """Run and record a compiled CPU function with a fixed argument ABI.

        The exported C function is ``int(const void*, size_t)``. It must not
        enter Python, call CUDA, or depend on outstanding device work.
        Record explicit stream synchronization before any dependent host read.
        Retain every array behind the argument pointers through ``owners``.
        Bind argument 0 to a same-sized frame field for changing time scalars.
        The library is loaded by CDLL; Python callback trampolines are excluded.
        A nonzero return stops recording/replay before downstream submission.
        For validation rc -1007, an optional error callback reconstructs the
        original exception from retained output metadata, only after failure.
        """
        self._same_thread()
        if not self._recording:
            raise RecordingError("native host work requires an active recording")
        if not isinstance(arguments, C.Structure):
            raise TypeError("native host arguments require a ctypes Structure")
        if error is not None and not callable(error):
            raise TypeError("native error must construct the model exception")
        module = C.CDLL(str(Path(library).resolve()))
        function = getattr(module, symbol)
        self.resources.extend((module, arguments, *owners))
        call = self.library.awstep_native_host
        call.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t,
                         C.c_char_p, C.POINTER(C.c_size_t)]
        call.restype = C.c_int
        operation = C.c_size_t()
        rc = call(self.handle, C.cast(function, C.c_void_p),
                  C.byref(arguments), C.sizeof(arguments),
                  symbol.encode(), C.byref(operation))
        if error is not None:
            self._guard_errors[operation.value] = lambda index, value: error()
        self._check(rc)
        return operation.value

    def bind_host_argument(self, operation, argument, readback, *, byte_offset=0):
        """Bind a later kernel argument to current synchronized readback bytes.

        The source is the operation yielded by ``readback``. Its owned pinned
        buffer is retained until close. Source must precede the kernel and its
        predicate must cover every execution of that kernel. No host arithmetic
        or second Python submission is introduced; exact scalar bits are used.
        """
        self._same_thread()
        function = self.library.awstep_bind_host_argument
        function.argtypes = [C.c_void_p] + [C.c_size_t] * 4
        function.restype = C.c_int
        self._check(function(self.handle, operation, argument, readback, byte_offset))

    def bind(self, operation, argument, frame_type, field):
        """Bind an argument to one exact ctypes Structure field."""
        self._same_thread()
        fields = {name: kind for name, kind, *rest in frame_type._fields_}
        kind = fields[field]
        offset = getattr(frame_type, field).offset
        self._check(self.library.awstep_bind(
            self.handle, operation, argument, offset, C.sizeof(kind)))

    def submit(self, frame=None, *, flags=0):
        """One FFI submission; captured syncs and native guards still execute."""
        self._same_thread()
        if not self._ready:
            raise RecordingError("recording did not complete")
        pointer = C.byref(frame) if frame is not None else None
        size = C.sizeof(frame) if frame is not None else 0
        rc = self.library.awstep_execute(self.handle, pointer, size, flags)
        if rc:
            self._ready = False
        self._check(rc)

    def describe(self):
        self._same_thread()
        rows = []
        for i in range(self.library.awstep_count(self.handle)):
            kind, nargs, name = C.c_uint(), C.c_size_t(), C.create_string_buffer(160)
            self._check(self.library.awstep_op(self.handle, i, C.byref(kind),
                                               C.byref(nargs), name, len(name)))
            api = C.create_string_buffer(64)
            self._check(self.library.awstep_api(self.handle, i, api, len(api)))
            sizes = [self.library.awstep_arg(self.handle, i, j, None, 0)
                     for j in range(nargs.value)]
            rows.append({"index": i, "kind": kind.value, "api": api.value.decode(), "kernel": name.value.decode() if kind.value == 1 else None,
                         "private_event": bool(self.library.awstep_event_owned(self.handle, i)) if hasattr(self.library,"awstep_event_owned") else False,
                         "argument_bytes": sizes, "transfer_elements": self.library.awstep_transfer_bytes(self.handle, i)})
        return {"operations": rows,
                "pool_reserved_bytes": self._allocation_pool.total_bytes() if self._allocation_pool else 0,
                "kernel_launches": self.library.awstep_kernel_count(self.handle),
                "ready": self._ready, "audit_events": self.audit_events, "failure_stack": self.failure_stack,
                "error": self.library.awstep_error(self.handle).decode()}

    def close(self):
        """Call only after the caller has completed the relevant streams."""
        self._same_thread()
        if self._recording:
            raise RecordingError("cannot close an active recording")
        self.library.awstep_destroy(self.handle)
        self.handle = None
        pool = self._allocation_pool
        self.resources.clear()
        self._guard_errors.clear()
        self._allocation_pool = None
        # Model carriers can outlive this plan and keep its private allocator
        # alive. Return every unused capture block now, including radiation
        # workspaces, while preserving all still-live carrier allocations.
        if pool is not None:
            pool.free_all_blocks()
        self._closed = True
