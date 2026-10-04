"""Orchestration for bounded Rust prepared-array writes."""
from __future__ import annotations

import ctypes
import errno
import io
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np


class _ArrayWrite(ctypes.Structure):
    _fields_ = [
        ("path", ctypes.c_void_p), ("path_length", ctypes.c_size_t),
        ("header", ctypes.c_void_p), ("header_length", ctypes.c_size_t),
        ("data", ctypes.c_void_p), ("data_length", ctypes.c_size_t),
        ("hash_prefix", ctypes.c_void_p), ("hash_prefix_length", ctypes.c_size_t),
    ]


def native_writer():
    """Optional additive ABI; older installed bridges keep serial writes."""
    from gpuwm.core import portable_math
    entry = getattr(portable_math._load(), "gpuwm_write_prepared_arrays", None)
    if entry is not None:
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        entry.argtypes = [pointer, size, size, pointer, pointer, pointer]
        entry.restype = ctypes.c_int32
    return entry


def batch_budget():
    from gpuwm.ingest.cpu_backend import automatic_workers
    from gpuwm.ingest.preparation_workers import host_available_bytes
    workers = automatic_workers()
    memory = host_available_bytes()
    # At most 10% of currently available RAM, and 256 MiB per worker.
    # An individual array is the irreducible transfer of the existing
    # writer; callers flush first when it alone exceeds this batch size.
    limit = workers * 256 * 1024**2
    if memory is not None:
        limit = min(limit, max(1, memory // 10))
    return workers, limit


def write_arrays(entry, arrays, *, workers):
    """Write [(private_path, immutable contiguous_array)] and return hashes.

    All native work finishes before this function returns. Atomic renames
    and manifest publication remain with the ordered caller.
    """
    started = perf_counter()
    keepalive = []
    jobs = (_ArrayWrite * len(arrays))()
    for index, (path, array) in enumerate(arrays):
        header = io.BytesIO()
        # This is precisely np.save's metadata formatting, including its
        # version choice, padding and Unicode dtype descriptors.
        try:
            from numpy.lib._format_impl import _write_array_header
        except ImportError:  # NumPy before the format module split.
            from numpy.lib.format import _write_array_header
        _write_array_header(header, np.lib.format.header_data_from_array_1_0(array), version=None)
        path_bytes = str(Path(path)).encode("utf-8")
        header_bytes = header.getvalue()
        prefix = (array.dtype.str + ";" + json.dumps(
            list(array.shape), separators=(",", ":")) + ";").encode("ascii")
        buffers = [ctypes.create_string_buffer(value) for value in (path_bytes, header_bytes, prefix)]
        keepalive.extend(buffers)
        jobs[index] = _ArrayWrite(
            ctypes.addressof(buffers[0]), len(path_bytes),
            ctypes.addressof(buffers[1]), len(header_bytes),
            array.ctypes.data, array.nbytes,
            ctypes.addressof(buffers[2]), len(prefix))
    digests = (ctypes.c_ubyte * (len(arrays) * 32))()
    codes = (ctypes.c_int32 * len(arrays))()
    created = (ctypes.c_ubyte * len(arrays))()
    try:
        code = int(entry(jobs, len(arrays), workers, digests, codes, created))
        if code:
            raise RuntimeError(f"native prepared-array writer failed with code {code}")
        for index, result in enumerate(codes):
            if result:
                number = result if result > 0 else errno.EIO
                detail = ctypes.FormatError(number) if os.name == "nt" else os.strerror(number)
                raise OSError(number, detail, str(arrays[index][0]))
    except BaseException:
        for owned, (path, _array) in zip(created, arrays):
            if owned:
                Path(path).unlink(missing_ok=True)
        raise
    payload_bytes = sum(array.nbytes for _path, array in arrays)
    from gpuwm.core import portable_math
    query = getattr(portable_math._load(), "gpuwm_preprocess_cpu_parallelism", None)
    actual = None
    if query is not None:
        query.argtypes = [ctypes.c_size_t]
        query.restype = ctypes.c_size_t
        actual = int(query(min(workers, len(arrays))))
    from gpuwm.ingest.preparation_workers import diagnostic
    diagnostic("GPUWM_PREP_WRITE " + json.dumps({
        "stage": "prepared_array_write_and_hash", "arrays": len(arrays),
        "requested_workers": workers, "effective_workers": actual,
        "payload_bytes": payload_bytes,
        "native_buffer_bytes_per_worker": 128 * 1024,
        "elapsed_seconds": perf_counter() - started,
    }, sort_keys=True))
    return [bytes(digests[index * 32:(index + 1) * 32]).hex() for index in range(len(arrays))]
