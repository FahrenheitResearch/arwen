"""Read-only NPY views whose lifetime does not retain Unix descriptors.

Python 3.13 can ask its mmap object not to duplicate the input descriptor.
Older 64-bit Unix runtimes use the operating system's mmap directly. This
module only owns file and virtual-memory resources; NumPy parses the NPY
metadata and views the original file bytes without copying or transforming
them. The mapping owner stays alive through every escaped array view.
"""
from __future__ import annotations

import ctypes
import errno
from functools import lru_cache
import math
import mmap
import os
from pathlib import Path
import sys
import weakref

import numpy as np
try:
    from numpy.lib.format import _read_array_header
except ImportError:
    # NumPy 2.3 moved the same parser behind the public format facade.
    from numpy.lib._format_impl import _read_array_header


_MAX_HEADER_CHARACTERS = 10_000


class _FileMmap(mmap.mmap):
    """Identify our file-backed maps without tagging anonymous mappings."""


class _HeaderReader:
    """Bound declared header reads before NumPy allocates their contents."""

    def __init__(self, stream, version):
        self._stream = stream
        # Format 3 uses UTF-8, with at most four bytes per character. NumPy
        # also enforces its ordinary decoded-character limit after parsing.
        self._remaining = _MAX_HEADER_CHARACTERS * (4 if version == (3, 0) else 1) + 4

    def read(self, size=-1):
        if size < 0 or size > self._remaining:
            raise ValueError("NPY array header exceeds the safe size limit")
        result = self._stream.read(size)
        self._remaining -= len(result)
        return result


@lru_cache(maxsize=1)
def _unix_mapping_functions():
    """The public 64-bit Unix mmap ABI, independent of Python's mmap ABI."""
    if os.name != "posix" or sys.maxsize <= 2**32:
        raise OSError(errno.ENOTSUP,
                      "descriptor-free mappings require a 64-bit Unix runtime")
    library = ctypes.CDLL(None, use_errno=True)
    create = library.mmap
    create.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int64]
    create.restype = ctypes.c_void_p
    release = library.munmap
    release.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    release.restype = ctypes.c_int
    return create, release


class _UnixMapOwner:
    """Release virtual memory only after the last exported array is gone."""

    def __init__(self, release, address, length):
        self._finalizer = weakref.finalize(self, release, address, length)
        # Process exit already releases mappings. Running this finalizer
        # before other exit callbacks could invalidate their live views.
        self._finalizer.atexit = False


def _native_unix_buffer(descriptor, length):
    create, release = _unix_mapping_functions()
    address = create(None, length, mmap.PROT_READ, mmap.MAP_SHARED, descriptor, 0)
    if address == ctypes.c_void_p(-1).value:
        number = ctypes.get_errno() or errno.EIO
        raise OSError(number, os.strerror(number))
    try:
        owner = _UnixMapOwner(release, address, length)
    except BaseException:
        release(address, length)
        raise
    try:
        buffer = (ctypes.c_ubyte * length).from_address(address)
        # ctypes' buffer keeps this owner. frombuffer below keeps a readonly
        # memoryview of that buffer, so slices retain the same owner safely.
        buffer._mapping_owner = owner
        return memoryview(buffer).toreadonly()
    except BaseException:
        owner._finalizer()
        raise


def _readonly_file_buffer(descriptor, length):
    if os.name == "nt":
        # Windows mmap owns OS mapping handles, not a duplicated CRT fd.
        return _FileMmap(descriptor, length, access=mmap.ACCESS_READ)
    try:
        return _FileMmap(descriptor, length, access=mmap.ACCESS_READ, trackfd=False)
    except TypeError:
        # Python 3.11 and 3.12 do not expose trackfd. The native mapping
        # itself does not need an open fd after the input stream closes.
        return _native_unix_buffer(descriptor, length)


def is_file_backed_array(array) -> bool:
    """Recognize reclaimable file pages through an array's ownership chain."""
    owner = array
    seen = set()
    while owner is not None and id(owner) not in seen:
        seen.add(id(owner))
        if isinstance(owner, (np.memmap, _FileMmap)) or isinstance(
                getattr(owner, "_mapping_owner", None), _UnixMapOwner):
            return True
        owner = owner.obj if isinstance(owner, memoryview) else getattr(owner, "base", None)
    return False


def map_npy_readonly(path) -> np.ndarray:
    """Map one NPY array without retaining its source file descriptor.

    The file must remain immutable while any returned array or view lives.
    Manifest verification and file-identity checks belong to the caller.
    OS failures stay OSError, allowing resource limits to be distinguished
    from malformed payloads. No explicit close invalidates escaped views.
    """
    with Path(path).open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version not in {(1, 0), (2, 0), (3, 0)}:
            raise ValueError(f"unsupported NPY format version {version!r}")
        shape, fortran_order, dtype = _read_array_header(
            _HeaderReader(stream, version), version,
            max_header_size=_MAX_HEADER_CHARACTERS)
        if dtype.hasobject:
            raise ValueError("object arrays cannot be mapped without pickle")
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError("NPY array shape must contain nonnegative integer dimensions")
        offset = stream.tell()
        count = math.prod(shape)
        nbytes = count * dtype.itemsize
        length = os.fstat(stream.fileno()).st_size
        if nbytes > sys.maxsize or offset + nbytes > length:
            raise ValueError("NPY array payload is shorter than its declared shape")
        order = "F" if fortran_order else "C"
        if nbytes == 0:
            # A zero-byte array needs no mapping. Immutable bytes preserve
            # readonly behavior even for dtypes with itemsize zero.
            return np.ndarray(shape, dtype=dtype, buffer=b"", order=order)
        buffer = _readonly_file_buffer(stream.fileno(), length)
        # frombuffer preserves a readonly memoryview as its base. Using
        # ndarray(buffer=...) here can unwrap it to writable ctypes storage.
        return np.frombuffer(buffer, dtype=dtype, count=count,
                             offset=offset).reshape(shape, order=order)
