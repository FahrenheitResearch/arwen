"""A numpy namespace that counts where CuPy would synchronise the stream.

RUC's cost on the card is not its arithmetic, it is the number of times the
host has to wait for the device inside one call.  Counting those on the card
needs the card; this module counts them on the CPU tier instead, by standing
in for the device namespace with numpy and recording every point at which
CuPy would have had to bring a value back:

* a rank-0 array turned into a Python value -- ``bool``, ``int``, ``float``,
  ``item``, ``tolist``, ``get``.  CuPy copies to pageable host memory, which
  makes the copy synchronous;
* boolean-mask indexing, on BOTH the read and the write side.  CuPy has to
  know how many elements the mask selects before it can shape the gather or
  broadcast the assignment, and it learns that by reading the last element of
  a scan (``_prepare_mask_indexing_single``).

**Why this stand-in is trusted.**  Its per-call counts were compared against
the profile of record -- 50 warm root steps of the 299x299 + 282x129 pair,
``cProfile`` attribution of 3,206 measured device reads per root step over
5 RUC calls -- on a grid shaped like that case (warm land, no snow):

=====================================  ==========  ============
call site                              this model  profile/call
=====================================  ==========  ============
``ruc.py`` ``_horizontal_float_field``        122         121.9
``ruc.py`` ``ruc_surface_temperature_step``    97          93.8
``ruc.py`` ``ruc_land_surface_step``           67          64.8
``ruc.py`` ``ruc_surface_parameters``          35          34.4
=====================================  ==========  ============

Every row agrees to within 3.5 %, which is what says the model counts the
same events the profiler counted rather than a plausible-looking different
set.  It is a MODEL: it cannot see reads inside ``gpuwm.core.ruc_gpu``'s
kernels-and-cupy leaves, so a count taken through it covers the host
transcription and its dispatch, which is where two of the four largest sites
live.
"""

from __future__ import annotations

import collections
import sys
from types import SimpleNamespace

import numpy as _np

#: ``(site, kind) -> count`` since the last :func:`reset`.
READS: collections.Counter = collections.Counter()


def reset() -> None:
    READS.clear()


def total() -> int:
    return READS[("TOTAL", "all")]


def _record(kind: str) -> None:
    frame = sys._getframe(2)
    while frame is not None:
        path = frame.f_code.co_filename.replace("\\", "/")
        if "/gpuwm/core/" in path:
            site = "%s:%d" % (path.rsplit("/", 1)[-1], frame.f_code.co_firstlineno)
            READS[(site, frame.f_code.co_name)] += 1
            break
        frame = frame.f_back
    else:
        READS[("outside gpuwm.core", kind)] += 1
    READS[("TOTAL", kind)] += 1
    READS[("TOTAL", "all")] += 1


def _has_bool_index(key) -> bool:
    for part in (key if isinstance(key, tuple) else (key,)):
        if isinstance(part, _np.ndarray) and part.dtype == _np.bool_:
            return True
    return False


class CountingArray(_np.ndarray):
    """A numpy array that reports the reads its CuPy twin would have made."""

    def __bool__(self):
        _record("scalar")
        return bool(_np.ndarray.__bool__(self))

    def __int__(self):
        _record("scalar")
        return int(_np.asarray(self))

    def __float__(self):
        _record("scalar")
        return float(_np.asarray(self))

    def __index__(self):
        _record("scalar")
        return _np.asarray(self).__index__()

    def item(self, *arguments):
        _record("scalar")
        return _np.asarray(self).item(*arguments)

    def tolist(self):
        _record("scalar")
        return _np.asarray(self).tolist()

    def get(self):
        _record("scalar")
        return _np.asarray(self)

    def __getitem__(self, key):
        if _has_bool_index(key):
            _record("mask_gather")
        return _np.ndarray.__getitem__(self, key)

    def __setitem__(self, key, value):
        if _has_bool_index(key):
            _record("mask_scatter")
        return _np.ndarray.__setitem__(self, key, value)


def counted(value) -> CountingArray:
    """``value`` as an array whose reads are counted."""
    return _np.asarray(value).view(CountingArray)


def _counting(function):
    def call(*arguments, **keywords):
        result = function(*arguments, **keywords)
        if isinstance(result, _np.ndarray):
            return result.view(CountingArray)
        return result

    call.__name__ = getattr(function, "__name__", "call")
    return call


def _counting_tuple(function):
    """``nonzero``: CuPy reads the scanned count to size its output, once."""
    def call(*arguments, **keywords):
        _record("nonzero")
        result = function(*arguments, **keywords)
        return tuple(part.view(CountingArray) for part in result)

    call.__name__ = getattr(function, "__name__", "call")
    return call


def _dtype_normalising(function):
    def call(*arguments, dtype=None, **keywords):
        if dtype is not None:
            return function(*arguments, dtype=_np.dtype(dtype), **keywords)
        return function(*arguments, **keywords)

    call.__name__ = getattr(function, "__name__", "call")
    return call


class _Float32:
    """``RUC_DEVICE_ARRAYS.float32``'s cast-or-dtype duck, over numpy."""

    dtype = _np.dtype(_np.float32)

    def __call__(self, value):
        if isinstance(value, _np.ndarray):
            return value.astype(_np.float32).view(CountingArray)
        return _np.float32(value)


def _validate_batch(group, flags) -> None:
    """The stand-in for the batched device scan: writes flags, reads nothing."""
    for index, array in enumerate(group):
        flags[index:index + 1] = ~_np.all(_np.isfinite(_np.asarray(array)))


def namespace() -> SimpleNamespace:
    """A counting mirror of ``gpuwm.core.ruc_gpu.RUC_DEVICE_ARRAYS``.

    The same names, and only those names, so a body that reaches for a
    numpy name this namespace does not carry fails here exactly as it would
    fail on the device namespace.
    """
    return SimpleNamespace(
        float32=_Float32(),
        int32=_np.int32,
        intp=_np.intp,
        integer=_np.integer,
        ndarray=CountingArray,
        issubdtype=_np.issubdtype,
        prod=_np.prod,
        abs=_counting(_np.abs),
        all=_counting(_np.all),
        any=_counting(_np.any),
        arange=_dtype_normalising(_counting(_np.arange)),
        array=_dtype_normalising(_counting(_np.array)),
        asarray=_dtype_normalising(_counting(_np.asarray)),
        atleast_1d=_counting(_np.atleast_1d),
        broadcast_to=_counting(_np.broadcast_to),
        count_nonzero=_counting(_np.count_nonzero),
        empty=_dtype_normalising(_counting(_np.empty)),
        full=_dtype_normalising(_counting(_np.full)),
        isfinite=_counting(_np.isfinite),
        maximum=_counting(_np.maximum),
        minimum=_counting(_np.minimum),
        nonzero=_counting_tuple(_np.nonzero),
        stack=_counting(_np.stack),
        where=_counting(_np.where),
        zeros=_dtype_normalising(_counting(_np.zeros)),
        ruc_validate_batch=_validate_batch,
    )
