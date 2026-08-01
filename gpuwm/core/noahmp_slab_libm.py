"""The FP32 primitives the vectorised Noah-MP composition is allowed to use.

Noah-MP's column composition -- ENERGY's own arithmetic and NOAHMP_SFLX's
prefix and postfix -- is transcribed in :mod:`gpuwm.core.noahmp_energy` and
:mod:`gpuwm.core.noahmp_sflx` as scalar CPython: Python floats with an explicit
``f32`` rounding call after every operation.  That transcription is what the
unmodified-WRF fixtures pin, and it is also, measured, essentially the entire
cost of a Noah-MP land-surface call.

Evaluating it for every land column at once needs the same arithmetic over
arrays.  This module states, in one place, which array operation reproduces
which scalar one **bitwise**, and refuses to guess:

``+ - * /`` and ``sqrt``
    IEEE-754 correctly rounded in binary32, so a CuPy float32 ufunc is
    bit-identical to ``f32(a op b)`` with no wrapper at all.  Each ufunc is
    its own launch, so nothing here can be contracted into an FMA either.
    ``test_ieee_agreement`` and ``test_sqrt_agreement`` are the gates that say
    so, over 40,006 argument pairs each.

``min`` / ``max``
    **Not** ``cupy.minimum``/``cupy.maximum``.  ENERGY spells them
    ``a if a < b else b`` (:func:`gpuwm.core.noahmp_energy._mn`), which returns
    the *second* argument on a tie; ``cupy.minimum(-0.0, +0.0)`` returns
    ``-0.0`` where ``_mn(-0.0, +0.0)`` returns ``+0.0``.  gfortran does not
    flush, so signed zeros are exactly the input class this project has lost
    bugs to.  :func:`fmn` and :func:`fmx` are the ``where`` forms that match.

``powf`` / ``expf`` / ``logf`` / ``tanhf``
    Neither numpy's float32 versions nor CUDA's device libm are glibc's, and
    gfortran on x86-64 calls glibc's.  These go through
    ``gpuwm/core/kernels/noahmp_libm_slab.cu``, which is a bare elementwise
    wrapper around the single audited device transcription already in this
    tree (``r_pow``/``r_exp``/``r_log`` in ``noahmp_leaves.cu`` and
    ``nmpe_tanhf`` in ``noahmp_energy.cu``).  No second copy is created here,
    which is also why the entry points are spelled ``slab_powf`` and not
    ``powf``: ``tests/test_noahmp_radiation.py`` treats a module under
    ``gpuwm/`` that *defines* ``powf`` as a fork of the transcription, and it
    is right to.

``tests/test_noahmp_slab_libm.py`` holds every claim above against
:mod:`gpuwm.core.noahmp_libm`, with four negative controls: ``cupy.minimum``,
``numpy.power`` and ``cupy.exp`` are each shown failing a gate the wrapper
passes, and ``__double2float_rn`` is shown still flushing the subnormals that
``nmp_d2f_rn`` exists to recover.
"""

from __future__ import annotations

import numpy as np

from gpuwm.certify.kernel_manifest import record_module
from gpuwm.core.noahmp_kernel_sources import translation_unit_source

#: One CUDA thread per column; these kernels are memory bound and elementwise.
THREADS = 128

_MODULE_NAME = "noahmp_libm_slab"


def _blocks(n: int) -> int:
    return (n + THREADS - 1) // THREADS


def _module():
    """Compile ``noahmp_libm_slab`` as the three-part unit it is."""
    global _MODULE_CACHE
    if _MODULE_CACHE is None:
        import cupy as cp

        source = translation_unit_source(_MODULE_NAME)
        _MODULE_CACHE = cp.RawModule(
            code=source,
            options=("-std=c++17",), name_expressions=None)
        _MODULE_CACHE.compile()
        record_module(f"gpuwm.core.noahmp_slab_libm:{_MODULE_NAME}",
                      source=source, options=("-std=c++17",),
                      module=_MODULE_CACHE)
    return _MODULE_CACHE


_MODULE_CACHE = None
_KERNEL_CACHE: dict[str, object] = {}


def _kernel(name: str):
    if name not in _KERNEL_CACHE:
        _KERNEL_CACHE[name] = _module().get_function(name)
    return _KERNEL_CACHE[name]


def _contiguous(array):
    """A raw kernel argument carries a pointer, not a stride.

    Handing a strided view to a raw kernel is the defect that fed column 0's
    neighbouring slots to PRECIP_HEAT on every column; it is silent, it
    survives every per-leaf oracle, and it is one call away every time an
    array is sliced.  Every argument below goes through here.
    """
    import cupy as cp

    return cp.ascontiguousarray(cp.asarray(array, dtype=cp.float32))


def _unary(name, x):
    import cupy as cp

    x = _contiguous(x)
    out = cp.empty_like(x)
    n = int(x.size)
    if n:
        _kernel(name)((_blocks(n),), (THREADS,), (x, out, np.int32(n)))
    return out


def slab_powf(x, y):
    """glibc 2.39 ``powf`` over arrays, elementwise.

    ``y`` may be a scalar; it is broadcast to ``x``'s shape *before* the
    launch, because the kernel reads one exponent per element.
    """
    import cupy as cp

    x = _contiguous(x)
    y = _contiguous(cp.broadcast_to(cp.asarray(y, dtype=cp.float32), x.shape))
    out = cp.empty_like(x)
    n = int(x.size)
    if n:
        _kernel("nmp_slab_powf")((_blocks(n),), (THREADS,),
                                 (x, y, out, np.int32(n)))
    return out


def slab_expf(x):
    """glibc 2.39 ``expf`` over arrays, elementwise."""
    return _unary("nmp_slab_expf", x)


def slab_logf(x):
    """glibc 2.39 ``logf`` over arrays, elementwise."""
    return _unary("nmp_slab_logf", x)


def slab_tanhf(x):
    """glibc 2.39 ``tanhf`` over arrays, elementwise."""
    return _unary("nmp_slab_tanhf", x)


def slab_sqrtf(x):
    """IEEE-754 square root: correctly rounded, so CuPy's is glibc's."""
    import cupy as cp

    return cp.sqrt(cp.asarray(x, dtype=cp.float32))


def fmn(a, b):
    """``gpuwm.core.noahmp_energy._mn`` over arrays.

    ``a if a < b else b``.  On a tie -- including ``(-0.0, +0.0)`` -- this
    returns ``b``, which ``cupy.minimum`` does not.
    """
    import cupy as cp

    a = cp.asarray(a, dtype=cp.float32)
    b = cp.asarray(b, dtype=cp.float32)
    return cp.where(a < b, a, b)


def fmx(a, b):
    """``gpuwm.core.noahmp_energy._mx`` over arrays: ``a if a > b else b``."""
    import cupy as cp

    a = cp.asarray(a, dtype=cp.float32)
    b = cp.asarray(b, dtype=cp.float32)
    return cp.where(a > b, a, b)


def slab_fabsf(x):
    """Fortran ``ABS`` on ``REAL(4)``: a sign-bit clear, ``-0.0 -> +0.0``."""
    import cupy as cp

    return cp.abs(cp.asarray(x, dtype=cp.float32))


__all__ = [
    "fmn",
    "fmx",
    "slab_expf",
    "slab_fabsf",
    "slab_logf",
    "slab_powf",
    "slab_sqrtf",
    "slab_tanhf",
]
