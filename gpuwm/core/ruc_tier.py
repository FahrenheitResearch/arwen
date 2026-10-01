"""The RUC soil column's compile-time geometry tier.

``gpuwm/core/kernels/ruc.cu`` sizes every per-thread soil scratch array from
``RUC_NZS`` and selects its level table with the same macro.  This module is
the one place that decides what ``RUC_NZS`` a given soil geometry compiles
with, and it is the RUC analogue of the ``WPHI_MAX_LEV`` ladder that
:mod:`gpuwm.core.acoustic` owns for the implicit w''-phi'' solve.

WHY THIS IS ITS OWN MODULE, and not three functions in
:mod:`gpuwm.core.ruc_gpu` where the launchers that use them live:
``ruc_gpu`` imports CuPy at module scope, so importing it requires a CuPy
install.  The whole value of :func:`ruc_kernel_source` is that the string
NVRTC will receive can be digested, preprocessed and compiled to PTX on a
box with no CuPy and no card -- that is what makes the nine-level
bit-identity claim in ``tests/test_ruc_nzs_tier.py`` a measurement rather
than an assertion.  A tier helper that can only be imported next to a GPU
cannot carry that proof, so the tier lives here, beside the contract, and
``ruc_gpu`` imports it.  :mod:`gpuwm.core.kernels` is CuPy-free to import
for the same reason (its ``import cupy`` calls are inside the loaders).
"""

from __future__ import annotations

from gpuwm.core.kernels import (get_kernel, get_kernel_int_defines,
                                module_source, module_source_int_defines)
from gpuwm.core.ruc_contract import (NUM_SOIL_LAYERS,
                                     WRF_SUPPORTED_NUM_SOIL_LAYERS)

#: The RUC translation unit's name in :mod:`gpuwm.core.kernels`.
RUC_MODULE = "ruc"


def ruc_module_defines(nzs: int) -> tuple[tuple[str, int], ...]:
    """Integer defines the RUC module compiles with at this soil geometry.

    EMPTY at the shipped geometry.  That emptiness is the whole mechanism:
    it routes the launcher to the unspecialized loader, which assembles the
    same string it assembled before the ladder existed, so no nine-level run
    can see a different translation unit or a different manifest key.  The
    launcher branches HERE and nowhere else.  Mirrors
    :func:`gpuwm.core.acoustic.wphi_module_defines`.
    """
    nzs = int(nzs)
    if nzs not in WRF_SUPPORTED_NUM_SOIL_LAYERS:
        raise ValueError(
            f"RUC soil geometry {nzs} is not one of "
            f"{WRF_SUPPORTED_NUM_SOIL_LAYERS}")
    if nzs == NUM_SOIL_LAYERS:
        return ()
    return (("RUC_NZS", nzs),)


def ruc_kernel(func: str, nzs: int):
    """The RUC kernel ``func`` compiled at this geometry's tier.

    At the shipped geometry this is :func:`gpuwm.core.kernels.get_kernel` --
    the same call, on the same module, every untiered RUC launcher makes --
    so a nine-level process compiles exactly one RUC translation unit and it
    is the pre-ladder one.
    """
    defines = ruc_module_defines(nzs)
    if not defines:
        return get_kernel(RUC_MODULE, func)
    return get_kernel_int_defines(RUC_MODULE, func, defines)


def ruc_kernel_source(nzs: int) -> str:
    """The exact string NVRTC receives for RUC at ``nzs``.  CPU-only.

    Imports no CuPy and touches no device, so the identity of the nine-level
    translation unit is testable on any box.
    """
    defines = ruc_module_defines(nzs)
    if not defines:
        return module_source(RUC_MODULE)
    return module_source_int_defines(RUC_MODULE, defines)


# ---------------------------------------------------------------------------
# The fused RUC translation unit.
# ---------------------------------------------------------------------------

#: The fused column kernels' manifest name.
RUC_FUSED_MODULE = "ruc_fused"

#: The fused kernels' own sources, appended in this order after ``ruc.cu``.
#: They are ``.cuh`` fragments, not ``.cu`` modules: neither compiles alone,
#: because both call the leaf bodies ``ruc.cu`` defines.
RUC_FUSED_SOURCES = ("ruc_fused_sfctmp.cuh", "ruc_fused_driver.cuh")

#: ``ruc.cu``'s leaves are ``extern "C" __global__`` functions whose body
#: finds its column as ``blockIdx.x * blockDim.x + threadIdx.x`` and returns
#: past ``n``.  Compiled once more with ``__global__`` spelled ``__device__``,
#: the SAME text becomes a set of column functions a fused kernel calls from
#: the thread that owns that column, so the leaf arithmetic has one source
#: and ``ruc.cu`` does not change by a byte.  Every FP operation in it is an
#: explicit round-to-nearest intrinsic, so the calling context cannot
#: contract or reorder it.
#:
#: NVRTC does not honour ``#pragma push_macro``/``pop_macro`` (measured: a
#: kernel declared after the pop is still a device function), so the close
#: restores the spelling CUDA's host_defines.h gives ``__global__``.
_RUC_AS_DEVICE_OPEN = (
    "#undef __global__\n"
    "#define __global__ __device__\n")
_RUC_AS_DEVICE_CLOSE = (
    "\n#undef __global__\n"
    "#define __global__ __location__(global)\n")


def ruc_fused_source(nzs: int) -> str:
    """The exact string NVRTC receives for the fused RUC unit.  CPU-only.

    The preamble, then the tier define exactly where
    :func:`gpuwm.core.kernels.module_source_int_defines` places it for
    ``ruc.cu`` (before its text, so the ladder sees it), then ``ruc.cu``
    with ``__global__`` read as ``__device__``, then the fused sources.
    """
    from gpuwm.core.kernels import _ENCODING, _KDIR, _preamble

    defines = ruc_module_defines(nzs)
    prefix = "".join(f"#define {key} {value}\n" for key, value in defines)
    parts = [_preamble(), prefix, _RUC_AS_DEVICE_OPEN,
             (_KDIR / f"{RUC_MODULE}.cu").read_text(encoding=_ENCODING),
             _RUC_AS_DEVICE_CLOSE]
    parts += [(_KDIR / name).read_text(encoding=_ENCODING)
              for name in RUC_FUSED_SOURCES]
    return "".join(parts)


def _ruc_fused_module_key(nzs: int) -> str:
    from gpuwm.core.kernels import MODULE_KEY_ROOT

    defines = ruc_module_defines(nzs)
    key = f"{MODULE_KEY_ROOT}:{RUC_FUSED_MODULE}"
    if defines:
        key += "[" + ",".join(f"{k}={v}" for k, v in defines) + "]"
    return key


_RUC_FUSED_MODULES: dict[int, object] = {}


def ruc_fused_kernel(func: str, nzs: int):
    """A kernel of the fused RUC unit at this soil geometry.

    One compile per geometry per process, recorded in the kernel manifest
    under its own key like every other translation unit.
    """
    nzs = int(nzs)
    module = _RUC_FUSED_MODULES.get(nzs)
    if module is None:
        import cupy as cp

        from gpuwm.certify.kernel_manifest import record_module
        from gpuwm.core.kernels import _compile_observed

        source = ruc_fused_source(nzs)
        key = _ruc_fused_module_key(nzs)
        module = cp.RawModule(code=source, options=("-std=c++17",),
                              name_expressions=None)
        _compile_observed(module, key)
        record_module(key, source=source, options=("-std=c++17",),
                      module=module)
        _RUC_FUSED_MODULES[nzs] = module
    return module.get_function(func)
