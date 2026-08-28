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
