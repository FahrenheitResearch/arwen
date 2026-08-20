"""MYJ (Mellor-Yamada-Janjic level 2.5) PBL, ``bl_pbl_physics = 2``.

CUDA launcher around :mod:`gpuwm.core.kernels.myjpbl`, the device mirror of
the float32 CPU authority ``gpuwm.verify.myj_ref.np_myjpbl_column``, itself
transcribed from the byte-frozen WRF v4.6.1 ``phys/module_bl_myjpbl.F``.

WHAT THIS SCHEME OWNS.  MYJ predicts TKE (``q**2 = 2*TKE``) with Janjic's
extension of Mellor-Yamada level 2.5, builds the mixing length from it,
forms the exchange coefficients, and then does its OWN implicit vertical
diffusion of heat, moisture, cloud water/ice and momentum -- so the scheme
returns finished TENDENCIES, not diffusivities for someone else to apply.
That is why its surface pairing is not negotiable: the lower boundary of
every one of those solves is ``AKHS``/``AKMS``/``THZ0``/``QZ0``/``UZ0``/
``VZ0``, which only the Eta surface layer produces.

COLUMN LIMIT.  The kernel keeps the whole column in per-thread local
arrays, as YSU's does, so it carries the same compile-time ceiling.  A
deeper column is refused here rather than silently truncated.

Conformance status: implemented-unverified.  ``tests/test_myj_port.py``
asserts CPU-vs-CUDA agreement and physical sanity; NO oracle comparison
against the WRF Fortran has been run.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import DTYPE
from gpuwm.physics_vertical_contract import (
    outside_vertical_bounds, refuse_vertical_levels)

_TPB = 128

#: Deepest column ``kernels/myjpbl.cu``'s per-thread arrays hold
#: (``MYJ_KMAX``).  Kept in step with the kernel by
#: ``tests/test_myj_port.py``.
MYJ_MAX_COLUMN_LEVELS = 128

#: Shallowest column the tridiagonal chain is defined on.  VDIFQ solves
#: over ``LMH-2`` interior rows (module_bl_myjpbl.F:1376-1403) and MIXLEN's
#: profile smoothing reads ``REL(K-1)``/``REL(K+1)``; four levels is the
#: floor at which every one of those slices is non-empty.
MYJ_MIN_COLUMN_LEVELS = 4

#: Full bottom-up ``(nz, ny, nx)`` inputs, in kernel argument order.
_COLUMN_INPUTS = ("dz", "u", "v", "t", "th", "exner", "qv", "qc", "qi", "p")
#: Read-only surface inputs, in kernel argument order.
_SURFACE_INPUTS = ("psfc", "ust", "tsk", "chklowq", "xland", "sice", "snow",
                   "akhs", "akms", "elflx", "uz0", "vz0")
#: WRF INOUT surface state MYJPBL rewrites (:542, :556-566, :852).
MYJ_PBL_INOUT = ("thz0", "qz0", "qsfc", "ct")
#: Bottom-up ``(nz, ny, nx)`` outputs, in kernel argument order.
MYJ_PBL_COLUMN_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                          "rqcblten", "rqiblten", "el_myj", "exch_h")
#: Surface outputs, in kernel argument order.  ``kpbl`` is int32.
MYJ_PBL_SURFACE_OUTPUTS = ("pblh", "kpbl", "mixht")


def launch_myj_pbl(columns, surface, state, tke, outputs, *,
                   dtturbl: float, flqi: bool) -> None:
    """Run one MYJ PBL step in place.

    ``tke`` is the scheme's carried TKE_MYJ column, read as ``2*TKE`` on
    entry and rewritten as ``0.5*q2`` on exit (module_bl_myjpbl.F:377,
    :462).  ``flqi`` selects WRF's ``PRESENT(QCI)`` arm: with ice the
    species stack is four rows and RQIBLTEN is produced, without it three
    and RQIBLTEN stays zero (:264-273).
    """
    shape = surface["psfc"].shape
    nz = int(columns["dz"].shape[0])
    bounds = (MYJ_MIN_COLUMN_LEVELS, MYJ_MAX_COLUMN_LEVELS)
    if outside_vertical_bounds(nz, bounds):
        raise refuse_vertical_levels(
            "MYJ PBL", bounds, nz,
            breakage=(
                "the kernel holds at most "
                f"{MYJ_MAX_COLUMN_LEVELS} levels per column (MYJ_KMAX in "
                "kernels/myjpbl.cu) -- raise the define and re-measure "
                "rather than truncating -- and below the floor its TKE "
                "tridiagonal solve has no interior rows."))
    column_arrays = []
    for name in _COLUMN_INPUTS:
        array = columns.get(name)
        if array is None:
            if name != "qi" or flqi:
                raise KeyError(f"the MYJ PBL requires a {name!r} column")
            # WRF's OPTIONAL QCI is absent: the kernel never reads the
            # pointer when flqi is false, but a null argument is not
            # something CuPy will marshal, so pass the qc column and let
            # the flag hold it out.
            array = columns["qc"]
        column_arrays.append(array)
    for array in (*column_arrays, tke,
                  *(outputs[name] for name in MYJ_PBL_COLUMN_OUTPUTS)):
        if (array.shape != (nz, *shape) or array.dtype != DTYPE
                or not array.flags.c_contiguous):
            raise ValueError(
                "launch_myj_pbl requires contiguous float32 "
                f"({nz}, {shape[0]}, {shape[1]}) columns")
    surface_arrays = ([surface[name] for name in _SURFACE_INPUTS]
                      + [state[name] for name in MYJ_PBL_INOUT]
                      + [outputs[name] for name in MYJ_PBL_SURFACE_OUTPUTS])
    for array in surface_arrays:
        expected = cp.int32 if array is outputs["kpbl"] else DTYPE
        if (array.shape != shape or array.dtype != expected
                or not array.flags.c_contiguous):
            raise ValueError("launch_myj_pbl requires same-shape contiguous "
                             "surface arrays (float32, kpbl int32)")
    n = int(np.prod(shape))
    blocks = (n + _TPB - 1) // _TPB
    kernel = get_kernel("myjpbl", "myjpbl_column")
    kernel((blocks,), (_TPB,),
           tuple(column_arrays) + (tke,)
           + tuple(surface[name] for name in _SURFACE_INPUTS)
           + tuple(state[name] for name in MYJ_PBL_INOUT)
           + tuple(outputs[name] for name in MYJ_PBL_COLUMN_OUTPUTS)
           + tuple(outputs[name] for name in MYJ_PBL_SURFACE_OUTPUTS)
           + (DTYPE(dtturbl), np.int32(bool(flqi)), np.int32(nz),
              np.int32(n)))


def myj_pbl_step(columns, surface, state, tke, *,
                 dtturbl: float, flqi: bool) -> dict:
    """Allocate MYJ's output set, run one step, and return the outputs.

    The returned dict carries the WRF names the kernel writes plus the
    ``du/dv/dtheta/dqv/dqc/dqi`` aliases :func:`couple_ysu_tendencies`
    consumes -- the SAME device arrays under both keys, not copies, so a
    caller cannot mass-couple one spelling while reading the other.
    """
    shape = surface["psfc"].shape
    nz = int(columns["dz"].shape[0])
    outputs = {name: cp.empty((nz, *shape), dtype=DTYPE)
               for name in MYJ_PBL_COLUMN_OUTPUTS}
    outputs["pblh"] = cp.empty(shape, dtype=DTYPE)
    outputs["mixht"] = cp.empty(shape, dtype=DTYPE)
    outputs["kpbl"] = cp.empty(shape, dtype=cp.int32)
    launch_myj_pbl(columns, surface, state, tke, outputs,
                   dtturbl=dtturbl, flqi=flqi)
    outputs.update(
        du=outputs["rublten"], dv=outputs["rvblten"],
        dtheta=outputs["rthblten"], dqv=outputs["rqvblten"],
        dqc=outputs["rqcblten"],
        # WRF's phy_bl_ten adds RQIBLTEN exactly when the moist set
        # carries ice (module_physics_addtendc.F, IF(F_QI)); without ice
        # MYJ mixed three species and the row is a published zero, so it
        # is withheld rather than coupled as a real ice tendency.
        dqi=outputs["rqiblten"] if flqi else None)
    if outputs["dqi"] is None:
        del outputs["dqi"]
    return outputs


def validate_myj_pbl_outputs(outputs) -> str | None:
    """Name the first non-finite MYJ tendency field, or ``None``.

    Host-side and deliberately cheap: the four tendency planes plus the
    published TKE-derived diagnostics are what a downstream integrator
    consumes, and a NaN in any of them must name its field rather than
    silently advance the state.  Unlike Shin-Hong, MYJ has no known WRF
    0/0 of its own, so a NaN here is a defect until an oracle campaign
    proves otherwise.
    """
    for name in ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                 "rqiblten", "exch_h", "el_myj", "pblh", "mixht"):
        array = outputs[name]
        if not bool(cp.all(cp.isfinite(array))):
            return name
    return None
