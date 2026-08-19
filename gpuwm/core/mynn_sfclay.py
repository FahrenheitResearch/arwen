"""CUDA execution wrapper for the pinned WRF v4.6.1 MYNN surface layer.

Only the option identities represented by
``gpuwm.core.mynn_surface.mynn_surface_layer_default`` are implemented here --
every defined ``isftcflx`` over water, with ``iz0tlnd=0``.
The surface-layer and PBL selectors follow WRF's own compatibility table. WRF's
surface driver also pairs that suite with Noah, RUC and Noah-MP; the constants
below make the exact exchange-field staging at those LSM seams explicit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cupy as cp
import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.mynn_surface import ISFTCFLX_DEFINED
from gpuwm.core.state import DTYPE


MYNN_SURFACE_INPUTS = (
    "u1", "v1", "t1", "qv1", "p1", "rho1", "dz1",
    "u2", "v2", "dz2", "psfc", "tsk", "pblh", "mavail",
    "hfx", "qfx", "znt", "qsfc", "ust", "xland", "snowh",
)

from gpuwm.core.physics_inventory import MYNN_SURFACE_OUTPUTS  # noqa: F401  (one home; re-exported here)


#: Fields MYNN stages for RUC and RUC leaves intact.  LSMRUC has FLHC/FLQC as
#: ``INTENT(IN)`` and no UST/CHS2/CQS2 argument (module_sf_ruclsm.F:219-230);
#: the surface driver recomputes CHS from FLHC after the call.
MYNN_RUC_EXCHANGE_HANDOFF = (
    "ust", "flhc", "flqc", "chs2", "cqs2",
)

#: Fields MYNN stages for Noah-MP and NOAHMPSCHEME never receives, so MYNN
#: retains ownership through the land-surface call
#: (module_surface_driver.F:3127-3181).
MYNN_NOAHMP_EXCHANGE_HANDOFF = (
    "ust", "chs", "chs2", "cqs2", "flhc", "flqc",
)

#: Both LSMs overwrite these values after MYNN has run.  The actual writers
#: also update scheme-specific state such as QSFC/ZNT.
MYNN_LSM_FLUX_WRITEBACK = ("tsk", "hfx", "qfx", "lh")

#: Final 2-m ownership belongs to SFCDIAGS_RUCLSM or Noah-MP's category/fraction
#: post-pass, not to MYNN's earlier diagnosis.
MYNN_POST_LSM_2M_WRITEBACK = ("t2", "q2", "th2")


@dataclass
class MynnSurfaceResult:
    """FP32 device outputs of one default MYNN surface-layer call."""

    regime: cp.ndarray
    zol: cp.ndarray
    rmol: cp.ndarray
    ust: cp.ndarray
    ustm: cp.ndarray
    mol: cp.ndarray
    psim: cp.ndarray
    psih: cp.ndarray
    chs: cp.ndarray
    chs2: cp.ndarray
    cqs2: cp.ndarray
    ch: cp.ndarray
    flhc: cp.ndarray
    flqc: cp.ndarray
    qgh: cp.ndarray
    qsfc: cp.ndarray
    hfx: cp.ndarray
    qfx: cp.ndarray
    lh: cp.ndarray
    u10: cp.ndarray
    v10: cp.ndarray
    th2: cp.ndarray
    t2: cp.ndarray
    q2: cp.ndarray
    gz1oz0: cp.ndarray
    wspd: cp.ndarray
    br: cp.ndarray
    ck: cp.ndarray
    cka: cp.ndarray
    cd: cp.ndarray
    cda: cp.ndarray
    wstar: cp.ndarray
    qstar: cp.ndarray
    cpm: cp.ndarray
    znt: cp.ndarray


_TPB = 128


def seed_mynn_surface_first_step(
    u1: cp.ndarray,
    v1: cp.ndarray,
    qv1: cp.ndarray,
    *,
    ust: cp.ndarray,
    mol: cp.ndarray,
    qsfc: cp.ndarray,
    qstar: cp.ndarray,
) -> None:
    """Apply WRF's ``SFCLAY_mynn`` ``itimestep==1`` seeding in place.

    ``module_sf_mynn.F:329-337`` runs this block in the wrapper, before
    ``SFCLAY1D_mynn`` is entered.  It belongs to the driver seam, not to the
    column solver, so the kernel never performs it.
    """

    for name, array in (("u1", u1), ("v1", v1), ("qv1", qv1), ("ust", ust),
                        ("mol", mol), ("qsfc", qsfc), ("qstar", qstar)):
        if array.shape != u1.shape or array.dtype != DTYPE:
            raise ValueError(
                f"seed_mynn_surface_first_step requires same-shape float32 "
                f"surface arrays; {name} is {array.shape}/{array.dtype}"
            )
    ust[...] = cp.maximum(
        DTYPE(0.04) * cp.sqrt(u1 * u1 + v1 * v1), DTYPE(0.001)
    )
    mol[...] = DTYPE(0.0)
    qsfc[...] = qv1 / (DTYPE(1.0) + qv1)
    qstar[...] = DTYPE(0.0)


def _validate_options(
    dx: float, itimestep: int, isfflx: int, isftcflx: int
) -> None:
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be positive and finite")
    if isinstance(itimestep, bool) or not isinstance(itimestep, int) \
            or itimestep < 1:
        raise ValueError("itimestep must be a positive integer")
    if isfflx not in (0, 1):
        raise ValueError("isfflx must be 0 or 1")
    # The kernel has no arm for ISFTCFLX=4 because WRF has no defined answer
    # for it: module_sf_mynn.F:680-702 assigns no z_t/z_q there.  Rejecting it
    # on the host is what keeps the device from reading a value the CPU
    # reference refuses to invent.
    if (not isinstance(isftcflx, (int, np.integer))
            or isinstance(isftcflx, bool)
            or isftcflx not in ISFTCFLX_DEFINED):
        raise ValueError(
            "isftcflx must be 0 (COARE 3.0 z0 and zt/zq), 1 (Davis z0, "
            "COARE 3.0 zt/zq), 2 (Davis z0, Garratt zt/zq) or 3 "
            f"(Taylor-Yelland z0, COARE 3.0 zt/zq); got {isftcflx}"
        )


def _surface_array(value, shape, name: str, default: float | None = None):
    if value is None:
        if default is None:
            raise TypeError(f"{name} is required")
        return cp.full(shape, default, dtype=DTYPE)
    array = cp.asarray(value, dtype=DTYPE)
    if array.shape != shape:
        try:
            array = cp.broadcast_to(array, shape)
        except ValueError as exc:
            raise ValueError(
                f"{name} shape {array.shape} is not broadcastable to "
                f"surface shape {shape}"
            ) from exc
    return cp.ascontiguousarray(array)


def _allocate_result(shape) -> MynnSurfaceResult:
    return MynnSurfaceResult(**{
        name: cp.empty(shape, dtype=DTYPE) for name in MYNN_SURFACE_OUTPUTS
    })


def launch_mynn_surface_layer(
    inputs: Mapping[str, cp.ndarray],
    mol: cp.ndarray,
    ustm: cp.ndarray,
    result: MynnSurfaceResult,
    *,
    dx: float = 3000.0,
    itimestep: int = 1,
    isfflx: int = 1,
    isftcflx: int = 0,
) -> None:
    """Launch the MYNN surface kernel into preallocated outputs."""

    _validate_options(dx, itimestep, isfflx, isftcflx)
    missing = [name for name in MYNN_SURFACE_INPUTS if name not in inputs]
    if missing:
        raise TypeError(f"missing MYNN surface inputs: {', '.join(missing)}")
    shape = inputs["u1"].shape
    arrays = tuple(inputs[name] for name in MYNN_SURFACE_INPUTS) + (mol, ustm)
    arrays += tuple(getattr(result, name) for name in MYNN_SURFACE_OUTPUTS)
    for array in arrays:
        if array.shape != shape or array.dtype != DTYPE \
                or not array.flags.c_contiguous:
            raise ValueError(
                "launch_mynn_surface_layer requires same-shape contiguous "
                "float32 surface arrays"
            )
    n = int(np.prod(shape))
    blocks = (n + _TPB - 1) // _TPB
    kernel = get_kernel("mynn_surface", "mynn_surface_column")
    kernel(
        (blocks,),
        (_TPB,),
        arrays + (
            DTYPE(dx), np.int32(itimestep), np.int32(isfflx),
            np.int32(isftcflx), np.int32(n),
        ),
    )


def mynn_surface_layer(
    values: Mapping[str, object],
    *,
    dx: float = 3000.0,
    itimestep: int = 1,
    isfflx: int = 1,
    isftcflx: int = 0,
    mol: object | None = None,
    ustm: object | None = None,
) -> MynnSurfaceResult:
    """Evaluate WRF MYNN surface physics on 2-D device fields."""

    _validate_options(dx, itimestep, isfflx, isftcflx)
    missing = [name for name in MYNN_SURFACE_INPUTS if name not in values]
    if missing:
        raise TypeError(f"missing MYNN surface inputs: {', '.join(missing)}")
    u1 = cp.ascontiguousarray(cp.asarray(values["u1"], dtype=DTYPE))
    if u1.ndim != 2:
        raise ValueError(
            f"MYNN surface inputs must be 2-D (ny,nx), got {u1.shape}"
        )
    shape = u1.shape
    inputs = {"u1": u1}
    for name in MYNN_SURFACE_INPUTS[1:]:
        inputs[name] = _surface_array(values[name], shape, name)
    mol_array = _surface_array(mol, shape, "mol", 0.0)
    ustm_array = _surface_array(
        values["ust"] if ustm is None else ustm, shape, "ustm"
    )
    result = _allocate_result(shape)
    launch_mynn_surface_layer(
        inputs, mol_array, ustm_array, result,
        dx=dx, itimestep=itimestep, isfflx=isfflx, isftcflx=isftcflx,
    )
    return result


__all__ = [
    "MYNN_LSM_FLUX_WRITEBACK",
    "MYNN_NOAHMP_EXCHANGE_HANDOFF",
    "MYNN_POST_LSM_2M_WRITEBACK",
    "MYNN_RUC_EXCHANGE_HANDOFF",
    "MYNN_SURFACE_INPUTS",
    "MYNN_SURFACE_OUTPUTS",
    "MynnSurfaceResult",
    "launch_mynn_surface_layer",
    "mynn_surface_layer",
    "seed_mynn_surface_first_step",
]
